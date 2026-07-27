from __future__ import annotations

import json
import re

import numpy as np

from mindset.methods.base import ConversationEvent
from mindset.methods.mindset import MindsetConfig, MindsetMemory


class TokenEncoder:
    vocabulary = (
        "live", "lived", "moved", "delhi", "pune", "prefer", "tea", "coffee",
        "home", "work", "garden", "book", "novel", "author", "jazz", "hiking",
    )

    def encode(self, texts, **_kwargs):
        rows = []
        for text in texts:
            tokens = set(re.findall(r"[a-z]+", text.casefold()))
            vector = np.asarray(
                [float(token in tokens) for token in self.vocabulary],
                dtype=np.float32,
            )
            if not vector.any():
                vector[0] = 0.01
            rows.append(vector / np.linalg.norm(vector))
        return np.asarray(rows)


class KeywordReranker:
    def predict(self, pairs, **_kwargs):
        return np.asarray(
            [
                sum(token in document.casefold() for token in query.casefold().split())
                for query, document in pairs
            ],
            dtype=np.float32,
        )


def event(index: int, text: str, *, speaker: str = "Sam") -> ConversationEvent:
    return ConversationEvent(
        f"e{index}", "c1", f"s{index}", f"2024-01-{index:02d}", speaker, text
    )


def method(config: MindsetConfig | None = None) -> MindsetMemory:
    return MindsetMemory(
        config or MindsetConfig(retrieval_mode="legacy"), encoder=TokenEncoder()
    )


def test_ingestion_is_local_extractive_and_provenance_preserving() -> None:
    memory = method()
    memory.ingest([event(1, "I live in Delhi"), event(2, "I live in Delhi")])

    assert memory.telemetry()["llm_ingestion_calls"] == 0
    assert memory.telemetry()["operation_counts"] == {"new": 1, "reinforce": 1}
    schema = next(iter(memory.schemas.values()))
    assert schema.medoid_episode_id in {"e1", "e2"}
    assert schema.support_episode_ids == ["e1", "e2"]
    assert memory.episodes[schema.medoid_episode_id].text == "I live in Delhi"
    assert memory.validate_invariants()["valid"] is True


def test_explicit_temporal_change_supersedes_without_losing_history() -> None:
    memory = method()
    memory.ingest([event(1, "I live in Delhi"), event(2, "I moved to Pune")])

    assert memory.operation_log.all()[-1].operation == "supersede"
    assert len([item for item in memory.schemas.values() if item.status == "historical"]) == 1
    assert len([item for item in memory.schemas.values() if item.status == "active"]) == 1
    historical = memory.retrieve("Where did Sam live before?", top_k=2)
    current = memory.retrieve("Where does Sam currently live?", top_k=2)
    assert any("Delhi" in hit.text for hit in historical)
    assert any("Pune" in hit.text for hit in current)


def test_contextual_coexistence_splits_instead_of_overwriting() -> None:
    memory = method()
    memory.ingest([event(1, "I prefer tea at home"), event(2, "I prefer coffee at work")])

    assert memory.operation_log.all()[-1].operation == "split"
    assert len([item for item in memory.schemas.values() if item.status == "active"]) == 2
    assert not [item for item in memory.schemas.values() if item.status == "historical"]


def test_hysteresis_requires_recurrence_for_weak_state_flip() -> None:
    memory = method()
    memory.ingest([event(1, "I live in Delhi"), event(2, "I live in Pune")])
    assert memory.operation_log.all()[-1].operation == "reinforce"
    assert memory.operation_log.all()[-1].stored_as_exception

    memory.ingest([event(3, "I live in Pune")])
    assert memory.operation_log.all()[-1].operation == "supersede"

    no_hysteresis = method(
        MindsetConfig(ablation="no_hysteresis", retrieval_mode="legacy")
    )
    no_hysteresis.ingest([event(1, "I live in Delhi"), event(2, "I live in Pune")])
    assert no_hysteresis.operation_log.all()[-1].operation == "supersede"


def test_energy_log_contains_every_candidate_and_mathematical_term() -> None:
    memory = method()
    memory.ingest([event(1, "I live in Delhi"), event(2, "I moved to Pune")])
    record = memory.operation_log.all()[-1]

    assert set(record.candidate_energies) == {"reinforce", "supersede", "split", "new"}
    assert set(record.energy_terms["reinforce"]) == {
        "distortion", "contradiction", "historical_damage", "fragmentation",
        "inconsistency", "total",
    }
    assert record.signals["temporal_potential"] > record.signals["contextual_potential"]


def test_fragmentation_ablation_removes_only_fragmentation_weight() -> None:
    full = MindsetConfig()
    ablated = MindsetConfig(ablation="no_fragmentation_penalty")
    assert full.energy_weights()["fragmentation"] > 0
    assert ablated.energy_weights()["fragmentation"] == 0
    assert {
        key: value for key, value in full.energy_weights().items() if key != "fragmentation"
    } == {
        key: value for key, value in ablated.energy_weights().items() if key != "fragmentation"
    }


def test_retrieval_returns_only_raw_source_episodes_under_exact_budget() -> None:
    memory = method()
    memory.ingest(
        [
            event(1, "I read a novel in Delhi"),
            event(2, "The book author was fascinating"),
            event(3, "The garden is quiet"),
        ]
    )
    hits = memory.retrieve("Which book or novel did Sam read?", top_k=2)

    assert len(hits) == 2
    assert all(len(hit.source_event_ids) == 1 for hit in hits)
    assert all(hit.text.startswith("[2024-") for hit in hits)
    assert all("reranker_model" not in hit.metadata for hit in hits)
    assert memory.telemetry()["llm_retrieval_calls"] == 0


def test_no_provenance_expansion_is_executable() -> None:
    memory = method(
        MindsetConfig(ablation="no_provenance_expansion", retrieval_mode="legacy")
    )
    memory.ingest([event(1, "I read a novel"), event(2, "The book author was fascinating")])
    hits = memory.retrieve("book author", top_k=2)
    assert all(hit.memory_type == "episodic" for hit in hits)


def test_snapshot_is_auditable_and_contains_no_generated_schema_text(tmp_path) -> None:
    memory = method()
    memory.ingest([event(1, "I live in Delhi"), event(2, "I moved to Pune")])
    path = memory.snapshot(tmp_path / "energy.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["method_version"] == "free-energy-memory-v1"
    assert payload["llm_calls_during_ingestion"] == 0
    assert all("claim" not in schema and "summary" not in schema for schema in payload["schemas"])
    assert all(schema["medoid_episode_id"] for schema in payload["schemas"])


def test_local_rerank_pool_is_normalized_and_returns_only_raw_episodes() -> None:
    memory = MindsetMemory(
        MindsetConfig(retrieval_mode="query_adaptive"),
        encoder=TokenEncoder(),
        local_reranker=KeywordReranker(),
    )
    memory.ingest(
        [
            event(1, "I read a novel in Delhi"),
            event(2, "The book author was fascinating"),
            event(3, "The garden is quiet"),
        ]
    )
    pool = memory.retrieve_candidate_pool("Which book did Sam read?")
    hits = memory.select_candidate_pool(
        "Which book did Sam read?", pool, mode="query_adaptive", top_k=2
    )
    assert pool
    assert all(
        set(item.components)
        == {"cross_encoder", "dense", "bm25", "schema", "entity", "temporal", "validity"}
        for item in pool
    )
    assert all(
        0.0 <= value <= 1.0 for item in pool for value in item.components.values()
    )
    assert all(hit.memory_type == "locally_reranked_raw_episode" for hit in hits)
    assert all("fes-" not in hit.text and "energy" not in hit.text.casefold() for hit in hits)
