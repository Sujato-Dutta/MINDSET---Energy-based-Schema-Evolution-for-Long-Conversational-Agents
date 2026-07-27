from __future__ import annotations

import numpy as np

from mindset.baselines.lightmem import LightMemACL2026Reproduction
from mindset.methods.base import ConversationEvent


class KeywordEncoder:
    def encode(self, texts, **kwargs):
        values = []
        for text in texts:
            lowered = text.casefold()
            vector = np.asarray(
                [
                    float("jazz" in lowered),
                    float("garden" in lowered),
                    float("paris" in lowered),
                    0.01,
                ],
                dtype=np.float32,
            )
            values.append(vector / np.linalg.norm(vector))
        return np.asarray(values)


class SemanticSelector:
    def predict(self, pairs, **kwargs):
        return np.asarray(
            [
                2.0 * float("paris" in passage.casefold())
                + float("jazz" in passage.casefold())
                for _, passage in pairs
            ],
            dtype=np.float32,
        )


def event(identifier: str, text: str) -> ConversationEvent:
    return ConversationEvent(identifier, "person-1", "s", "2025-01-01", "A", text)


def test_lightmem_preserves_fixed_two_stage_budget_and_provenance() -> None:
    method = LightMemACL2026Reproduction(KeywordEncoder(), SemanticSelector())
    method.ingest(
        [
            event("e1", "The garden needs water."),
            event("e2", "A jazz show happened in Paris."),
            event("e3", "A local jazz band rehearsed."),
            event("e4", "The Paris trip was delayed."),
        ]
    )
    hits = method.retrieve("What happened in Paris?", top_k=2)
    assert len(hits) == 2
    assert hits[0].source_event_ids in {("e2",), ("e4",)}
    assert all(hit.metadata["coarse_budget"] == 4 for hit in hits)
    assert all(hit.memory_type == "lightmem_mtm" for hit in hits)
    telemetry = method.telemetry()
    assert telemetry["reproduction_label"] == "paper_guided_cpu_local_reproduction"
    assert telemetry["write_api_calls"] == 0
    assert telemetry["retrieval_api_calls"] == 0
    assert telemetry["ltm_nodes"] > 0


def test_lightmem_enforces_mtm_capacity() -> None:
    method = LightMemACL2026Reproduction(
        KeywordEncoder(), SemanticSelector(), mtm_capacity=2
    )
    method.ingest([event("e1", "old"), event("e2", "middle"), event("e3", "new")])
    assert [item.source_event_id for item in method.mtm] == ["e2", "e3"]
