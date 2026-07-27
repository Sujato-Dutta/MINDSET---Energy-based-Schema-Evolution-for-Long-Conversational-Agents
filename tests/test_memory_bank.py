from __future__ import annotations

import numpy as np

from mindset.baselines.memory_bank import MemoryBankReproduction
from mindset.methods.base import ConversationEvent


class FakeEncoder:
    def encode(self, texts, **kwargs):
        rows = []
        for text in texts:
            lowered = str(text).lower()
            rows.append([
                float("python" in lowered or "code" in lowered),
                float("music" in lowered or "guitar" in lowered),
                float("user" in lowered or "personality" in lowered),
                1.0,
            ])
        values = np.asarray(rows, dtype=np.float32)
        if kwargs.get("normalize_embeddings"):
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            values = values / np.maximum(norms, 1e-8)
        return values


def events() -> list[ConversationEvent]:
    return [
        ConversationEvent(
            "e1", "case", "s1", "2024-01-01", "user", "I like Python code",
            {"memory_bank_date": "2024-01-01"},
        ),
        ConversationEvent(
            "e2", "case", "s1", "2024-01-02", "user", "I play guitar",
            {"memory_bank_date": "2024-01-02"},
        ),
    ]


def test_memory_bank_builds_daily_and_global_memory_without_api_calls():
    method = MemoryBankReproduction(FakeEncoder(), top_k=4, forgetting=False, random_seed=0)
    method.ingest(events())

    telemetry = method.telemetry()
    assert telemetry["memory_items"] == 6
    assert telemetry["daily_summary_items"] == 2
    assert telemetry["summary_calls"] == 0
    assert len(method.retrieve("What code does the user like?", top_k=4)) == 4


def test_memory_bank_summarizer_receives_official_prompt_roles():
    seen: list[tuple[str, str]] = []

    def summarize(kind: str, prompt: str) -> str:
        seen.append((kind, prompt))
        return f"summary for {kind}"

    method = MemoryBankReproduction(FakeEncoder(), summarizer=summarize, top_k=4)
    method.ingest(events())
    kinds = [kind for kind, _ in seen]
    assert kinds == [
        "daily_event", "daily_personality",
        "daily_event", "daily_personality",
        "global_event_summary", "global_personality",
    ]
    assert "Please summarize the following dialogue" in seen[0][1]
    assert "personality traits and emotions" in seen[1][1]
    assert method.telemetry()["summary_calls"] == 2
    assert method.telemetry()["global_summary_calls"] == 1


def test_memory_bank_recall_strength_increases_and_provenance_is_preserved():
    method = MemoryBankReproduction(FakeEncoder(), top_k=4, random_seed=0)
    method.ingest(events())
    before = next(item.memory_strength for item in method.items if item.memory_type == "memory_bank_turn")
    hits = method.retrieve("Python code", top_k=4)
    after = next(item.memory_strength for item in method.items if item.memory_id == hits[0].memory_id)
    assert after >= before
    assert hits[0].source_event_ids
    assert "memory_strength" in hits[0].metadata
    assert hits[0].metadata["paper"] == "MemoryBank"
