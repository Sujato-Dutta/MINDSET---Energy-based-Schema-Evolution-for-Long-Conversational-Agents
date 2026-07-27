from __future__ import annotations

import numpy as np
import pytest

from mindset.baselines.hybrid_rag import HybridRAGIndex
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
                    float("flight" in lowered),
                ],
                dtype=np.float32,
            )
            norm = float(np.linalg.norm(vector))
            values.append(vector / norm if norm else vector)
        return np.asarray(values)


def event(identifier: str, text: str) -> ConversationEvent:
    return ConversationEvent(identifier, "c", "s", "2025-01-01", "A", text)


def test_hybrid_rag_fuses_dense_and_lexical_channels() -> None:
    index = HybridRAGIndex(
        [
            event("e1", "The garden needs water."),
            event("e2", "A jazz flight was booked."),
            event("e3", "The concert featured jazz."),
        ],
        KeywordEncoder(),
        candidate_k=3,
    )
    hits = index.retrieve("jazz concert", top_k=2)
    assert [hit.source_event_ids for hit in hits] == [("e3",), ("e2",)]
    assert all(hit.memory_type == "hybrid_episodic" for hit in hits)
    assert index.telemetry()["retrieval_api_calls"] == 0


def test_hybrid_rag_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="candidate_k"):
        HybridRAGIndex([], KeywordEncoder(), candidate_k=0)
