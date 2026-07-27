from __future__ import annotations

import json

import numpy as np
import pytest

from mindset.methods.base import ConversationEvent
from mindset.methods.cache import write_json_once
from mindset.methods.nemori import NemoriFaithfulReproduction


class TokenEncoder:
    vocabulary = ("cat", "dog", "pune", "tea", "quantum", "garden")

    def encode(self, texts, **kwargs):
        result = []
        for text in texts:
            lowered = text.lower()
            vector = np.asarray([float(word in lowered) for word in self.vocabulary], dtype=np.float32)
            if not vector.any():
                vector[0] = 0.01
            result.append(vector / np.linalg.norm(vector))
        return np.asarray(result)


def event(identifier: str, text: str, session: str = "s1") -> ConversationEvent:
    return ConversationEvent(identifier, "c1", session, f"2024-01-0{identifier[-1]}", "Sam", text)


def test_nemori_aligns_boundaries_and_preserves_provenance() -> None:
    method = NemoriFaithfulReproduction(TokenEncoder(), boundary_threshold=0.2)
    method.ingest(
        [
            event("e1", "My cat is in the garden"),
            event("e2", "The cat likes the garden"),
            event("e3", "By the way quantum work is difficult"),
        ]
    )
    assert len(method.episodes) == 2
    assert method.episodes[0].source_event_ids == ("e1", "e2")
    assert "[2024-01-01] Sam" in method.episodes[0].narrative


def test_nemori_predict_calibrate_and_unified_retrieval() -> None:
    method = NemoriFaithfulReproduction(TokenEncoder(), prediction_gap_threshold=0.2)
    method.ingest([event("e1", "My cat is in the garden")])
    method.ingest([event("e2", "I prefer tea in Pune", session="s2")])
    assert method.semantic_memories
    hits = method.retrieve("tea Pune", top_k=3)
    assert hits[0].source_event_ids == ("e2",)
    assert {hit.memory_type for hit in hits} <= {"episodic", "semantic"}


def test_json_artifacts_are_immutable(tmp_path) -> None:
    path = tmp_path / "artifact.json"
    write_json_once(path, {"b": 2, "a": 1})
    assert json.loads(path.read_text()) == {"a": 1, "b": 2}
    write_json_once(path, {"a": 1, "b": 2})
    with pytest.raises(FileExistsError):
        write_json_once(path, {"a": 9})
