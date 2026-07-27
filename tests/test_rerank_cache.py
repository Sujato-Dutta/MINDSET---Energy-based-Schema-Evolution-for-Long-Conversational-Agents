from __future__ import annotations

import numpy as np

from mindset.methods.rerank_cache import CachedReranker


class FakeReranker:
    def __init__(self) -> None:
        self.calls = []

    def predict(self, pairs, **_kwargs):
        self.calls.append(list(pairs))
        return np.asarray([len(query) + len(document) for query, document in pairs])


def test_rerank_scores_are_persisted_and_deduplicated(tmp_path) -> None:
    raw = FakeReranker()
    cached = CachedReranker(raw, "model", tmp_path / "scores.sqlite")
    pairs = [("q", "one"), ("q", "one"), ("q", "two")]
    first = cached.predict(pairs)
    second = cached.predict(pairs)
    assert first.tolist() == second.tolist() == [4.0, 4.0, 4.0]
    assert len(raw.calls) == 1
    assert len(raw.calls[0]) == 2
