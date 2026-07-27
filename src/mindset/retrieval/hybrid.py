from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np

from mindset.memory.energy import tokenize, unit_vector


class BM25Index:
    """Small deterministic BM25 implementation over immutable episode text."""

    def __init__(
        self,
        documents: Sequence[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(document) for document in documents]
        self.term_counts = [Counter(items) for items in self.tokens]
        self.lengths = np.asarray([len(items) for items in self.tokens], dtype=np.float32)
        self.average_length = float(np.mean(self.lengths)) if len(self.lengths) else 1.0
        document_frequency: Counter[str] = Counter()
        for items in self.tokens:
            document_frequency.update(set(items))
        count = max(1, len(self.tokens))
        self.idf = {
            token: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def scores(self, query: str) -> np.ndarray:
        if not self.tokens:
            return np.empty(0, dtype=np.float32)
        result = np.zeros(len(self.tokens), dtype=np.float32)
        for token in tokenize(query):
            idf = self.idf.get(token)
            if idf is None:
                continue
            for index, counts in enumerate(self.term_counts):
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * self.lengths[index] / max(self.average_length, 1.0)
                )
                result[index] += idf * frequency * (self.k1 + 1.0) / denominator
        return result


def normalize_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if not values.size:
        return values
    low, high = float(np.min(values)), float(np.max(values))
    if high - low < 1e-9:
        return np.zeros_like(values) if high == 0 else np.ones_like(values)
    return (values - low) / (high - low)


def ranked_indices(scores: np.ndarray, limit: int | None = None) -> list[int]:
    ordered = np.argsort(-np.asarray(scores), kind="stable")
    if limit is not None:
        ordered = ordered[:limit]
    return [int(index) for index in ordered]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    *,
    rank_constant: int = 60,
) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, identifier in enumerate(ranking, start=1):
            fused[identifier] = fused.get(identifier, 0.0) + 1.0 / (rank_constant + rank)
    return fused


def maximal_marginal_relevance(
    candidates: Sequence[int],
    relevance: dict[int, float],
    vectors: np.ndarray,
    *,
    top_k: int,
    relevance_weight: float = 0.82,
) -> list[int]:
    if top_k <= 0:
        return []
    remaining = list(dict.fromkeys(int(item) for item in candidates))
    selected: list[int] = []
    while remaining and len(selected) < top_k:
        best = max(
            remaining,
            key=lambda item: (
                relevance_weight * relevance.get(item, 0.0)
                - (1.0 - relevance_weight) * max(
                    (
                        float(np.dot(unit_vector(vectors[item]), unit_vector(vectors[chosen])))
                        for chosen in selected
                    ),
                    default=0.0,
                ),
                relevance.get(item, 0.0),
                -item,
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return selected
