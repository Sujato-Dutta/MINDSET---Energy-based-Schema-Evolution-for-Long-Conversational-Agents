from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from mindset.baselines.event_rag import render_event
from mindset.methods.base import ConversationEvent, MemoryHit
from mindset.retrieval.hybrid import BM25Index, ranked_indices, reciprocal_rank_fusion


class HybridRAGIndex:
    """Local dense+BM25 retrieval with reciprocal-rank fusion."""

    method_name = "hybrid_rag"

    def __init__(
        self,
        events: Sequence[ConversationEvent],
        encoder: Any,
        *,
        candidate_k: int = 24,
        rank_constant: int = 60,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if rank_constant <= 0:
            raise ValueError("rank_constant must be positive")
        self.events = list(events)
        self.encoder = encoder
        self.candidate_k = candidate_k
        self.rank_constant = rank_constant
        self.texts = [render_event(event) for event in self.events]
        self.bm25 = BM25Index(self.texts)
        self.vectors = (
            np.asarray(
                encoder.encode(
                    self.texts,
                    batch_size=64,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
            if self.texts
            else np.empty((0, 0), dtype=np.float32)
        )

    def retrieve(self, query: str, top_k: int = 10) -> list[MemoryHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self.events:
            return []
        query_vector = np.asarray(
            self.encoder.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0],
            dtype=np.float32,
        )
        dense_scores = self.vectors @ query_vector
        lexical_scores = self.bm25.scores(query)
        pool_size = min(len(self.events), max(top_k, self.candidate_k))
        dense_ranking = ranked_indices(dense_scores, pool_size)
        lexical_ranking = ranked_indices(lexical_scores, pool_size)
        fused = reciprocal_rank_fusion(
            (dense_ranking, lexical_ranking), rank_constant=self.rank_constant
        )
        ordered = sorted(fused, key=lambda index: (-fused[index], index))[:top_k]
        return [
            MemoryHit(
                memory_id=self.events[index].event_id,
                text=self.texts[index],
                score=float(fused[index]),
                source_event_ids=(self.events[index].event_id,),
                memory_type="hybrid_episodic",
                metadata={
                    **self.events[index].metadata,
                    "dense_score": float(dense_scores[index]),
                    "bm25_score": float(lexical_scores[index]),
                    "fusion": "rrf",
                    "rank_constant": self.rank_constant,
                },
            )
            for index in ordered
        ]

    def telemetry(self) -> dict[str, Any]:
        return {
            "event_count": len(self.events),
            "candidate_k_per_channel": self.candidate_k,
            "fusion": "reciprocal_rank_fusion",
            "rank_constant": self.rank_constant,
            "write_api_calls": 0,
            "retrieval_api_calls": 0,
        }
