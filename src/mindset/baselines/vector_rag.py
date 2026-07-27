from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from mindset.datasets.locomo import LoCoMoSample
from mindset.utils.hashing import stable_hash


@dataclass(frozen=True)
class TextChunk:
    turn_id: str
    text: str


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[TextChunk]
    scores: list[float]
    latency_seconds: float


def conversation_chunks(sample: LoCoMoSample) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for session in sample.sessions:
        date = session.date_time or "date unknown"
        for turn in session.turns:
            chunks.append(
                TextChunk(
                    turn_id=turn.dia_id,
                    text=f"[{date}] {turn.speaker}: {turn.text}",
                )
            )
    return chunks


class VectorRAGIndex:
    """A flat, normalized sentence-embedding index with a persistent local cache."""

    def __init__(self, chunks: list[TextChunk], embeddings: np.ndarray, encoder: object) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        self.encoder = encoder

    @classmethod
    def build(
        cls,
        sample: LoCoMoSample,
        encoder: object,
        embedding_model: str,
        cache_dir: str | Path,
    ) -> "VectorRAGIndex":
        chunks = conversation_chunks(sample)
        fingerprint = stable_hash(
            {
                "embedding_model": embedding_model,
                "sample_id": sample.sample_id,
                "chunks": [(chunk.turn_id, chunk.text) for chunk in chunks],
            }
        )
        cache_path = Path(cache_dir) / f"{sample.sample_id}_{fingerprint[:16]}.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            embeddings = np.load(cache_path)["embeddings"]
        else:
            embeddings = encoder.encode(
                [chunk.text for chunk in chunks],
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)
            np.savez_compressed(cache_path, embeddings=embeddings)
        if embeddings.shape[0] != len(chunks):
            raise ValueError(f"Embedding cache mismatch for {sample.sample_id}")
        return cls(chunks, embeddings, encoder)

    def retrieve(self, query: str, top_k: int) -> RetrievalResult:
        started = perf_counter()
        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = self.embeddings @ query_embedding
        indices = np.argsort(-scores, kind="stable")[:top_k]
        return RetrievalResult(
            chunks=[self.chunks[int(index)] for index in indices],
            scores=[float(scores[int(index)]) for index in indices],
            latency_seconds=perf_counter() - started,
        )
