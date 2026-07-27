"""Common, provenance-preserving answer-evidence normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from mindset.api.tokenizer import TokenCounter
from mindset.methods.base import MemoryHit


@dataclass(frozen=True)
class EvidenceBudget:
    candidate_k: int = 20
    chunk_size_tokens: int = 768
    chunk_overlap_tokens: int = 96
    max_chunks: int = 8
    max_tokens: int = 8000
    max_chunks_per_memory: int = 3

    def __post_init__(self) -> None:
        if self.candidate_k <= 0 or self.chunk_size_tokens <= 0:
            raise ValueError("evidence candidate and chunk sizes must be positive")
        if not 0 <= self.chunk_overlap_tokens < self.chunk_size_tokens:
            raise ValueError("chunk overlap must be smaller than chunk size")
        if self.max_chunks <= 0 or self.max_tokens <= 0 or self.max_chunks_per_memory <= 0:
            raise ValueError("evidence limits must be positive")


@dataclass(frozen=True)
class _Passage:
    source: MemoryHit
    text: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    passage_index: int
    token_count: int


def _grounding_text(hit: MemoryHit) -> str:
    if hit.memory_type.startswith("schema_") and "\nGrounding:\n" in hit.text:
        return hit.text.split("\nGrounding:\n", 1)[1]
    return hit.text


def _passages(text: str, hit: MemoryHit, counter: TokenCounter, budget: EvidenceBudget) -> list[_Passage]:
    offsets = counter.offsets(text)
    if offsets:
        step = budget.chunk_size_tokens - budget.chunk_overlap_tokens
        output: list[_Passage] = []
        for index, start_token in enumerate(range(0, len(offsets), step)):
            end_token = min(len(offsets), start_token + budget.chunk_size_tokens)
            char_start = offsets[start_token][0]
            char_end = offsets[end_token - 1][1]
            chunk = text[char_start:char_end]
            output.append(
                _Passage(
                    source=hit,
                    text=chunk,
                    char_start=char_start,
                    char_end=char_end,
                    token_start=start_token,
                    token_end=end_token,
                    passage_index=index,
                    token_count=end_token - start_token,
                )
            )
            if end_token >= len(offsets):
                break
        return output

    # Fail-closed fallback: whitespace windows are bounded by the byte safety
    # estimator and retain exact character offsets for provenance.
    words = list(__import__("re").finditer(r"\S+", text))
    if not words:
        return []
    step = budget.chunk_size_tokens - budget.chunk_overlap_tokens
    output = []
    for index, start_word in enumerate(range(0, len(words), step)):
        end_word = min(len(words), start_word + budget.chunk_size_tokens)
        char_start = words[start_word].start()
        char_end = words[end_word - 1].end()
        output.append(
            _Passage(
                source=hit,
                text=text[char_start:char_end],
                char_start=char_start,
                char_end=char_end,
                token_start=start_word,
                token_end=end_word,
                passage_index=index,
                token_count=end_word - start_word,
            )
        )
        if end_word >= len(words):
            break
    return output


def bound_evidence(
    hits: Sequence[MemoryHit],
    query: str,
    encoder: Any,
    *,
    budget: EvidenceBudget,
    tokenizer: TokenCounter | None = None,
) -> list[MemoryHit]:
    """Chunk, dense-rerank, and greedily select evidence under one budget."""
    if not hits:
        return []
    counter = tokenizer or TokenCounter()
    passages: list[_Passage] = []
    for hit in hits:
        text = _grounding_text(hit)
        passages.extend(_passages(text, hit, counter, budget))
    if not passages:
        return []

    texts = [query] + [passage.text for passage in passages]
    vectors = np.asarray(
        encoder.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    scores = vectors[1:] @ vectors[0]
    ranked = sorted(
        zip(passages, scores.tolist(), strict=True),
        key=lambda item: (-float(item[1]), item[0].source.memory_id, item[0].passage_index),
    )

    selected: list[MemoryHit] = []
    selected_tokens = 0
    per_memory: dict[str, int] = {}
    for passage, score in ranked:
        source_id = passage.source.memory_id
        if per_memory.get(source_id, 0) >= budget.max_chunks_per_memory:
            continue
        if len(selected) >= budget.max_chunks:
            break
        if selected_tokens + passage.token_count > budget.max_tokens:
            continue
        provenance = (
            f"[source_memory_id={source_id}; passage={passage.passage_index}; "
            f"char_offset={passage.char_start}:{passage.char_end}; "
            f"token_offset={passage.token_start}:{passage.token_end}]"
        )
        metadata = {
            **passage.source.metadata,
            "bounded_evidence": True,
            "original_memory_id": source_id,
            "passage_index": passage.passage_index,
            "char_offset_start": passage.char_start,
            "char_offset_end": passage.char_end,
            "token_offset_start": passage.token_start,
            "token_offset_end": passage.token_end,
            "passage_token_count": passage.token_count,
            "selection_score": float(score),
            "tokenizer": counter.name,
        }
        selected.append(
            MemoryHit(
                memory_id=f"{source_id}#passage-{passage.passage_index:03d}",
                text=f"{provenance}\n{passage.text}",
                score=float(score),
                source_event_ids=passage.source.source_event_ids,
                memory_type=f"{passage.source.memory_type}_passage",
                metadata=metadata,
            )
        )
        selected_tokens += passage.token_count
        per_memory[source_id] = per_memory.get(source_id, 0) + 1
    return selected
