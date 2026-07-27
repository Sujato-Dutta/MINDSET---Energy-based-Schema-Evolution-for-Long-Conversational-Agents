from __future__ import annotations

from dataclasses import dataclass

from mindset.baselines.vector_rag import TextChunk, conversation_chunks
from mindset.datasets.locomo import LoCoMoSample


@dataclass(frozen=True)
class LongContextPacket:
    text: str
    included_turn_ids: tuple[str, ...]
    omitted_turn_count: int
    truncation_strategy: str


def build_long_context(
    sample: LoCoMoSample,
    *,
    max_history_bytes: int,
) -> LongContextPacket:
    """Render chronological history, truncating oldest turns only when necessary."""
    if max_history_bytes <= 0:
        raise ValueError("max_history_bytes must be positive")
    chunks = conversation_chunks(sample)
    rendered = "\n".join(_render(chunk) for chunk in chunks)
    if len(rendered.encode("utf-8")) <= max_history_bytes:
        return LongContextPacket(
            text=rendered,
            included_turn_ids=tuple(chunk.turn_id for chunk in chunks),
            omitted_turn_count=0,
            truncation_strategy="none_full_history",
        )

    selected: list[TextChunk] = []
    used = 0
    for chunk in reversed(chunks):
        encoded_size = len(_render(chunk).encode("utf-8")) + 1
        if selected and used + encoded_size > max_history_bytes:
            break
        if encoded_size > max_history_bytes:
            continue
        selected.append(chunk)
        used += encoded_size
    selected.reverse()
    omitted = len(chunks) - len(selected)
    prefix = f"[Oldest {omitted} turns omitted by recency truncation.]\n"
    return LongContextPacket(
        text=prefix + "\n".join(_render(chunk) for chunk in selected),
        included_turn_ids=tuple(chunk.turn_id for chunk in selected),
        omitted_turn_count=omitted,
        truncation_strategy="drop_oldest_turns_keep_most_recent",
    )


def _render(chunk: TextChunk) -> str:
    return f"<{chunk.turn_id}> {chunk.text}"
