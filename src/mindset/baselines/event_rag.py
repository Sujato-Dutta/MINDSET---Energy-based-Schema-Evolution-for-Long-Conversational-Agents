from __future__ import annotations

from typing import Sequence

import numpy as np

from mindset.methods.base import ConversationEvent, MemoryHit


class EventRAGIndex:
    """Dataset-neutral flat MiniLM index over immutable conversation events."""

    method_name = "vector_rag"

    def __init__(self, events: Sequence[ConversationEvent], encoder: object) -> None:
        self.events = list(events)
        self.encoder = encoder
        texts = [render_event(event) for event in self.events]
        self.vectors = np.asarray(
            encoder.encode(
                texts, batch_size=64, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            ),
            dtype=np.float32,
        ) if texts else np.empty((0, 0), dtype=np.float32)

    def retrieve(self, query: str, top_k: int = 10) -> list[MemoryHit]:
        if top_k <= 0 or not self.events:
            return []
        vector = np.asarray(
            self.encoder.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False,
            )[0], dtype=np.float32,
        )
        scores = self.vectors @ vector
        indices = np.argsort(-scores, kind="stable")[:top_k]
        return [
            MemoryHit(
                self.events[int(index)].event_id,
                render_event(self.events[int(index)]),
                float(scores[int(index)]),
                (self.events[int(index)].event_id,),
                "episodic",
                dict(self.events[int(index)].metadata),
            )
            for index in indices
        ]


def render_event(event: ConversationEvent) -> str:
    return f"[{event.timestamp}] | Speaker: {event.speaker}\n{event.text}"
