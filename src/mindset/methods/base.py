from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class ConversationEvent:
    event_id: str
    conversation_id: str
    session_id: str
    timestamp: str
    speaker: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryHit:
    memory_id: str
    text: str
    score: float
    source_event_ids: tuple[str, ...]
    memory_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryMethod(Protocol):
    method_name: str

    def ingest(self, events: Sequence[ConversationEvent]) -> None: ...

    def retrieve(self, query: str, top_k: int = 10) -> list[MemoryHit]: ...

    def snapshot(self, path: str | Path) -> Path: ...

    def telemetry(self) -> dict[str, Any]: ...
