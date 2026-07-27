from __future__ import annotations

import re
from typing import Any

from mindset.methods.base import ConversationEvent


def longmemeval_events(record: dict[str, Any]) -> list[ConversationEvent]:
    question_id = str(record["question_id"])
    sessions = record.get("haystack_sessions", [])
    session_ids = record.get("haystack_session_ids", [])
    dates = record.get("haystack_dates", [])
    if not (len(sessions) == len(session_ids) == len(dates)):
        raise ValueError(f"LongMemEval session arrays are misaligned for {question_id}")
    events = []
    for session_index, (messages, session_id, date) in enumerate(zip(sessions, session_ids, dates, strict=True)):
        for message_index, message in enumerate(messages):
            events.append(
                ConversationEvent(
                    event_id=f"{question_id}:{session_id}:{message_index}",
                    conversation_id=question_id,
                    session_id=str(session_id),
                    timestamp=str(date),
                    speaker=str(message.get("role", "unknown")),
                    text=str(message.get("content", "")),
                    metadata={"benchmark_session_id": str(session_id), "session_index": session_index},
                )
            )
    return events


def subtlememory_events(record: dict[str, Any]) -> list[ConversationEvent]:
    instance_id = str(record['instance_id'])
    facts = [str(value) for value in record.get('facts', [])]
    return [
        ConversationEvent(
            event_id=f'{instance_id}:fact-{index:04d}',
            conversation_id=instance_id,
            session_id=str(record.get('source_split', 'subtlememory')),
            timestamp=str(index),
            speaker='memory',
            text=text,
            metadata={
                'relation_type': str(record.get('relation_type', 'unknown')),
                'relation_subtype': str(record.get('relation_subtype', 'unknown')),
                'source_split': str(record.get('source_split', 'unknown')),
                'memory_bank_date': instance_id,
            },
        )
        for index, text in enumerate(facts)
    ]


def memoryagentbench_context_events(
    context: str,
    *,
    source_row: int,
    max_chunk_bytes: int = 2400,
) -> list[ConversationEvent]:
    if max_chunk_bytes <= 0:
        raise ValueError("max_chunk_bytes must be positive")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", context) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else current + "\n\n" + paragraph
        if current and len(candidate.encode("utf-8")) > max_chunk_bytes:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return [
        ConversationEvent(
            event_id=f"mab-row-{source_row}:chunk-{index:06d}",
            conversation_id=f"mab-row-{source_row}",
            session_id=f"row-{source_row}",
            timestamp=str(index),
            speaker="context",
            text=text,
            metadata={"source_row": source_row, "memory_bank_date": f"row-{source_row}"},
        )
        for index, text in enumerate(chunks)
    ]
