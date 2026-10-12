from __future__ import annotations

from typing import Any
from collections import Counter

from mindset.methods.base import ConversationEvent


def longmemeval_events(record: dict[str, Any]) -> list[ConversationEvent]:
    question_id = str(record["question_id"])
    sessions = record.get("haystack_sessions", [])
    session_ids = record.get("haystack_session_ids", [])
    dates = record.get("haystack_dates", [])
    if not (len(sessions) == len(session_ids) == len(dates)):
        raise ValueError(f"LongMemEval session arrays are misaligned for {question_id}")
    events = []
    occurrences=Counter()
    for session_index, (messages, session_id, date) in enumerate(zip(sessions, session_ids, dates, strict=True)):
        occurrence=occurrences[str(session_id)]; occurrences[str(session_id)]+=1
        for message_index, message in enumerate(messages):
            identifier=f"{question_id}:{session_id}:{message_index}"
            # Preserve every raw occurrence, including repeated native sessions.
            # The native retrieval/gold session ID stays unchanged. Only the
            # colliding episode occurrence receives a deterministic suffix.
            if occurrence: identifier+=f":occurrence={occurrence}"
            metadata={"benchmark_session_id":str(session_id),"session_index":session_index}
            if occurrence: metadata["session_occurrence"]=occurrence
            events.append(
                ConversationEvent(
                    event_id=identifier,
                    conversation_id=question_id,
                    session_id=str(session_id),
                    timestamp=str(date),
                    speaker=str(message.get("role", "unknown")),
                    text=str(message.get("content", "")),
                    metadata=metadata,
                )
            )
    return events
