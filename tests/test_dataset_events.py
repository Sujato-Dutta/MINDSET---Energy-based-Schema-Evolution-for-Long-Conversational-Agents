from __future__ import annotations

from mindset.datasets.events import longmemeval_events, memoryagentbench_context_events


def test_longmemeval_events_preserve_session_provenance() -> None:
    events = longmemeval_events({
        "question_id": "q", "haystack_session_ids": ["s"], "haystack_dates": ["date"],
        "haystack_sessions": [[{"role": "user", "content": "hello"}]],
    })
    assert events[0].metadata["benchmark_session_id"] == "s"
    assert events[0].event_id == "q:s:0"


def test_mab_context_chunking_is_bounded_and_stable() -> None:
    events = memoryagentbench_context_events("first paragraph\n\nsecond paragraph", source_row=2, max_chunk_bytes=20)
    assert [event.event_id for event in events] == ["mab-row-2:chunk-000000", "mab-row-2:chunk-000001"]
