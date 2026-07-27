from __future__ import annotations

from pathlib import Path

from mindset.baselines.long_context import build_long_context
from mindset.datasets.locomo import load_locomo


def test_long_context_keeps_full_history_when_it_fits() -> None:
    root = Path(__file__).resolve().parents[1]
    sample = load_locomo(root / "data" / "locomo" / "locomo10.json")[0]
    packet = build_long_context(sample, max_history_bytes=1_000_000)
    assert packet.omitted_turn_count == 0
    assert packet.truncation_strategy == "none_full_history"
    assert packet.included_turn_ids[0] == sample.sessions[0].turns[0].dia_id


def test_long_context_truncation_is_recency_preserving_and_bounded() -> None:
    root = Path(__file__).resolve().parents[1]
    sample = load_locomo(root / "data" / "locomo" / "locomo10.json")[0]
    packet = build_long_context(sample, max_history_bytes=2_000)
    assert packet.omitted_turn_count > 0
    assert packet.included_turn_ids[-1] == sample.sessions[-1].turns[-1].dia_id
    assert packet.truncation_strategy == "drop_oldest_turns_keep_most_recent"
