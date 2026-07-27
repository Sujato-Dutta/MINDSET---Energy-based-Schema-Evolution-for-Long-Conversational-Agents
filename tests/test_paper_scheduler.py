from __future__ import annotations

import pytest

from scripts.run_paper_protocol import build_command


def test_build_command_preserves_repeat_identity_and_method_limits() -> None:
    command = build_command(
        {
            "dataset": "locomo",
            "method": "nemori",
            "model": "gpt_oss_120b",
            "manifest": "repeat.json",
            "repetition_id": "repeat-2",
            "ablation": "none",
            "scope": "repeatability",
        }
    )
    assert command[-2:] == ["--batch-size", "4"]
    assert command[command.index("--repetition-id") + 1] == "repeat-2"


def test_longmemeval_primary_uses_external_proxy_judge() -> None:
    command = build_command(
        {
            "dataset": "longmemeval",
            "method": "mindset",
            "model": "gpt_oss_120b",
            "manifest": "primary.json",
            "repetition_id": "primary",
            "ablation": "none",
            "scope": "primary",
        }
    )
    assert "--proxy-judge" in command


def test_judge_model_is_rejected_as_a_task_model() -> None:
    with pytest.raises(ValueError, match="not an answer model"):
        build_command(
            {
                "dataset": "locomo",
                "method": "mindset",
                "model": "gemini_judge",
                "manifest": "primary.json",
                "repetition_id": "primary",
                "ablation": "none",
                "scope": "primary",
            }
        )
