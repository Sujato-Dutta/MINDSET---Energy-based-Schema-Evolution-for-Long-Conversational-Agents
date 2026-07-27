from __future__ import annotations

from mindset.api.models import CompletionResult, TokenUsage
from mindset.evaluation.proxy_judge import judge_with_proxy_consensus


class FakeBackend:
    def __init__(self):
        self.calls = 0

    def complete(self, request, **kwargs):
        self.calls += 1
        verdict = "correct" if self.calls == 1 else "partially_correct"
        return CompletionResult(
            text="",
            provider="fake",
            model_id=request.model_name,
            key_alias="fake",
            usage=TokenUsage(),
            latency_seconds=0,
            cache_key=str(self.calls),
            parsed_json={"verdict": verdict, "rationale": "test"},
        )


def test_proxy_judge_marks_disagreement_and_resolves_conservatively() -> None:
    decision = judge_with_proxy_consensus(
        FakeBackend(),
        question_id="q1",
        question="q",
        reference="a",
        prediction="b",
        judge_models=("m1", "m2"),
    )
    assert not decision.agreement
    assert decision.verdict == "partially_correct"
    assert "not_official_human_gold" in decision.protocol
