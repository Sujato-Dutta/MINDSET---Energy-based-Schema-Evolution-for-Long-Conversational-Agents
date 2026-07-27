from __future__ import annotations

import pytest

from mindset.api.ledger import TokenBudgetExceeded
from mindset.api.models import BackendConfig, ModelConfig, ProviderConfig
from mindset.experiments.batching import (
    AnswerQuestion,
    answer_batches,
    assert_budget_preflight,
    build_answer_batches,
    estimate_batch_reservation,
)
from mindset.methods.base import MemoryHit


def hit(identifier: str, text: str) -> MemoryHit:
    return MemoryHit(identifier, text, 1.0, (identifier,), "test")


def test_batch_prompt_isolates_evidence_per_question() -> None:
    shared = hit("m1", "one shared memory")
    questions = [
        AnswerQuestion("q1", "first?", (shared,)),
        AnswerQuestion("q2", "second?", (shared, hit("m2", "other"))),
    ]
    batches = build_answer_batches(questions, max_questions=8)
    assert len(batches) == 1
    assert batches[0].user_prompt.count("one shared memory") == 2
    assert '"memories"' not in batches[0].user_prompt
    assert estimate_batch_reservation(batches, max_output_tokens=100, max_retries=2) > 0


def test_batch_prompt_strips_derived_schema_summary() -> None:
    schema = MemoryHit(
        "s1",
        "Structured memory: unsupported summary\nGrounding:\n[date] Sam: raw fact",
        1.0,
        ("e1",),
        "schema_active",
    )
    prompt = build_answer_batches([AnswerQuestion("q1", "question?", (schema,))])[0].user_prompt
    assert "unsupported summary" not in prompt
    assert "raw fact" in prompt


def test_batch_limits_are_enforced() -> None:
    questions = [AnswerQuestion(f"q{i}", "question", (hit(f"m{i}", "memory"),)) for i in range(3)]
    assert len(build_answer_batches(questions, max_questions=2)) == 2


class FakeLedger:
    def report(self):
        return {"models": [{"model_name": "m", "input_tokens": 700, "output_tokens": 0, "reserved_tokens": 0}]}


class FakeBackend:
    config = BackendConfig(
        providers={"p": ProviderConfig(endpoint="http://unused", key_env="X")},
        models={"m": ModelConfig(provider="p", model_id="id", hard_token_budget=1000)},
        cache_path="unused",
        ledger_path="unused",
    )
    ledger = FakeLedger()


def test_preflight_preserves_safety_reserve() -> None:
    with pytest.raises(TokenBudgetExceeded, match="Preflight blocked"):
        assert_budget_preflight(FakeBackend(), model_name="m", required_tokens=250, reserve_fraction=0.1)
    report = assert_budget_preflight(FakeBackend(), model_name="m", required_tokens=200, reserve_fraction=0.1)
    assert report["usable_tokens"] == 200


class CompletenessRetryBackend:
    config = BackendConfig(
        providers={"p": ProviderConfig(endpoint="http://unused", key_env="X")},
        models={"m": ModelConfig(provider="p", model_id="id", hard_token_budget=100_000)},
        cache_path="unused",
        ledger_path="unused",
    )

    class Ledger:
        def report(self):
            return {"models": []}

    ledger = Ledger()

    def __init__(self) -> None:
        self.requests = []

    def complete(self, request, **_kwargs):
        self.requests.append(request)
        if len(self.requests) == 1:
            payload = {"answers": [{"question_id": "q1", "answer": "one"}]}
        else:
            payload = {
                "answers": [
                    {"question_id": "q1", "answer": "one"},
                    {"question_id": "q2", "answer": "two"},
                ]
            }
        return type("Result", (), {"parsed_json": payload})()


def test_answer_batches_retries_incomplete_id_set_with_fresh_identity() -> None:
    backend = CompletenessRetryBackend()
    batch = build_answer_batches(
        [
            AnswerQuestion("q1", "first?", (hit("m1", "one"),)),
            AnswerQuestion("q2", "second?", (hit("m2", "two"),)),
        ]
    )
    answers = answer_batches(
        backend,
        batch,
        model_name="m",
        method="test",
        code_version="v1",
        max_output_tokens=100,
        max_retries=1,
        reserve_fraction=0,
    )
    assert answers == {"q1": "one", "q2": "two"}
    assert [request.code_version for request in backend.requests] == [
        "v1",
        "v1-completeness-retry-1",
    ]
    assert "Return exactly one answer" in backend.requests[1].user_prompt
