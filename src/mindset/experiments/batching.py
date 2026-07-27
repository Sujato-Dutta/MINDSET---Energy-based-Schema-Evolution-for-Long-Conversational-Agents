from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from mindset.api.backend import APIBackend
from mindset.api.ledger import TokenBudgetExceeded
from mindset.api.models import CompletionRequest
from mindset.methods.base import MemoryHit


@dataclass(frozen=True)
class AnswerQuestion:
    question_id: str
    question: str
    evidence: tuple[MemoryHit, ...]


@dataclass(frozen=True)
class AnswerBatch:
    batch_id: str
    questions: tuple[AnswerQuestion, ...]
    user_prompt: str


class BatchAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    answer: str = Field(min_length=1, max_length=2000)


class BatchAnswerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answers: list[BatchAnswer]


SYSTEM_PROMPT = """Answer every question using only the supplied memory evidence.
Return one strict JSON object with an `answers` array. Each array item must contain
exactly `question_id` and a concise `answer`. Preserve every question ID exactly.
Each question contains its own isolated `evidence` array; evidence from another
question is out of scope. Exact names, numbers, dates, colors, attributes, and list members must
be explicitly supported by that question's grounding evidence.
If the evidence does not support an answer, answer exactly `No information available.`;
do not guess."""


def build_answer_batches(
    questions: Sequence[AnswerQuestion],
    *,
    max_questions: int = 8,
    max_prompt_bytes: int = 90_000,
) -> list[AnswerBatch]:
    if max_questions <= 0 or max_prompt_bytes <= 0:
        raise ValueError("batch limits must be positive")
    batches: list[AnswerBatch] = []
    pending: list[AnswerQuestion] = []
    for question in questions:
        candidate = [*pending, question]
        prompt = render_batch_prompt(candidate)
        if pending and (len(candidate) > max_questions or len(prompt.encode("utf-8")) > max_prompt_bytes):
            batches.append(_make_batch(len(batches), pending))
            pending = [question]
            prompt = render_batch_prompt(pending)
        else:
            pending = candidate
        if len(prompt.encode("utf-8")) > max_prompt_bytes:
            raise ValueError(f"Single question {question.question_id} exceeds max_prompt_bytes")
    if pending:
        batches.append(_make_batch(len(batches), pending))
    return batches


def render_batch_prompt(questions: Sequence[AnswerQuestion]) -> str:
    rendered_questions = []
    for question in questions:
        evidence = [
            f"[{rank}] {_answer_evidence_text(hit)}"
            for rank, hit in enumerate(question.evidence, start=1)
        ]
        rendered_questions.append(
            {
                "question_id": question.question_id,
                "question": question.question,
                "evidence": evidence,
            }
        )
    payload = {"questions": rendered_questions}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _answer_evidence_text(hit: MemoryHit) -> str:
    """Expose raw grounding, never a schema's derived summary, to the answer model."""
    if hit.memory_type.startswith("schema_") and "\nGrounding:\n" in hit.text:
        return hit.text.split("\nGrounding:\n", 1)[1]
    return hit.text


def estimate_batch_reservation(
    batches: Iterable[AnswerBatch],
    *,
    max_output_tokens: int,
    max_retries: int,
) -> int:
    if max_output_tokens <= 0 or max_retries < 0:
        raise ValueError("invalid output/retry limits")
    # Whole-run preflight uses a conservative four UTF-8 bytes per token and
    # one successful attempt. Each actual request separately reserves the byte
    # upper bound for every retry in APIBackend, so retries still fail closed
    # without making a multi-cell paper plan appear several times larger.
    return sum(
        math.ceil(len((SYSTEM_PROMPT + batch.user_prompt).encode("utf-8")) / 4)
        + max_output_tokens
        for batch in batches
    )


def assert_budget_preflight(
    backend: APIBackend,
    *,
    model_name: str,
    required_tokens: int,
    reserve_fraction: float = 0.10,
) -> dict[str, int | float]:
    if not 0 <= reserve_fraction < 1:
        raise ValueError("reserve_fraction must lie in [0, 1)")
    model = backend.config.models[model_name]
    rows = {row["model_name"]: row for row in backend.ledger.report().get("models", [])}
    row = rows.get(model_name, {})
    used = int(row.get("input_tokens", 0)) + int(row.get("output_tokens", 0))
    reserved = int(row.get("reserved_tokens", 0))
    safety_reserve = int(model.hard_token_budget * reserve_fraction)
    usable = model.hard_token_budget - used - reserved - safety_reserve
    report = {
        "hard_budget": model.hard_token_budget,
        "used_tokens": used,
        "outstanding_reservations": reserved,
        "safety_reserve": safety_reserve,
        "usable_tokens": usable,
        "required_tokens": required_tokens,
    }
    if required_tokens > usable:
        raise TokenBudgetExceeded(
            f"Preflight blocked {model_name}: requires {required_tokens:,}, only {usable:,} "
            f"available after {reserve_fraction:.0%} reserve"
        )
    return report


def answer_batches(
    backend: APIBackend,
    batches: Sequence[AnswerBatch],
    *,
    model_name: str,
    method: str,
    code_version: str,
    max_output_tokens: int = 768,
    max_retries: int = 2,
    reserve_fraction: float = 0.10,
    telemetry: list[dict[str, object]] | None = None,
    request_metadata: Mapping[str, object] | None = None,
) -> dict[str, str]:
    required = estimate_batch_reservation(
        batches, max_output_tokens=max_output_tokens, max_retries=max_retries
    )
    assert_budget_preflight(
        backend,
        model_name=model_name,
        required_tokens=required,
        reserve_fraction=reserve_fraction,
    )
    answers: dict[str, str] = {}
    for batch in batches:
        expected = {question.question_id for question in batch.questions}
        expected_list = sorted(expected)
        for completeness_attempt in range(max_retries + 1):
            retry_instruction = ""
            if completeness_attempt:
                retry_instruction = (
                    "\n\nYour previous response failed completeness validation. Return exactly one "
                    f"answer for each of these {len(expected_list)} IDs, with no other IDs: "
                    + json.dumps(expected_list, ensure_ascii=False)
                )
            result = backend.complete(
                CompletionRequest(
                    model_name=model_name,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=batch.user_prompt + retry_instruction,
                    dataset_example_id=batch.batch_id,
                    method=method,
                    schema_version="batch-answer-v4-compact-evidence",
                    code_version=(
                        code_version
                        if completeness_attempt == 0
                        else f"{code_version}-completeness-retry-{completeness_attempt}"
                    ),
                    max_output_tokens=max_output_tokens,
                    response_format="json_object",
                    metadata={
                        **dict(request_metadata or {}),
                        "batch_id": batch.batch_id,
                        "batch_question_count": len(batch.questions),
                        "completeness_attempt": completeness_attempt,
                    },
                ),
                structured_model=BatchAnswerOutput,
                max_retries=max_retries,
            )
            if telemetry is not None:
                telemetry.append(
                    {
                        "batch_id": batch.batch_id,
                        "question_count": len(batch.questions),
                        "completeness_attempt": completeness_attempt,
                        "provider": result.provider,
                        "model_id": result.model_id,
                        "key_alias": result.key_alias,
                        "request_id": result.request_id,
                        "cache_hit": result.cache_hit,
                        "billed_input_tokens": (
                            0 if result.cache_hit else result.usage.input_tokens
                        ),
                        "billed_output_tokens": (
                            0 if result.cache_hit else result.usage.output_tokens
                        ),
                        "logical_input_tokens": result.usage.input_tokens,
                        "logical_output_tokens": result.usage.output_tokens,
                        "latency_seconds": result.latency_seconds,
                        "retry_count": result.retry_count,
                    }
                )
            parsed = BatchAnswerOutput.model_validate(result.parsed_json)
            received = [item.question_id for item in parsed.answers]
            if set(received) == expected and len(received) == len(expected):
                answers.update({item.question_id: item.answer for item in parsed.answers})
                break
        else:
            raise ValueError(
                f"Batch {batch.batch_id} returned missing, duplicate, or unexpected IDs "
                f"after {max_retries + 1} completeness attempts"
            )
    return answers


def _make_batch(index: int, questions: Sequence[AnswerQuestion]) -> AnswerBatch:
    return AnswerBatch(
        batch_id=f"answer-batch-{index:05d}",
        questions=tuple(questions),
        user_prompt=render_batch_prompt(questions),
    )
