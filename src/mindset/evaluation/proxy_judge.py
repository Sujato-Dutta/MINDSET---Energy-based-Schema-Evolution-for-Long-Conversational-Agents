from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from mindset.api.backend import APIBackend
from mindset.api.models import CompletionRequest
from mindset.evaluation.judge import JudgeVerdict, build_blind_judge_prompt
from mindset.utils.hashing import stable_hash


@dataclass(frozen=True)
class ProxyJudgeDecision:
    verdict: str
    agreement: bool
    judge_verdicts: tuple[str, ...]
    judge_models: tuple[str, ...]
    protocol: str = "external_gemini_proxy_v2_not_official_human_gold"


def judge_with_proxy_consensus(
    backend: APIBackend,
    *,
    question_id: str,
    question: str,
    reference: str,
    prediction: str,
    judge_models: Sequence[str] = ("gemini_judge",),
    evidence: str | None = None,
    code_version: str = "proxy-judge-v1",
    telemetry: list[dict[str, object]] | None = None,
    request_metadata: Mapping[str, object] | None = None,
) -> ProxyJudgeDecision:
    if not judge_models or len(set(judge_models)) != len(judge_models):
        raise ValueError("At least one distinct external judge model is required")
    prompt = build_blind_judge_prompt(question, reference, prediction, evidence)
    verdicts: list[str] = []
    for model in judge_models:
        result = backend.complete(
            CompletionRequest(
                model_name=model,
                system_prompt="You are a blinded benchmark answer judge. Follow the requested JSON schema.",
                user_prompt=prompt,
                dataset_example_id=f"judge-{question_id}-{stable_hash(model)[:8]}",
                method="proxy_judge",
                schema_version="judge-verdict-v1",
                code_version=code_version,
                max_output_tokens=256,
                response_format="json_object",
                metadata={
                    **dict(request_metadata or {}),
                    "question_id": question_id,
                    "judge_model": model,
                },
            ),
            structured_model=JudgeVerdict,
            max_retries=2,
        )
        if telemetry is not None:
            telemetry.append(
                {
                    "question_id": question_id,
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
        verdicts.append(JudgeVerdict.model_validate(result.parsed_json).verdict)
    agreement = len(set(verdicts)) == 1
    verdict = verdicts[0] if agreement else _conservative_resolution(verdicts)
    return ProxyJudgeDecision(verdict, agreement, tuple(verdicts), tuple(judge_models))


def _conservative_resolution(verdicts: Sequence[str]) -> str:
    # Disagreement is never upgraded. Unsupported is treated as incorrect for
    # correctness aggregation but remains visible in the per-judge record.
    rank = {"unsupported": 0, "incorrect": 0, "partially_correct": 1, "correct": 2}
    return min(verdicts, key=lambda value: rank[value])
