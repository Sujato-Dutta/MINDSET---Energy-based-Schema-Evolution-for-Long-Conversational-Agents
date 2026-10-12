from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

from mindset.api.backend import APIBackend
from mindset.api.models import CompletionRequest
from mindset.evaluation.metrics import exact_match, normalize_answer, token_f1


def retrieval_scores(
    ranked_ids: list[str], relevant_ids: Iterable[str], k: int
) -> dict[str, float]:
    relevant = set(relevant_ids)
    retrieved = ranked_ids[:k]
    hits = [int(identifier in relevant) for identifier in retrieved]
    ideal_hits = min(len(relevant), k)
    actual_dcg = sum(hit if index == 0 else hit / math.log2(index + 1) for index, hit in enumerate(hits))
    ideal_dcg = sum(
        1.0 if index == 0 else 1.0 / math.log2(index + 1)
        for index in range(ideal_hits)
    )
    return {
        f"recall_any@{k}": float(any(hits)),
        f"recall_all@{k}": float(relevant.issubset(set(retrieved))),
        f"recall@{k}": sum(hits) / len(relevant) if relevant else 1.0,
        f"precision@{k}": sum(hits) / k if k else 0.0,
        f"mrr@{k}": next(
            (1.0 / (index + 1) for index, hit in enumerate(hits) if hit), 0.0
        ),
        f"ndcg@{k}": actual_dcg / ideal_dcg if ideal_dcg else 1.0,
        f"ndcg_any@{k}": actual_dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def aggregate_judge_labels(
    records: list[dict[str, Any]], *, judge_model: str = "gpt-4o-2024-08-06"
) -> dict[str, Any]:
    by_type: dict[str, list[int]] = defaultdict(list)
    abstention: list[int] = []
    for record in records:
        value = int(bool(record["judge_label"]))
        question_type = str(record["question_type"])
        by_type[question_type].append(value)
        question_id = record.get("question_id", record.get("id", ""))
        if str(question_id).endswith("_abs"):
            abstention.append(value)
    all_values = [value for values in by_type.values() for value in values]
    type_accuracy = {
        label: sum(values) / len(values) for label, values in sorted(by_type.items())
    }
    return {
        "judge_model": judge_model,
        "reference_judge_model": "gpt-4o-2024-08-06",
        "overall_accuracy": sum(all_values) / len(all_values) if all_values else None,
        "task_averaged_accuracy": (
            sum(type_accuracy.values()) / len(type_accuracy) if type_accuracy else None
        ),
        "abstention_accuracy": (
            sum(abstention) / len(abstention) if abstention else None
        ),
        "accuracy_by_type": type_accuracy,
        "count_by_type": {label: len(values) for label, values in sorted(by_type.items())},
    }


def deterministic_answer_scores(prediction: object, answer: object) -> dict[str, float]:
    predicted = normalize_answer(prediction)
    expected = normalize_answer(answer)
    return {
        "normalized_exact_match": exact_match(prediction, answer),
        "token_f1": token_f1(prediction, answer),
        "answer_containment": float(bool(expected) and expected in predicted),
    }


def aggregate_deterministic_answers(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_example = []
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        scores = deterministic_answer_scores(record["prediction"], record["answer"])
        per_example.append({"id": record["id"], **scores})
        for name, value in scores.items():
            values[name].append(value)
    return {
        **{
            name: sum(metric_values) / len(metric_values) if metric_values else None
            for name, metric_values in sorted(values.items())
        },
        "count": len(per_example),
        "per_example": per_example,
    }


def judge_native_answer(
    backend: APIBackend,
    *,
    question_id: str,
    question_type: str,
    question: str,
    answer: str,
    prediction: str,
    telemetry: list[dict[str, object]] | None = None,
    request_metadata: dict[str, object] | None = None,
) -> tuple[bool, str]:
    prompt = judge_prompt(
        question_type,
        question,
        answer,
        prediction,
        abstention=question_id.endswith("_abs"),
    )
    result = backend.complete(
        CompletionRequest(
            model_name="gemini_judge",
            system_prompt="Follow the evaluation instruction exactly.",
            user_prompt=prompt,
            dataset_example_id=f"longmemeval-native-judge:{question_id}",
            method="longmemeval_native_judge",
            schema_version="official-longmemeval-judge-prompt-v1",
            code_version="longmemeval-native-gemini-substitution-v1",
            temperature=0.0,
            max_output_tokens=10,
            response_format="text",
            metadata={**dict(request_metadata or {}), "question_id": question_id},
        ),
        max_retries=2,
    )
    raw = result.text.strip()
    label = "yes" in raw.lower()
    if telemetry is not None:
        telemetry.append(
            {
                "question_id": question_id,
                "provider": result.provider,
                "model_id": result.model_id,
                "key_alias": result.key_alias,
                "request_id": result.request_id,
                "cache_hit": result.cache_hit,
                "billed_input_tokens": 0 if result.cache_hit else result.usage.input_tokens,
                "billed_output_tokens": 0 if result.cache_hit else result.usage.output_tokens,
                "logical_input_tokens": result.usage.input_tokens,
                "logical_output_tokens": result.usage.output_tokens,
                "latency_seconds": result.latency_seconds,
                "retry_count": result.retry_count,
                "raw_response": raw,
                "label": label,
            }
        )
    return label, raw


def judge_prompt(
    question_type: str,
    question: str,
    answer: str,
    prediction: str,
    *,
    abstention: bool = False,
) -> str:
    if abstention:
        instruction = (
            "Decide whether the model response correctly identifies the question as unanswerable. "
            "It may say the information is incomplete or that the requested fact was not given."
        )
        answer_label = "Explanation"
    elif question_type == "single-session-preference":
        instruction = (
            "Decide whether the response satisfies the personalized-answer rubric. It need not "
            "mention every rubric point, but must correctly use the user's personal information."
        )
        answer_label = "Rubric"
    elif question_type == "temporal-reasoning":
        instruction = (
            "Decide whether the response contains an equivalent correct answer. Accept an "
            "off-by-one error when the requested answer is a number of days, weeks, or months."
        )
        answer_label = "Correct answer"
    elif question_type == "knowledge-update":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response contains some previous information along with an updated answer, the "
            "response should be considered as correct as long as the updated answer is the required "
            f"answer.\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel "
            f"Response: {prediction}\n\nIs the model response correct? Answer yes or no only."
        )
        answer_label = "Correct answer"
    else:
        instruction = (
            "Decide whether the response contains an equivalent complete correct answer. A response "
            "with only a subset of required information is incorrect."
        )
        answer_label = "Correct answer"
    return (
        f"{instruction}\n\nQuestion: {question}\n\n{answer_label}: {answer}\n\n"
        f"Model response: {prediction}\n\nAnswer yes or no only."
    )
