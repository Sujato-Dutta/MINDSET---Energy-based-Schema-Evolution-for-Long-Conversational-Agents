from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


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
        f"ndcg_any@{k}": actual_dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def aggregate_judge_labels(records: list[dict[str, Any]]) -> dict[str, Any]:
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
        "official_judge_model": "gpt-4o-2024-08-06",
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
        instruction = (
            "Decide whether the response contains the required updated answer. Older information "
            "may also appear, provided the updated answer is present."
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
