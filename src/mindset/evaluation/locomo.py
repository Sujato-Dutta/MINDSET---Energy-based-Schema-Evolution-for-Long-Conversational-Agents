from __future__ import annotations

import re
import string
from collections import Counter, defaultdict
from typing import Any

from nltk.stem import PorterStemmer


_STEMMER = PorterStemmer()


def locomo_normalize(text: object) -> str:
    value = str(text).replace(",", "").lower()
    value = "".join(character for character in value if character not in set(string.punctuation))
    value = re.sub(r"\b(a|an|the|and)\b", " ", value)
    return " ".join(value.split())


def locomo_exact_match(prediction: object, ground_truth: object) -> float:
    predicted = set(locomo_normalize(prediction).split())
    expected = set(locomo_normalize(ground_truth).split())
    return float(predicted == expected)


def locomo_f1_score(prediction: object, ground_truth: object) -> float:
    predicted = [_STEMMER.stem(word) for word in locomo_normalize(prediction).split()]
    expected = [_STEMMER.stem(word) for word in locomo_normalize(ground_truth).split()]
    if not predicted or not expected:
        return float(predicted == expected)
    count = sum((Counter(predicted) & Counter(expected)).values())
    if count == 0:
        return 0.0
    precision = count / len(predicted)
    recall = count / len(expected)
    return 2 * precision * recall / (precision + recall)


def locomo_multi_answer_f1(prediction: object, ground_truth: object) -> float:
    predictions = [item.strip() for item in str(prediction).split(",")]
    answers = [item.strip() for item in str(ground_truth).split(",")]
    if not answers:
        return 0.0
    return sum(
        max(locomo_f1_score(candidate, answer) for candidate in predictions) for answer in answers
    ) / len(answers)


def locomo_native_score(prediction: object, answer: object, category: int) -> float:
    """Faithful local reproduction of official task_eval/evaluation.py QA F1."""
    expected = str(answer)
    if category == 3:
        expected = expected.split(";")[0].strip()
    if category in {2, 3, 4}:
        return locomo_f1_score(prediction, expected)
    if category == 1:
        return locomo_multi_answer_f1(prediction, expected)
    if category == 5:
        lowered = str(prediction).lower()
        return float("no information available" in lowered or "not mentioned" in lowered)
    raise ValueError(f"Unknown LoCoMo category: {category}")


def evidence_recall(retrieved_ids: list[str], evidence: list[str]) -> float:
    expanded = [part.strip() for group in evidence for part in group.split(";") if part.strip()]
    if not expanded:
        return 1.0
    retrieved = set(retrieved_ids)
    if retrieved and all(item.startswith("S") for item in retrieved):
        sessions = {item[1:] for item in retrieved}
        return sum(item.split(":")[0][1:] in sessions for item in expanded) / len(expanded)
    return sum(item in retrieved for item in expanded) / len(expanded)


def evaluate_locomo(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[int, list[float]] = defaultdict(list)
    scored: list[dict[str, Any]] = []
    for record in records:
        category = int(record["category"])
        score = locomo_native_score(record["prediction"], record["answer"], category)
        by_category[category].append(score)
        scored.append({"id": record.get("id"), "category": category, "score": score})
    all_scores = [item["score"] for item in scored]
    return {
        "native_f1": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "category_f1": {
            str(category): sum(values) / len(values)
            for category, values in sorted(by_category.items())
        },
        "category_counts": {
            str(category): len(values) for category, values in sorted(by_category.items())
        },
        "per_example": scored,
    }
