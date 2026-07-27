from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Iterable


def normalize_answer(text: object) -> str:
    value = str(text).lower()
    value = "".join(character for character in value if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def exact_match(prediction: object, references: object | Iterable[object]) -> float:
    refs = _references(references)
    normalized = normalize_answer(prediction)
    return float(any(normalized == normalize_answer(reference) for reference in refs))


def token_f1(prediction: object, references: object | Iterable[object]) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    scores = [
        _f1_tokens(prediction_tokens, normalize_answer(reference).split())
        for reference in _references(references)
    ]
    return max(scores, default=0.0)


def _f1_tokens(prediction: list[str], reference: list[str]) -> float:
    if not prediction or not reference:
        return float(prediction == reference)
    common = sum((Counter(prediction) & Counter(reference)).values())
    if common == 0:
        return 0.0
    precision = common / len(prediction)
    recall = common / len(reference)
    return 2 * precision * recall / (precision + recall)


def _references(references: object | Iterable[object]) -> list[object]:
    if isinstance(references, (str, bytes, int, float)) or references is None:
        return [references]
    return list(references)


def retrieval_metrics(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> dict[str, float]:
    top = retrieved_ids[:k]
    hits = [int(item in relevant_ids) for item in top]
    recall = sum(hits) / len(relevant_ids) if relevant_ids else 1.0
    precision = sum(hits) / k if k > 0 else 0.0
    reciprocal_rank = next((1.0 / (index + 1) for index, hit in enumerate(hits) if hit), 0.0)
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return {
        f"recall@{k}": recall,
        f"precision@{k}": precision,
        "mrr": reciprocal_rank,
        f"ndcg@{k}": dcg / idcg if idcg else 1.0,
    }
