from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from mindset.evaluation.metrics import exact_match


def parse_icl_prediction(text: str) -> str:
    """Match the official evaluator: use the first line, stripping an Answer: prefix."""
    first_line = str(text).splitlines()[0].strip() if str(text).splitlines() else ""
    return re.sub(r"^answer\s*:\s*", "", first_line, flags=re.IGNORECASE).strip()


def icl_exact_match(prediction: str, answers: str | Iterable[str]) -> float:
    return exact_match(parse_icl_prediction(prediction), answers)


def _movie_name(resource: str) -> str:
    filename = resource.split("/")[-1]
    cleaned = filename.replace("_", " ").replace("-", " ").replace(">", " ")
    cleaned = re.sub(r"\([^()]*\)", "", cleaned)
    return " ".join(cleaned.split())


def load_movie_entities(path: str | Path) -> tuple[dict[int, str], list[str]]:
    name_to_id = json.loads(Path(path).read_text(encoding="utf-8"))
    id_to_name = {int(entity_id): _movie_name(name) for name, entity_id in name_to_id.items()}
    # The upstream code converts candidates through a set. Sorting makes edit-
    # distance ties deterministic without changing non-tied matches.
    return id_to_name, sorted(set(id_to_name.values()))


def parse_recommendation_items(prediction: str) -> list[str]:
    try:
        _, recommendation_text = prediction.split("1.", maxsplit=1)
    except ValueError:
        recommendation_text = prediction.replace(",", "\n")
    items = []
    for raw in recommendation_text.splitlines():
        cleaned = re.sub(r"\([^()]*\)", "", raw)
        cleaned = re.sub(r"^(?:\d+[.)]?\s*[-–—]?\s*)?", "", cleaned)
        cleaned = " ".join(cleaned.split())
        if cleaned:
            items.append(cleaned)
    return items


def recommendation_recall(
    prediction: str,
    answer_ids: Iterable[str | int],
    entity_path: str | Path,
    *,
    distance: Callable[[str, str], int] | None = None,
) -> dict[str, Any]:
    if distance is None:
        try:
            import editdistance
        except ImportError as exc:
            raise RuntimeError("Install editdistance for native MemoryAgentBench Recsys scoring") from exc
        distance = editdistance.eval
    id_to_name, candidates = load_movie_entities(entity_path)
    raw_items = parse_recommendation_items(prediction)
    predicted = [
        min(candidates, key=lambda candidate: (distance(item.lower(), candidate.lower()), candidate))
        for item in raw_items
    ]
    ground_truth = [id_to_name[int(identifier)] for identifier in answer_ids]
    denominator = len(ground_truth)
    if denominator == 0:
        raise ValueError("Recommendation ground truth cannot be empty")
    scores = {
        f"recall@{k}": sum(movie in predicted[:k] for movie in ground_truth) / denominator
        for k in (1, 5, 10)
    }
    return {**scores, "parsed_prediction": predicted, "ground_truth_movies": ground_truth}


def evaluate_memoryagentbench_ttl(
    records: list[dict[str, Any]], entity_path: str | Path
) -> dict[str, Any]:
    icl_by_source: dict[str, list[float]] = defaultdict(list)
    recsys: list[float] = []
    per_example: list[dict[str, Any]] = []
    for record in records:
        source = str(record["category"])
        answers = record.get("answers", record["answer"])
        if source.startswith("recsys"):
            score = recommendation_recall(record["prediction"], answers, entity_path)
            recsys.append(float(score["recall@5"]))
            per_example.append({"id": record.get("id"), "source": source, **score})
        else:
            score_value = icl_exact_match(record["prediction"], answers)
            icl_by_source[source].append(score_value)
            per_example.append(
                {"id": record.get("id"), "source": source, "exact_match": score_value}
            )
    all_icl = [score for values in icl_by_source.values() for score in values]
    return {
        "icl_exact_match": sum(all_icl) / len(all_icl) if all_icl else None,
        "icl_exact_match_by_source": {
            source: sum(values) / len(values) for source, values in sorted(icl_by_source.items())
        },
        "recsys_recall@5": sum(recsys) / len(recsys) if recsys else None,
        "icl_count": len(all_icl),
        "recsys_count": len(recsys),
        "per_example": per_example,
    }
