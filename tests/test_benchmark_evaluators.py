from __future__ import annotations

import json

from mindset.evaluation.longmemeval import aggregate_judge_labels, retrieval_scores
from mindset.evaluation.memoryagentbench import (
    icl_exact_match,
    parse_recommendation_items,
    recommendation_recall,
)


def test_memoryagentbench_icl_exact_match_is_strict_after_first_line_parsing() -> None:
    assert icl_exact_match("Answer: 43\nExplanation", ["43"]) == 1.0
    assert icl_exact_match("label: 43", ["43"]) == 0.0


def test_memoryagentbench_recommendation_recall(tmp_path) -> None:
    mapping = {"<http://dbpedia.org/resource/Movie_One_(film)>": 7, "Movie_Two": 8}
    path = tmp_path / "entity2id.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    assert parse_recommendation_items("1. Movie One\n2. Movie Two") == ["Movie One", "Movie Two"]
    result = recommendation_recall(
        "1. Movie One\n2. Movie Two",
        ["8"],
        path,
        distance=lambda left, right: 0 if left == right else 100,
    )
    assert result["recall@1"] == 0.0
    assert result["recall@5"] == 1.0


def test_longmemeval_native_retrieval_and_judge_aggregation() -> None:
    scores = retrieval_scores(["s2", "s1", "x"], ["s1", "s2"], 3)
    assert scores["recall_any@3"] == 1.0
    assert scores["recall_all@3"] == 1.0
    report = aggregate_judge_labels(
        [
            {"question_id": "a", "question_type": "knowledge-update", "judge_label": True},
            {"question_id": "b_abs", "question_type": "multi-session", "judge_label": False},
        ]
    )
    assert report["overall_accuracy"] == 0.5
    assert report["abstention_accuracy"] == 0.0
