from __future__ import annotations

import math

from mindset.evaluation.locomo import (
    evidence_recall,
    locomo_exact_match,
    locomo_native_score,
)
from mindset.evaluation.metrics import exact_match, retrieval_metrics, token_f1


def test_generic_exact_and_f1_metrics() -> None:
    assert exact_match("The blue house", "blue house") == 1.0
    assert token_f1("blue blue house", "blue house") == 0.8


def test_official_locomo_set_exact_match_and_categories() -> None:
    assert locomo_exact_match("tea and hiking", "hiking, tea") == 1.0
    assert locomo_native_score("painting, hiking", "hiking, painting", 1) == 1.0
    assert locomo_native_score("No information available.", "", 5) == 1.0


def test_retrieval_metrics_and_semicolon_evidence() -> None:
    metrics = retrieval_metrics(["e2", "e1", "x"], {"e1", "e2"}, 3)
    assert metrics["recall@3"] == 1.0
    assert math.isclose(metrics["precision@3"], 2 / 3)
    assert evidence_recall(["D8:6", "D9:17"], ["D8:6; D9:17"]) == 1.0
