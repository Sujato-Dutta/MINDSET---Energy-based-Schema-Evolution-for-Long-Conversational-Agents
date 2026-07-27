from __future__ import annotations

from mindset.statistics.tests import (
    exact_mcnemar,
    holm_correction,
    paired_bootstrap,
    paired_wilcoxon,
)


def test_paired_bootstrap_known_delta() -> None:
    result = paired_bootstrap([1, 1, 1], [0, 0, 0], resamples=1000)
    assert result == {"delta": 1.0, "ci_low": 1.0, "ci_high": 1.0}


def test_exact_mcnemar_counts_disagreements() -> None:
    result = exact_mcnemar([True, True, False], [False, True, False])
    assert result["a_only"] == 1
    assert result["b_only"] == 0
    assert result["p_value"] == 1.0


def test_holm_correction_preserves_original_order() -> None:
    adjusted = holm_correction([0.01, 0.04, 0.03])
    assert adjusted == [0.03, 0.06, 0.06]


def test_wilcoxon_identical_vectors() -> None:
    assert paired_wilcoxon([1, 2], [1, 2])["p_value"] == 1.0
