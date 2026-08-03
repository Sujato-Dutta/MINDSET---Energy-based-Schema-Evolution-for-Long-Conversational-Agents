from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import binomtest, wilcoxon


def paired_bootstrap(
    system_a: Sequence[float],
    system_b: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    a, b = _paired_arrays(system_a, system_b)
    difference = a - b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(resamples, len(difference)))
    sampled = difference[indices].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {"delta": float(difference.mean()), "ci_low": float(low), "ci_high": float(high)}


def exact_mcnemar(
    system_a: Sequence[bool], system_b: Sequence[bool]
) -> dict[str, float | int]:
    a, b = _paired_arrays(system_a, system_b, dtype=bool)
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    discordant = a_only + b_only
    p_value = (
        1.0
        if discordant == 0
        else float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue)
    )
    return {
        "a_only": a_only,
        "b_only": b_only,
        "discordant": discordant,
        "p_value": p_value,
    }


def paired_wilcoxon(
    system_a: Sequence[float], system_b: Sequence[float]
) -> dict[str, float]:
    a, b = _paired_arrays(system_a, system_b)
    if np.allclose(a, b):
        return {"statistic": 0.0, "p_value": 1.0, "median_delta": 0.0}
    result = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "median_delta": float(np.median(a - b)),
    }


def holm_correction(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        [(len(values) - rank) * values[index] for rank, index in enumerate(order)]
    )
    adjusted = np.empty_like(values)
    for rank, index in enumerate(order):
        adjusted[index] = min(1.0, adjusted_sorted[rank])
    return adjusted.tolist()


def exact_sign_flip_pvalue(contributions: Sequence[float]) -> float:
    """Return a two-sided exact sign-flip p-value over paired contributions."""
    values = np.asarray(contributions, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Contributions must be a non-empty one-dimensional sequence")
    if len(values) > 20:
        raise ValueError("Exact sign-flip enumeration is limited to 20 contributions")

    masks = np.arange(2 ** len(values), dtype=np.uint64)[:, None]
    bit_positions = np.arange(len(values), dtype=np.uint64)[None, :]
    signs = 1.0 - 2.0 * ((masks >> bit_positions) & 1)
    permuted = signs @ values
    observed = abs(float(values.sum()))
    tolerance = np.finfo(float).eps * max(1.0, observed) * 16
    extreme = np.count_nonzero(np.abs(permuted) >= observed - tolerance)
    return float(extreme / len(permuted))


def _paired_arrays(a: Sequence, b: Sequence, dtype=float) -> tuple[np.ndarray, np.ndarray]:
    first, second = np.asarray(a, dtype=dtype), np.asarray(b, dtype=dtype)
    if (
        first.ndim != 1
        or second.ndim != 1
        or len(first) != len(second)
        or len(first) == 0
    ):
        raise ValueError("Inputs must be non-empty paired one-dimensional sequences")
    return first, second
