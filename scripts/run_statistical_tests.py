"""Paired statistical tests for the frozen MINDSET experiments.

Primary inference uses conversation-clustered paired bootstrap/permutation tests
for LoCoMo and paired question-level tests for MemoryAgentBench. All tests are
deterministic and operate only on saved prediction/metric artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindset.statistics.tests import exact_sign_flip_pvalue

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover - scipy is present in the project environment
    wilcoxon = None

OUT = ROOT / "outputs" / "statistics"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260720
BOOTSTRAPS = 20_000
PERMUTATIONS = 50_000
LOCOMO_RANKING_METRICS = frozenset(
    {"recall@8", "precision@8", "mrr", "ndcg@8"}
)

LOCOMO_MAIN = {
    "MINDSET": "locomo_mindset_full_gpt_oss_120b_20260717T051244Z",
    "Vector RAG": "locomo_vector_rag_gpt_oss_120b_20260717T050910Z",
    "Long Context": "locomo_long_context_gpt_oss_120b_20260719T043204Z",
    "Nemori": "locomo_nemori_gpt_oss_120b_20260717T045630Z",
    "LightMem": "locomo_lightmem_gpt_oss_120b_20260717T044232Z",
    "MemoryBank": "locomo_memory_bank_gpt_oss_120b_20260718T093223Z",
}

LOCOMO_ABLATIONS = {
    "Full MINDSET": "locomo_mindset_full_gpt_oss_120b_20260717T051244Z",
    "Nearest-schema merge": "locomo_mindset_nearest_schema_merge_gpt_oss_120b_20260719T051220Z",
    "No contextual term": "locomo_mindset_no_contextual_term_gpt_oss_120b_20260719T052237Z",
    "No temporal term": "locomo_mindset_no_temporal_term_gpt_oss_120b_20260719T052734Z",
    "No hysteresis": "locomo_mindset_no_hysteresis_gpt_oss_120b_20260719T053231Z",
    "No fragmentation penalty": "locomo_mindset_no_fragmentation_penalty_gpt_oss_120b_20260719T054344Z",
    "No provenance expansion": "locomo_mindset_no_provenance_expansion_gpt_oss_120b_20260719T054929Z",
}

MAB_MAIN = {
    "MINDSET": "memoryagentbench_mindset_full_gpt_oss_120b_primary_20260717T100608Z",
    "Vector RAG": "memoryagentbench_vector_rag_gpt_oss_120b_primary_20260717T104736Z",
    "Long Context": "memoryagentbench_long_context_gpt_oss_120b_primary_20260717T112052Z",
    "LightMem": "memoryagentbench_lightmem_gpt_oss_120b_primary_20260717T081708Z",
    "MemoryBank": "memoryagentbench_memory_bank_gpt_oss_120b_primary_20260718T162102Z",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_locomo(run_name: str) -> dict[str, dict[str, Any]]:
    run = ROOT / "outputs" / "runs" / run_name
    metrics = read_json(run / "metrics.json")
    predictions = [
        json.loads(line)
        for line in (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    answer = {row["id"]: {"answer_f1": float(row["score"]), "category": int(row["category"])} for row in metrics["per_example"]}
    retrieval = {row["id"]: row for row in predictions}
    if set(answer) != set(retrieval):
        raise AssertionError(f"LoCoMo ID mismatch in {run_name}: {len(answer)} vs {len(retrieval)}")
    output: dict[str, dict[str, Any]] = {}
    for identifier, row in answer.items():
        pred = retrieval[identifier]
        ranked = pred.get("ranked_retrieval", {})
        has_gold_evidence = any(
            part.strip()
            for group in pred.get("gold_evidence_ids", [])
            for part in str(group).split(";")
        )
        output[identifier] = {
            **row,
            "recall@8": float(ranked.get("recall@8", np.nan)),
            "precision@8": float(ranked.get("precision@8", np.nan)),
            "mrr": float(ranked.get("mrr", np.nan)),
            "ndcg@8": float(ranked.get("ndcg@8", np.nan)),
            "evidence_recall": float(pred.get("evidence_recall", np.nan)),
            "has_gold_evidence": has_gold_evidence,
            "cluster": identifier.split(":", 1)[0],
        }
    return output


def load_mab(run_name: str) -> dict[str, dict[str, Any]]:
    run = ROOT / "outputs" / "runs" / run_name
    metrics = read_json(run / "metrics.json")
    output: dict[str, dict[str, Any]] = {}
    for row in metrics["per_example"]:
        identifier = row["id"]
        if "exact_match" in row:
            output[identifier] = {
                "metric": "icl_exact_match",
                "score": float(row["exact_match"]),
                "source": row["source"],
                "cluster": row["source"],
            }
        elif "recall@5" in row:
            output[identifier] = {
                "metric": "recsys_recall@5",
                "score": float(row["recall@5"]),
                "source": row["source"],
                "cluster": row["source"],
            }
        else:
            raise AssertionError(f"Unknown MAB metric row in {run_name}: {row}")
    return output


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        adjusted_value = min(1.0, (len(ordered) - rank) * value)
        running = max(running, adjusted_value)
        adjusted[name] = running
    return adjusted


def stats_for_pair(
    first: np.ndarray,
    second: np.ndarray,
    groups: list[str] | None,
    rng: np.random.Generator,
) -> dict[str, Any]:
    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    if np.isnan(differences).any():
        raise AssertionError("NaN in paired metric")
    n = len(differences)
    if groups is None:
        group_values = differences
        group_sums = None
        group_counts = None
    else:
        labels = np.asarray(groups)
        unique = np.array(sorted(set(labels)))
        group_sums = np.array([differences[labels == label].sum() for label in unique])
        group_counts = np.array([np.sum(labels == label) for label in unique])
        group_values = differences

    point = float(differences.mean())
    exact_permutation_p: float | None = None
    if group_sums is None:
        boot_indices = rng.integers(0, n, size=(BOOTSTRAPS, n))
        boot = differences[boot_indices].mean(axis=1)
        perm_groups = differences
        perm_indices = rng.integers(0, 2, size=(PERMUTATIONS, n), dtype=np.int8)
        perm = ((1 - 2 * perm_indices) * perm_groups).mean(axis=1)
    else:
        g = len(group_sums)
        boot_indices = rng.integers(0, g, size=(BOOTSTRAPS, g))
        boot = group_sums[boot_indices].sum(axis=1) / group_counts[boot_indices].sum(axis=1)
        if g <= 12:
            perm = None
            exact_permutation_p = exact_sign_flip_pvalue(group_sums)
        else:
            perm_signs = rng.integers(0, 2, size=(PERMUTATIONS, g), dtype=np.int8)
            perm = ((1 - 2 * perm_signs) * group_sums[None, :]).sum(axis=1) / group_counts.sum()
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
    if exact_permutation_p is not None:
        p_permutation = exact_permutation_p
    else:
        assert perm is not None
        p_permutation = float(
            (np.count_nonzero(np.abs(perm) >= abs(point)) + 1) / (len(perm) + 1)
        )
    nonzero = differences != 0
    if wilcoxon is None or not nonzero.any():
        p_wilcoxon = 1.0
    else:
        p_wilcoxon = float(wilcoxon(differences[nonzero], zero_method="wilcox", alternative="two-sided", method="auto").pvalue)
    return {
        "n": n,
        "mean_difference": point,
        "median_difference": float(np.median(differences)),
        "bootstrap_ci95": [float(ci_low), float(ci_high)],
        "wins_first": int(np.count_nonzero(differences > 0)),
        "ties": int(np.count_nonzero(differences == 0)),
        "wins_second": int(np.count_nonzero(differences < 0)),
        "permutation_p": p_permutation,
        "wilcoxon_p": p_wilcoxon,
        "clustered": groups is not None,
        "cluster_count": len(set(groups)) if groups is not None else n,
    }


def paired_ids_for_metric(
    reference_rows: dict[str, dict[str, Any]],
    comparison_rows: dict[str, dict[str, Any]],
    metric: str,
) -> list[str]:
    shared = sorted(set(reference_rows) & set(comparison_rows))
    if metric in LOCOMO_RANKING_METRICS:
        shared = [
            identifier
            for identifier in shared
            if reference_rows[identifier].get("has_gold_evidence", False)
            and comparison_rows[identifier].get("has_gold_evidence", False)
        ]
    return shared


def compare_family(
    data: dict[str, dict[str, dict[str, Any]]],
    reference: str,
    metrics: list[str],
    *,
    cluster: bool,
    rng: np.random.Generator,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for metric in metrics:
        metric_results: dict[str, Any] = {}
        for method, rows in data.items():
            if method == reference:
                continue
            shared = paired_ids_for_metric(data[reference], rows, metric)
            first = np.array([data[reference][identifier][metric] for identifier in shared], dtype=float)
            second = np.array([rows[identifier][metric] for identifier in shared], dtype=float)
            groups = [data[reference][identifier]["cluster"] for identifier in shared] if cluster else None
            metric_results[method] = stats_for_pair(first, second, groups, rng)
        adjusted = holm({method: value["permutation_p"] for method, value in metric_results.items()})
        for method, value in metric_results.items():
            value["holm_permutation_p"] = adjusted[method]
            value["significant_holm_0.05"] = adjusted[method] < 0.05
        results[metric] = metric_results
    return results


def format_float(value: float) -> str:
    return f"{value:+.4f}"


def markdown_family(title: str, results: dict[str, Any]) -> list[str]:
    reference_label = "Full MINDSET" if "ablation" in title.lower() else "MINDSET"
    lines = [f"## {title}", "", "| Comparison | Metric | Mean delta | 95% bootstrap CI | Wins-ties-losses | Permutation p | Holm p |", "|---|---|---:|---:|---:|---:|---:|"]
    for metric, comparisons in results.items():
        for method, value in comparisons.items():
            ci = value["bootstrap_ci95"]
            wtl = f'{value["wins_first"]}-{value["ties"]}-{value["wins_second"]}'
            lines.append(
                f"| {reference_label} vs {method} | {metric} | {format_float(value['mean_difference'])} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {wtl} | {value['permutation_p']:.5f} | {value['holm_permutation_p']:.5f} |"
            )
    lines.append("")
    return lines


def main() -> None:
    rng = np.random.default_rng(SEED)
    locomo_metrics = ["answer_f1", "recall@8", "precision@8", "mrr", "ndcg@8", "evidence_recall"]
    main_data = {method: load_locomo(run) for method, run in LOCOMO_MAIN.items()}
    ablation_data = {method: load_locomo(run) for method, run in LOCOMO_ABLATIONS.items()}
    main_results = compare_family(main_data, "MINDSET", locomo_metrics, cluster=True, rng=rng)
    ablation_results = compare_family(ablation_data, "Full MINDSET", locomo_metrics, cluster=True, rng=rng)

    mab_data = {method: load_mab(run) for method, run in MAB_MAIN.items()}
    mab_results: dict[str, Any] = {}
    for metric in ("icl_exact_match", "recsys_recall@5"):
        filtered = {
            method: {identifier: row for identifier, row in rows.items() if row["metric"] == metric}
            for method, rows in mab_data.items()
        }
        # MAB categories are independent instances in the frozen manifest; use paired question-level inference.
        metric_data = {
            method: {
                identifier: {"score": row["score"], "cluster": row["cluster"]}
                for identifier, row in rows.items()
            }
            for method, rows in filtered.items()
        }
        for method in metric_data:
            for identifier, row in list(metric_data[method].items()):
                metric_data[method][identifier][metric] = row.pop("score")
        family = compare_family(metric_data, "MINDSET", [metric], cluster=False, rng=rng)
        mab_results.update(family)

    result = {
        "protocol": "frozen-runs-paired-statistics-v1",
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "permutation_replicates_question_level": PERMUTATIONS,
        "loComo_primary_inference": "conversation-clustered paired bootstrap and two-sided exact sign-flip permutation over all 2^10 assignments; ten conversation clusters",
        "loComo_ranking_sample": "696 questions with nonempty gold evidence; answer F1 and evidence recall use all 700 questions",
        "mab_primary_inference": "paired question-level bootstrap and sign-flip permutation",
        "multiple_testing": "Holm correction within each metric family",
        "main_locomo": main_results,
        "ablation_locomo": ablation_results,
        "main_mab": mab_results,
        "run_manifests": {"locomo_main": LOCOMO_MAIN, "locomo_ablations": LOCOMO_ABLATIONS, "mab_main": MAB_MAIN},
    }
    (OUT / "statistical_tests_v1.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    markdown: list[str] = [
        "# Statistical Tests for MINDSET Experiments",
        "",
        "All tests use paired per-question scores from frozen run artifacts. LoCoMo ranking metrics use the 696 questions with nonempty gold evidence; answer F1 and evidence recall use all 700. The primary bootstrap resamples the ten conversation clusters and the two-sided exact sign-flip test enumerates all 2^10 cluster assignments. MAB uses paired question-level resampling. Positive deltas mean MINDSET/full MINDSET scores higher. P-values are Holm-adjusted within each metric family.",
        "",
    ]
    markdown += markdown_family("Main LoCoMo: MINDSET vs baseline", main_results)
    markdown += markdown_family("LoCoMo ablations: Full MINDSET vs ablation", ablation_results)
    markdown += markdown_family("Main MAB: MINDSET vs baseline", mab_results)
    (OUT / "statistical_tests_v1.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps({"output_json": str(OUT / "statistical_tests_v1.json"), "output_markdown": str(OUT / "statistical_tests_v1.md")}, indent=2))


if __name__ == "__main__":
    main()



