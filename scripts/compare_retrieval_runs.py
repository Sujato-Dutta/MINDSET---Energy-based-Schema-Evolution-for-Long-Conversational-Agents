from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindset.statistics.tests import paired_bootstrap


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalized_source(identifier: str) -> str:
    return identifier.split(":", 1)[-1]


def fixed_budget_recall(row: dict[str, Any], source_budget: int) -> float:
    selected: list[str] = []
    for hit in row["hits"]:
        for source in hit["source_event_ids"]:
            identifier = normalized_source(str(source))
            if identifier not in selected and len(selected) < source_budget:
                selected.append(identifier)
        if len(selected) == source_budget:
            break
    gold = [str(value) for value in row["gold_evidence_ids"]]
    if not gold:
        return 1.0
    return sum(identifier in selected for identifier in gold) / len(gold)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare retrieval runs under raw-unit and fixed-source budgets"
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="LABEL=RUN_DIRECTORY; repeat for every method",
    )
    parser.add_argument("--source-budget", type=int, default=8)
    parser.add_argument(
        "--output",
        default="outputs/evaluation/heldout_retrieval_v1.json",
    )
    args = parser.parse_args()
    if args.source_budget <= 0:
        raise ValueError("source-budget must be positive")

    runs: dict[str, Path] = {}
    for value in args.run:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            raise ValueError(f"Invalid --run value: {value}")
        runs[label] = Path(path)

    expected_ids: list[str] | None = None
    manifest_sha256: str | None = None
    methods: dict[str, Any] = {}
    fixed_scores_by_method: dict[str, list[float]] = {}
    for label, run_dir in runs.items():
        rows = load_jsonl(run_dir / "retrieval.jsonl")
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        identifiers = [str(row["id"]) for row in rows]
        if expected_ids is None:
            expected_ids = identifiers
        elif identifiers != expected_ids:
            raise ValueError(f"Question IDs differ for {label}")
        current_manifest_hash = str(config["manifest_sha256"])
        if manifest_sha256 is None:
            manifest_sha256 = current_manifest_hash
        elif current_manifest_hash != manifest_sha256:
            raise ValueError(f"Manifest hash differs for {label}")

        source_counts = [
            len(set(str(value) for value in row["retrieved_ids"])) for row in rows
        ]
        raw_scores = [float(row["evidence_recall"]) for row in rows]
        fixed_scores = [fixed_budget_recall(row, args.source_budget) for row in rows]
        fixed_scores_by_method[label] = fixed_scores
        methods[label] = {
            "run_dir": str(run_dir.resolve()),
            "question_count": len(rows),
            "raw_mean_evidence_recall": sum(raw_scores) / max(1, len(raw_scores)),
            "fixed_source_budget_recall": sum(fixed_scores) / max(1, len(fixed_scores)),
            "mean_unique_source_events": sum(source_counts) / max(1, len(source_counts)),
            "max_unique_source_events": max(source_counts, default=0),
        }

    comparisons = {}
    if "mindset" in fixed_scores_by_method:
        for label, scores in fixed_scores_by_method.items():
            if label == "mindset":
                continue
            comparisons[f"mindset_minus_{label}"] = paired_bootstrap(
                fixed_scores_by_method["mindset"],
                scores,
                resamples=10_000,
                seed=0,
            )
    report = {
        "comparison_version": "heldout-retrieval-v1",
        "manifest_sha256": manifest_sha256,
        "source_event_budget": args.source_budget,
        "network_calls_made": False,
        "methods": methods,
        "paired_bootstrap_fixed_source_budget": comparisons,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
