from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen 40-question energy answer gate"
    )
    parser.add_argument("--config", default="configs/kill_test.yaml")
    parser.add_argument(
        "--state", default="outputs/orchestration/six_method_answer40_v2.json"
    )
    parser.add_argument(
        "--output",
        default="outputs/orchestration/six_method_answer40_decision_v2.json",
    )
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    gate = config["answer_check_after_gate"]
    state = json.loads((ROOT / args.state).read_text(encoding="utf-8"))
    if state.get("status") != "complete":
        raise RuntimeError("Answer gate cannot be evaluated before all cells complete")

    runs = discover_runs(ROOT / "outputs/runs", gate["repetition_id"])
    missing = [method for method in gate["methods"] if method not in runs]
    if missing:
        raise RuntimeError("Missing answer cells: " + ", ".join(missing))
    candidate_method = "mindset"
    baseline_methods = [method for method in gate["methods"] if method != candidate_method]
    metric = str(gate["promotion_metric"])
    baseline_method = max(
        baseline_methods,
        key=lambda method: float(runs[method]["metrics"][metric]),
    )
    baseline = runs[baseline_method]["metrics"]
    candidate = runs[candidate_method]["metrics"]
    delta = float(candidate[metric]) - float(baseline[metric])
    minimum = float(gate["minimum_absolute_delta"])
    paired = paired_differences(candidate, baseline)
    interval = bootstrap_interval(paired, seed=20260716, repetitions=20_000)
    passed = delta >= minimum
    payload = {
        "protocol": config["protocol"],
        "decision": "pass" if passed else "fail",
        "final_paper_runs_authorized": bool(passed),
        "metric": metric,
        "minimum_absolute_delta": minimum,
        "mindset": float(candidate[metric]),
        "strongest_baseline": baseline_method,
        "baseline_scores": {
            method: float(runs[method]["metrics"][metric])
            for method in baseline_methods
        },
        "absolute_delta": delta,
        "relative_delta": delta / float(baseline[metric]) if baseline[metric] else None,
        "paired_bootstrap_95_interval": interval,
        "question_count": len(paired),
        "answer_model_network_calls_made": True,
        "judge_calls_made": False,
        "runs": {
            method: {
                "run_dir": str(item["path"].resolve()),
                "metrics": item["metrics"],
            }
            for method, item in sorted(runs.items())
        },
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def discover_runs(root: Path, repetition_id: str) -> dict[str, dict[str, Any]]:
    discovered: dict[str, dict[str, Any]] = {}
    for path in root.iterdir():
        config_path, metrics_path = path / "run_config.json", path / "metrics.json"
        if not config_path.exists() or not metrics_path.exists():
            continue
        run_config = json.loads(config_path.read_text(encoding="utf-8"))
        arguments = run_config.get("arguments", {})
        if arguments.get("repetition_id") != repetition_id:
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("status") != "complete":
            continue
        method = str(arguments["method"])
        previous = discovered.get(method)
        if previous is None or path.name > previous["path"].name:
            discovered[method] = {"path": path, "metrics": metrics}
    return discovered


def paired_differences(candidate: dict[str, Any], baseline: dict[str, Any]) -> list[float]:
    candidate_rows = {row["id"]: float(row["score"]) for row in candidate["per_example"]}
    baseline_rows = {row["id"]: float(row["score"]) for row in baseline["per_example"]}
    if candidate_rows.keys() != baseline_rows.keys():
        raise ValueError("Answer cells do not contain the same question IDs")
    return [candidate_rows[key] - baseline_rows[key] for key in sorted(candidate_rows)]


def bootstrap_interval(
    differences: list[float], *, seed: int, repetitions: int
) -> list[float]:
    if not differences:
        return [0.0, 0.0]
    generator = random.Random(seed)
    size = len(differences)
    means = sorted(
        sum(differences[generator.randrange(size)] for _ in range(size)) / size
        for _ in range(repetitions)
    )
    return [means[int(0.025 * repetitions)], means[int(0.975 * repetitions) - 1]]


if __name__ == "__main__":
    main()
