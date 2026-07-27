from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the local MINDSET-energy retrieval gate")
    parser.add_argument("--config", default="configs/kill_test.yaml")
    parser.add_argument("--state", default="outputs/orchestration/six_method_kill120_v2.json")
    parser.add_argument("--output", default="outputs/orchestration/six_method_kill_decision_v2.json")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    state = json.loads((ROOT / args.state).read_text(encoding="utf-8"))
    if state.get("status") != "complete":
        raise RuntimeError("Retrieval gate cannot be evaluated before all local cells complete")

    runs = discover_runs(ROOT / "outputs/runs", config["repetition_id"])
    missing = [method for method in config["methods"] if method not in runs]
    if missing:
        raise RuntimeError("Missing retrieval cells: " + ", ".join(missing))
    metrics = {method: item["metrics"] for method, item in runs.items()}
    baselines = list(config["retrieval_gate"]["comparison_baselines"])
    metric_name = config["retrieval_gate"]["metric"]
    mindset_value = float(metrics["mindset"][metric_name])
    primary = max(baselines, key=lambda method: float(metrics[method][metric_name]))
    baseline_value = float(metrics[primary][metric_name])
    deficit = baseline_value - mindset_value
    packet_ratio = ratio(
        float(metrics["mindset"]["mean_evidence_packet_bytes"]),
        float(metrics[primary]["mean_evidence_packet_bytes"]),
    )
    efficient_exception = (
        deficit <= config["retrieval_gate"]["maximum_recall_deficit_with_efficiency_exception"]
        and packet_ratio
        <= config["retrieval_gate"]["maximum_packet_byte_ratio_for_efficiency_exception"]
    )
    passed = mindset_value >= baseline_value or efficient_exception

    comparisons: dict[str, Any] = {}
    for method in config["methods"]:
        if method == "mindset":
            continue
        comparisons[f"mindset_minus_{method}"] = compare_metrics(
            metrics["mindset"], metrics[method]
        )
    payload = {
        "protocol": config["protocol"],
        "decision": "pass" if passed else "fail",
        "answer_generation_authorized": bool(passed),
        "network_calls_made": False,
        "gate": {
            "comparison_baselines": baselines,
            "strongest_baseline": primary,
            "metric": metric_name,
            "mindset": mindset_value,
            "baseline": baseline_value,
            "absolute_delta": mindset_value - baseline_value,
            "packet_byte_ratio": packet_ratio,
            "efficiency_exception": efficient_exception,
        },
        "methods": {
            method: {
                "run_dir": str(item["path"].resolve()),
                "metrics": metrics[method],
            }
            for method, item in sorted(runs.items())
        },
        "comparisons": comparisons,
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
        if metrics.get("status") != "retrieval_smoke_complete":
            continue
        method = str(arguments["method"])
        previous = discovered.get(method)
        if previous is None or path.name > previous["path"].name:
            discovered[method] = {"path": path, "metrics": metrics}
    return discovered


def compare_metrics(mindset: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    result = {
        "mean_evidence_recall_at_k": float(mindset["mean_evidence_recall_at_k"])
        - float(baseline["mean_evidence_recall_at_k"]),
        "mean_candidate_pool_recall": float(mindset["mean_candidate_pool_recall"])
        - float(baseline["mean_candidate_pool_recall"]),
        "mean_evidence_packet_bytes": float(mindset["mean_evidence_packet_bytes"])
        - float(baseline["mean_evidence_packet_bytes"]),
    }
    for name, value in mindset.get("answerable_retrieval", {}).items():
        if name in baseline.get("answerable_retrieval", {}):
            result[f"answerable_{name}"] = float(value) - float(
                baseline["answerable_retrieval"][name]
            )
    return result


def ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("inf") if numerator else 1.0
    return numerator / denominator


if __name__ == "__main__":
    main()
