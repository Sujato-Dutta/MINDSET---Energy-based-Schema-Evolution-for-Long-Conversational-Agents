"""Score frozen SubtleMemory predictions without rerunning answer methods.

The deterministic scores use the best matching reference from the benchmark's
multiple valid correct answers. The optional Gemini pass follows the existing
blinded proxy-judge prompt and is recorded separately from lexical F1.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from mindset.api.backend import APIBackend
from mindset.evaluation.metrics import exact_match, token_f1
from mindset.evaluation.proxy_judge import judge_with_proxy_consensus


ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "outputs/runs"
OUTPUT_ROOT = ROOT / "outputs/metrics"
ROWS_ROOT = OUTPUT_ROOT / "subtlememory_answer_quality_rows"
MODEL_CONFIG = ROOT / "configs/models.yaml"
METHODS = ("long_context", "vector_rag", "hybrid_rag", "nemori", "lightmem", "mindset")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_run(method: str) -> Path:
    candidates = []
    for path in RUNS_ROOT.glob("subtlememory_*_primary_*"):
        predictions = path / "predictions.jsonl"
        config_path = path / "run_config.json"
        if not predictions.exists() or not config_path.exists():
            continue
        config = read_json(config_path)
        if config.get("method") == method and config.get("status") != "preflight_passed":
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No completed SubtleMemory predictions found for {method}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_predictions(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(rows: list[dict[str, Any]], telemetry: list[dict[str, Any]]) -> dict[str, Any]:
    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_relation[str(row["relation_type"])].append(row)

    def deterministic(group: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "exact_match": sum(float(row["deterministic_exact_match"]) for row in group) / len(group),
            "token_f1": sum(float(row["deterministic_token_f1"]) for row in group) / len(group),
        }

    verdicts = Counter(str(row["judge_verdict"]) for row in rows)
    agreements = sum(bool(row["judge_agreement"]) for row in rows)
    quality = sum(str(row["judge_verdict"]) in {"correct", "partially_correct"} for row in rows)
    strict = sum(str(row["judge_verdict"]) == "correct" for row in rows)
    return {
        "question_count": len(rows),
        "deterministic": deterministic(rows),
        "deterministic_by_relation": {
            relation: deterministic(group) for relation, group in sorted(by_relation.items())
        },
        "proxy_judge": {
            "quality_rate_correct_or_partial": quality / len(rows),
            "strict_correct_rate": strict / len(rows),
            "verdict_counts": dict(sorted(verdicts.items())),
            "agreement_rate": agreements / len(rows),
            "quality_rate_by_relation": {
                relation: sum(
                    str(row["judge_verdict"]) in {"correct", "partially_correct"}
                    for row in group
                ) / len(group)
                for relation, group in sorted(by_relation.items())
            },
            "protocol": "external_gemini_proxy_v2_not_official_human_gold",
            "model": "gemini_judge",
        },
        "judge_efficiency": {
            "calls": len(telemetry),
            "input_tokens": sum(int(row.get("billed_input_tokens", 0)) for row in telemetry),
            "output_tokens": sum(int(row.get("billed_output_tokens", 0)) for row in telemetry),
            "total_tokens": sum(
                int(row.get("billed_input_tokens", 0)) + int(row.get("billed_output_tokens", 0))
                for row in telemetry
            ),
        },
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ROWS_ROOT.mkdir(parents=True, exist_ok=True)
    backend = APIBackend.from_file(MODEL_CONFIG)
    methods: dict[str, Any] = {}

    for method in METHODS:
        run_dir = latest_run(method)
        predictions = load_predictions(run_dir / "predictions.jsonl")
        telemetry: list[dict[str, Any]] = []
        scored: list[dict[str, Any]] = []
        for index, row in enumerate(predictions, start=1):
            references = row["answer"]
            decision = judge_with_proxy_consensus(
                backend,
                question_id=str(row["id"]),
                question=str(row["question"]),
                reference=str(references),
                prediction=str(row["prediction"]),
                telemetry=telemetry,
                code_version="subtlememory-answer-quality-v1",
                request_metadata={
                    "run_id": run_dir.name,
                    "dataset": "subtlememory",
                    "method": method,
                    "repetition_id": "primary",
                },
            )
            scored.append({
                "id": row["id"],
                "relation_type": row["relation_type"],
                "deterministic_exact_match": exact_match(row["prediction"], references),
                "deterministic_token_f1": token_f1(row["prediction"], references),
                "judge_verdict": decision.verdict,
                "judge_agreement": decision.agreement,
                "judge_verdicts": list(decision.judge_verdicts),
            })
            if index % 25 == 0:
                print(f"{method}: judged {index}/{len(predictions)}", flush=True)

        rows_path = ROWS_ROOT / f"{method}.jsonl"
        rows_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in scored),
            encoding="utf-8",
        )
        methods[method] = {
            "run_dir": str(run_dir.relative_to(ROOT)),
            "metrics": summarize(scored, telemetry),
        }
        print(f"{method}: complete", flush=True)

    output = {
        "protocol": "subtlememory-final150-v1",
        "dataset": "subtlememory",
        "question_count_per_method": 150,
        "reference_policy": "best match over all benchmark correct_answers",
        "deterministic_metric": "normalized exact match and max-reference token F1",
        "judge_policy": "blinded Gemini proxy judge; correct and partially_correct count as quality success",
        "methods": methods,
    }
    (OUTPUT_ROOT / "subtlememory_answer_quality_v1.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({method: data["metrics"] for method, data in methods.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
