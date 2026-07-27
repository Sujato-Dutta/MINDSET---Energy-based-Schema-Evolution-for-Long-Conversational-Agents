from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindset.api.backend import APIBackend


MODELS = ("gpt_oss_120b", "qwen_27b", "llama_scout_17b")


def _latest_complete(model_name: str) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(
        (ROOT / "outputs" / "runs").glob(f"locomo_vector_rag_smoke_{model_name}_*")
    )
    for run_dir in reversed(candidates):
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics.get("status") == "complete":
                return run_dir, metrics
    raise FileNotFoundError(f"No complete smoke run found for {model_name}")


def main() -> None:
    backend = APIBackend.from_file(ROOT / "configs" / "models.yaml")
    ledger = backend.ledger.report()
    ledger_models = {row["model_name"]: row for row in ledger["models"]}
    runs = []
    for model_name in MODELS:
        run_dir, metrics = _latest_complete(model_name)
        model = backend.config.models[model_name]
        usage = ledger_models[model_name]
        logical_tokens = metrics.get("logical_answer_tokens")
        if logical_tokens is None:
            predictions = [
                json.loads(line)
                for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            logical_tokens = {
                "input_tokens": sum(int(row["input_tokens"]) for row in predictions),
                "output_tokens": sum(int(row["output_tokens"]) for row in predictions),
            }
            logical_tokens["total_tokens"] = (
                logical_tokens["input_tokens"] + logical_tokens["output_tokens"]
            )
        runs.append(
            {
                "model_alias": model_name,
                "provider": model.provider,
                "model_id": model.model_id,
                "request_options": model.request_options,
                "run_id": metrics["run_id"],
                "run_dir": str(run_dir.resolve()),
                "question_count": metrics["completed_count"],
                "native_f1": metrics["native_f1"],
                "evidence_recall_at_8": metrics["mean_evidence_recall_at_k"],
                "logical_answer_tokens": logical_tokens,
                "cumulative_ledger_tokens": {
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "total_tokens": usage["input_tokens"] + usage["output_tokens"],
                    "hard_budget": usage["hard_budget"],
                    "remaining_budget": usage["hard_budget"]
                    - usage["input_tokens"]
                    - usage["output_tokens"],
                },
            }
        )
    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "Phase-3 Vector RAG smoke gate; not full benchmark performance.",
        "manifest": "data/manifests/locomo_smoke20.json",
        "runs": runs,
        "key_ledger": ledger["keys"],
    }
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "outputs" / "logs" / f"smoke_summary_{timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
