from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "outputs/plans/locomo_mindset_ablations700_v1.json"
DEFAULT_STATE = ROOT / "outputs/orchestration/locomo_mindset_ablations700_v1.json"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command_for(plan: dict, ablation: str, batch_size: int | None = None) -> list[str]:
    generation = plan["generation"]
    return [
        str(ROOT / ".venv/Scripts/python.exe"),
        str(ROOT / "scripts/run_locomo_experiment.py"),
        "--method", plan["method"],
        "--ablation", ablation,
        "--model", plan["model"],
        "--manifest", plan["manifest"],
        "--repetition-id", plan["repetition_id"],
        "--mindset-retrieval-mode", plan["retrieval_mode"],
        "--batch-size", str(batch_size or generation["batch_size"]),
        "--max-prompt-bytes", str(generation["max_prompt_bytes"]),
        "--max-output-tokens", str(generation["max_output_tokens"]),
        "--max-retries", str(generation["max_retries"]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable LoCoMo-700 MINDSET ablation runner")
    parser.add_argument("--plan", default=str(DEFAULT_PLAN.relative_to(ROOT)))
    parser.add_argument("--state", default=str(DEFAULT_STATE.relative_to(ROOT)))
    parser.add_argument("--resume", action="store_true", help="Resume the saved state")
    args = parser.parse_args()

    plan_path = ROOT / args.plan
    state_path = ROOT / args.state
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    expected = list(plan["ablations"])
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
        "protocol": plan["protocol"],
        "plan": str(plan_path.relative_to(ROOT)),
        "dataset": plan["dataset"],
        "question_count": plan["question_count"],
        "model": plan["model"],
        "ablations": expected,
        "completed": [],
        "attempts": [],
        "status": "ready",
    }
    if state.get("ablations") != expected or state.get("model") != plan["model"]:
        raise RuntimeError("State does not match the frozen ablation plan")

    state["status"] = "running"
    state["started_or_resumed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(state_path, state)

    child_env = os.environ.copy()
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        child_env.pop(name, None)

    for ablation in expected:
        if ablation in state.get("completed", []):
            continue
        command = command_for(plan, ablation)
        print(f"START {ablation} (LoCoMo-700, GPT-OSS-120B, batch=8)", flush=True)
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        result = subprocess.run(command, cwd=ROOT, env=child_env, check=False)
        attempt = {
            "ablation": ablation,
            "command": command,
            "batch_size": plan["generation"]["batch_size"],
            "started_at_utc": started,
            "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exit_code": result.returncode,
        }
        state.setdefault("attempts", []).append(attempt)
        if result.returncode != 0:
            state["status"] = "paused_on_failure"
            state["failed_ablation"] = ablation
            write_json(state_path, state)
            raise SystemExit(result.returncode)
        state.setdefault("completed", []).append(ablation)
        state["status"] = "running"
        write_json(state_path, state)
        print(f"DONE {ablation} ({len(state['completed'])}/{len(expected)})", flush=True)

    state["status"] = "complete"
    state["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(state_path, state)
    print("LoCoMo-700 MINDSET ablations complete.", flush=True)


if __name__ == "__main__":
    main()
