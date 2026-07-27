from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "outputs/plans/locomo_cross_backbone_final700_v1.json"
DEFAULT_STATE = ROOT / "outputs/orchestration/locomo_cross_backbone_final700_v1.json"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def task_list(plan: dict) -> list[dict[str, str]]:
    return [
        {"model": model, "method": method}
        for model in plan["models"]
        for method in plan["methods"]
    ]


def command_for(plan: dict, model: str, method: str) -> list[str]:
    shared = plan["shared"]
    settings = plan["method_settings"][method]
    command = [
        str(ROOT / ".venv/Scripts/python.exe"),
        str(ROOT / "scripts/run_locomo_experiment.py"),
        "--method", method,
        "--model", model,
        "--manifest", plan["manifest"],
        "--repetition-id", settings["repetition_id"],
        "--mindset-retrieval-mode", shared["retrieval_mode"],
        "--candidate-k", str(shared["candidate_k"]),
        "--top-k", str(shared["top_k"]),
        "--batch-size", str(settings["batch_size"]),
        "--max-prompt-bytes", str(shared["max_prompt_bytes"]),
        "--max-output-tokens", str(shared["max_output_tokens"]),
        "--max-retries", str(shared["max_retries"]),
    ]
    if method == "long_context":
        command.extend(["--long-context-max-history-bytes", str(settings["long_context_max_history_bytes"])])
    if shared.get("bounded_evidence") and method != "long_context":
        command.extend([
            "--bounded-evidence",
            "--evidence-candidate-k", str(shared["evidence_candidate_k"]),
            "--evidence-chunk-size-tokens", str(shared["evidence_chunk_size_tokens"]),
            "--evidence-chunk-overlap-tokens", str(shared["evidence_chunk_overlap_tokens"]),
            "--evidence-max-chunks", str(shared["evidence_max_chunks"]),
            "--evidence-token-budget", str(shared["evidence_token_budget"]),
            "--evidence-max-chunks-per-memory", str(shared["evidence_max_chunks_per_memory"]),
        ])
        if method == "memory_bank":
            command.extend(["--memory-bank-summary-prompt-chars", str(settings["memory_bank_summary_prompt_chars"])])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable LoCoMo-700 cross-backbone runner")
    parser.add_argument("--plan", default=str(DEFAULT_PLAN.relative_to(ROOT)))
    parser.add_argument("--state", default=str(DEFAULT_STATE.relative_to(ROOT)))
    parser.add_argument("--skip-task", action="append", default=[])
    args = parser.parse_args()

    plan_path = ROOT / args.plan
    state_path = ROOT / args.state
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    tasks = task_list(plan)
    task_keys = [f"{item['model']}:{item['method']}" for item in tasks]
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
        "protocol": plan["protocol"],
        "plan": str(plan_path.relative_to(ROOT)),
        "dataset": plan["dataset"],
        "question_count": plan["question_count"],
        "models": plan["models"],
        "methods": plan["methods"],
        "task_keys": task_keys,
        "completed": [],
        "attempts": [],
        "status": "ready",
    }
    if state.get("task_keys") != task_keys or state.get("question_count") != plan["question_count"]:
        raise RuntimeError("State does not match the frozen cross-backbone plan")

    state["status"] = "running"
    state.pop("failed_task", None)
    state["started_or_resumed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(state_path, state)

    child_env = os.environ.copy()
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        child_env.pop(name, None)

    # Keep Python/PyTorch/SQLite temporary files on the experiment drive. The
    # provider and persistent caches are already under E:\MINDSET, but Windows
    # otherwise defaults TEMP/TMP to C:, which can make local reranking fail
    # with a misleading "database or disk is full" error when C: is exhausted.
    scratch_dir = ROOT / "tmp"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    for name in ("TEMP", "TMP", "TMPDIR"):
        child_env[name] = str(scratch_dir)
    for task in tasks:
        key = f"{task['model']}:{task['method']}"
        if key in state.get("completed", []):
            continue
        if key in set(args.skip_task) or key in set(state.get("skipped", [])):
            state.setdefault("skipped", [])
            if key not in state["skipped"]:
                state["skipped"].append(key)
                write_json(state_path, state)
            print(f"SKIP {key}", flush=True)
            continue
        command = command_for(plan, task["model"], task["method"])
        print(f"START {key} (LoCoMo-{plan['question_count']})", flush=True)
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        result = subprocess.run(command, cwd=ROOT, env=child_env, check=False)
        state.setdefault("attempts", []).append({
            "task": key,
            "model": task["model"],
            "method": task["method"],
            "command": command,
            "started_at_utc": started,
            "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exit_code": result.returncode,
        })
        if result.returncode != 0:
            state["status"] = "paused_on_failure"
            state["failed_task"] = key
            write_json(state_path, state)
            raise SystemExit(result.returncode)
        state.setdefault("completed", []).append(key)
        state["status"] = "running"
        write_json(state_path, state)
        print(f"DONE {key} ({len(state['completed'])}/{len(tasks)})", flush=True)

    state["status"] = "complete_with_skips" if state.get("skipped") else "complete"
    state["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(state_path, state)
    print("LoCoMo cross-backbone final700 protocol complete.", flush=True)


if __name__ == "__main__":
    main()
