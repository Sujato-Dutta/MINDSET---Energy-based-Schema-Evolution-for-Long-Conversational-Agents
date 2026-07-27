from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "outputs/plans/memorybank_final1000_v1.json"
STATE = ROOT / "outputs/orchestration/memorybank_final1000_v1.json"


def write_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def command_for(task: dict) -> list[str]:
    python = str(ROOT / ".venv/Scripts/python.exe")
    common = [
        python,
        str(ROOT / task["runner"]),
        "--method", "memory_bank",
        "--model", "gpt_oss_120b",
        "--manifest", task["manifest"],
        "--repetition-id", "primary",
        "--bounded-evidence",
        "--evidence-candidate-k", "20",
        "--evidence-chunk-size-tokens", "768",
        "--evidence-chunk-overlap-tokens", "96",
        "--evidence-max-chunks", "8",
        "--evidence-token-budget", "8000",
        "--evidence-max-chunks-per-memory", "3",
        "--memory-bank-summary-prompt-chars", "16000",
        "--batch-size", "8",
    ]
    if task["dataset"] == "locomo":
        common.extend(["--max-prompt-bytes", "160000"])
    elif task["dataset"] == "memoryagentbench":
        common = [common[0], common[1], "--dataset", task["dataset"], *common[2:]]
        common[common.index("--batch-size") + 1] = "1"
        common.extend(["--max-prompt-bytes", "160000"])
    else:
        common = [common[0], common[1], "--dataset", task["dataset"], *common[2:]]
        common.extend(["--max-prompt-bytes", "160000", "--proxy-judge"])
    return common


def main() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {
        "plan": str(PLAN.relative_to(ROOT)),
        "plan_config_sha256": plan["config_sha256"],
        "completed": [],
        "attempts": [],
    }
    if state.get("plan_config_sha256") != plan["config_sha256"]:
        raise RuntimeError("State belongs to a different frozen MemoryBank plan")
    state["status"] = "running"
    write_state(state)
    for task in plan["tasks"]:
        if task["task_id"] in state.get("completed", []):
            continue
        command = command_for(task)
        print(f"START {task['task_id']} primary {task['dataset']}:memory_bank", flush=True)
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        completed = subprocess.run(command, cwd=ROOT, check=False)
        state.setdefault("attempts", []).append({
            "task_id": task["task_id"],
            "command": command,
            "started_at_utc": started,
            "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exit_code": completed.returncode,
        })
        if completed.returncode != 0:
            state["status"] = "paused_on_failure_or_quota"
            write_state(state)
            raise SystemExit(completed.returncode)
        state.setdefault("completed", []).append(task["task_id"])
        state["status"] = "running"
        write_state(state)
    state["status"] = "complete"
    write_state(state)
    print("MemoryBank final1000 protocol complete.", flush=True)


if __name__ == "__main__":
    main()