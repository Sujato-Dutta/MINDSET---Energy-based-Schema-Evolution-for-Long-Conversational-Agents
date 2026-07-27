from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ANSWER_MODELS = {"gpt_oss_120b", "qwen_27b", "llama_scout_17b", "glm_47", "gemma_4_31b"}
SCOPE_ORDER = {
    "primary": 0,
    "primary_ablation": 1,
    "targeted_appendix": 2,
    "cross_model": 3,
    "repeatability": 4,
    "local_only_diagnostic": 5,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume the final paper protocol after both development gates pass"
    )
    parser.add_argument("--plan", default="outputs/plans/main_final_v2.json")
    parser.add_argument("--state", default="outputs/orchestration/paper_protocol_final_v2.json")
    parser.add_argument(
        "--kill-decision",
        default="outputs/orchestration/six_method_answer40_decision_v2.json",
    )
    parser.add_argument('--dataset', choices=('locomo', 'longmemeval', 'memoryagentbench'), help='Run only tasks for this dataset while preserving the shared protocol state')
    parser.add_argument('--skip-method', action='append', default=[], help='Temporarily omit method tasks without marking them completed')
    args = parser.parse_args()

    decision = json.loads((ROOT / args.kill_decision).read_text(encoding="utf-8"))
    if decision.get("decision") != "pass" or not decision.get(
        "final_paper_runs_authorized"
    ):
        raise RuntimeError(
            "Primary paper runs remain locked until the frozen answer-level gate passes"
        )

    plan = json.loads((ROOT / args.plan).read_text(encoding="utf-8"))
    state_path = ROOT / args.state
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {
            "plan": args.plan,
            "plan_config_sha256": plan["config_sha256"],
            "completed": [],
            "attempts": [],
        }
    )
    if state["plan_config_sha256"] != plan["config_sha256"]:
        raise RuntimeError("State belongs to a different frozen plan")

    if args.dataset is not None:
        plan['tasks'] = [task for task in plan['tasks'] if task['dataset'] == args.dataset]
    if args.skip_method:
        plan['tasks'] = [task for task in plan['tasks'] if task['method'] not in set(args.skip_method)]
    tasks = sorted(
        plan["tasks"],
        key=lambda task: (
            SCOPE_ORDER.get(task["scope"], 99),
            task["method"] == "long_context",
            task["dataset"],
            task["task_id"],
        ),
    )
    for task in tasks:
        if task["task_id"] in state["completed"]:
            continue
        command = build_command(task)
        print(
            f"START {task['task_id']} {task['scope']} "
            f"{task['dataset']}:{task['method']}:{task['ablation']}",
            flush=True,
        )
        started_at = dt.datetime.now(dt.timezone.utc).isoformat()
        completed = subprocess.run(command, cwd=ROOT, check=False)
        state["attempts"].append(
            {
                "task_id": task["task_id"],
                "command": command,
                "started_at_utc": started_at,
                "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "exit_code": completed.returncode,
            }
        )
        if completed.returncode != 0:
            state["status"] = "paused_on_failure_or_daily_quota"
            _write_state(state_path, state)
            raise SystemExit(completed.returncode)
        state["completed"].append(task["task_id"])
        state["status"] = "running"
        _write_state(state_path, state)
    if args.dataset is not None or args.skip_method:
        state['status'] = 'running'
        _write_state(state_path, state)
        print(f'Scoped protocol complete: dataset={args.dataset} skip_method={args.skip_method}.', flush=True)
        return
    state["status"] = "complete"
    _write_state(state_path, state)
    print("Final paper protocol complete.", flush=True)


def build_command(task: dict[str, Any]) -> list[str]:
    if task["model"] not in ANSWER_MODELS:
        raise ValueError(
            f"{task['model']} is not an answer model; evaluation-only judges cannot run paper tasks"
        )
    if task["dataset"] == "locomo":
        command = [
            str(ROOT / ".venv/Scripts/python.exe"),
            str(ROOT / "scripts/run_locomo_experiment.py"),
            "--method",
            task["method"],
            "--model",
            task["model"],
            "--manifest",
            task["manifest"],
            "--repetition-id",
            task["repetition_id"],
        ]
    else:
        command = [
            str(ROOT / ".venv/Scripts/python.exe"),
            str(ROOT / "scripts/run_aux_benchmark.py"),
            "--dataset",
            task["dataset"],
            "--method",
            task["method"],
            "--model",
            task["model"],
            "--manifest",
            task["manifest"],
            "--repetition-id",
            task["repetition_id"],
        ]
        if task["dataset"] == "longmemeval":
            command.append("--proxy-judge")
    if task["ablation"] != "none":
        command.extend(["--ablation", task["ablation"]])
    if task["method"] == "nemori":
        command.extend(["--batch-size", "4"])
    if task["dataset"] == "memoryagentbench":
        # Keep the frozen sample/method/model protocol, but avoid provider TPM
        # failures from the large default answer batches.
        command.extend(["--batch-size", "1"])
        command.extend(["--max-prompt-bytes", "160000"])
        command.extend([
            "--bounded-evidence",
            "--evidence-candidate-k", "20",
            "--evidence-chunk-size-tokens", "768",
            "--evidence-chunk-overlap-tokens", "96",
            "--evidence-max-chunks", "8",
            "--evidence-token-budget", "8000",
            "--evidence-max-chunks-per-memory", "3",
        ])
    if task["scope"] == "local_only_diagnostic":
        command.append("--retrieval-only")
    return command


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

