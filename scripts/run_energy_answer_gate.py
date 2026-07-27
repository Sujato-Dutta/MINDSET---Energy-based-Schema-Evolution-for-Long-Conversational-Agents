from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the gated 40-question GPT-OSS answer comparison"
    )
    parser.add_argument("--config", default="configs/kill_test.yaml")
    parser.add_argument(
        "--retrieval-decision",
        default="outputs/orchestration/six_method_kill_decision_v2.json",
    )
    parser.add_argument(
        "--state", default="outputs/orchestration/six_method_answer40_v2.json"
    )
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    gate = config["answer_check_after_gate"]
    decision = json.loads(
        (ROOT / args.retrieval_decision).read_text(encoding="utf-8")
    )
    if not gate.get("enabled"):
        raise RuntimeError("The answer-level gate is disabled")
    if gate.get("requires_explicit_gate_pass") and not decision.get(
        "answer_generation_authorized"
    ):
        raise RuntimeError("Answer generation remains locked by the retrieval gate")

    manifest = json.loads((ROOT / gate["manifest"]).read_text(encoding="utf-8"))
    if int(manifest.get("count", -1)) != int(gate["question_count"]):
        raise ValueError("Answer manifest count does not match the frozen protocol")
    state_path = ROOT / args.state
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {
            "protocol": config["protocol"],
            "repetition_id": gate["repetition_id"],
            "manifest": gate["manifest"],
            "retrieval_decision": args.retrieval_decision,
            "completed": [],
            "attempts": [],
            "answer_model_network_calls_permitted": True,
            "judge_calls_permitted": False,
        }
    )
    for method in gate["methods"]:
        if method in state["completed"]:
            continue
        command = [
            str(ROOT / ".venv/Scripts/python.exe"),
            str(ROOT / "scripts/run_locomo_experiment.py"),
            "--method",
            str(method),
            "--model",
            str(gate["model"]),
            "--manifest",
            str(gate["manifest"]),
            "--repetition-id",
            str(gate["repetition_id"]),
        ]
        print(f"START answer gate {method}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        state["attempts"].append(
            {"method": method, "exit_code": completed.returncode}
        )
        if completed.returncode != 0:
            state["status"] = "paused_on_failure_or_daily_quota"
            write_state(state_path, state)
            raise SystemExit(completed.returncode)
        state["completed"].append(method)
        state["status"] = "running"
        write_state(state_path, state)
    state["status"] = "complete"
    write_state(state_path, state)
    print("Matched 40-question answer gate complete.", flush=True)


def write_state(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
