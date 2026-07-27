from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the zero-API MINDSET-energy retrieval kill test"
    )
    parser.add_argument("--config", default="configs/kill_test.yaml")
    parser.add_argument("--state", default="outputs/orchestration/six_method_kill120_v2.json")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    state_path = ROOT / args.state
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {
            "protocol": config["protocol"],
            "repetition_id": config["repetition_id"],
            "completed": [],
            "attempts": [],
            "network_calls_permitted": False,
        }
    )
    for method in config["methods"]:
        if method in state["completed"]:
            continue
        command = [
            str(ROOT / ".venv/Scripts/python.exe"),
            str(ROOT / "scripts/run_locomo_experiment.py"),
            "--method", method,
            "--model", config["model"],
            "--manifest", config["manifest"],
            "--repetition-id", config["repetition_id"],
            "--retrieval-only",
        ]
        print(f"START local retrieval {method}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        state["attempts"].append({"method": method, "exit_code": completed.returncode})
        if completed.returncode != 0:
            state["status"] = "paused_on_failure"
            write_state(state_path, state)
            raise SystemExit(completed.returncode)
        state["completed"].append(method)
        state["status"] = "running"
        write_state(state_path, state)
    state["status"] = "complete"
    write_state(state_path, state)
    print("Local retrieval kill test complete; no answer-model calls were made.")


def write_state(path: Path, state: dict[str, object]) -> None:
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
