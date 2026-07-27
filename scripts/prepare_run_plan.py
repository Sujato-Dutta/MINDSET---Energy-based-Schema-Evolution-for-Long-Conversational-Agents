from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindset.experiments.protocol import freeze_run_plan


def main():
    parser = argparse.ArgumentParser(description="Freeze an immutable experiment task plan")
    parser.add_argument("config", choices=("configs/kill_test.yaml", "configs/main.yaml"))
    parser.add_argument("--output")
    args = parser.parse_args()
    name = Path(args.config).stem
    output = args.output or f"outputs/plans/{name}_v1.json"
    path = freeze_run_plan(ROOT / args.config, ROOT / output)
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps({"path": str(path), "task_count": payload["task_count"]}, indent=2))


if __name__ == "__main__":
    main()
