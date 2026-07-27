from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindset.datasets.locomo import load_locomo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MINDSET schema statistics and an auditable transition case study"
    )
    parser.add_argument(
        "--run-dir",
        default="outputs/runs/locomo_mindset_full_gpt_oss_120b_20260717T051244Z",
    )
    parser.add_argument("--dataset", default="data/locomo/locomo10.json")
    parser.add_argument("--case-conversation", default="conv-47")
    parser.add_argument("--output", default="outputs/analysis/mindset_structure_v1.json")
    args = parser.parse_args()

    run_dir = ROOT / args.run_dir
    snapshots = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_dir / "memory").glob("*.json"))
    ]
    if not snapshots:
        raise FileNotFoundError(f"No memory snapshots found under {run_dir / 'memory'}")

    operation_counts: Counter[str] = Counter()
    schema_count = active_count = historical_count = exception_count = 0
    support_memberships = provenance_memberships = episode_count = 0
    per_conversation: list[dict[str, Any]] = []
    for snapshot in snapshots:
        schemas = snapshot["schemas"]
        transitions = snapshot["transitions"]
        counts = Counter(row["operation"] for row in transitions)
        operation_counts.update(counts)
        schema_count += len(schemas)
        active_count += sum(row["status"] == "active" for row in schemas)
        historical_count += sum(row["status"] == "historical" for row in schemas)
        exception_count += sum(len(row["exception_episode_ids"]) for row in schemas)
        support_memberships += sum(len(row["support_episode_ids"]) for row in schemas)
        provenance_memberships += sum(
            len(row["support_episode_ids"]) + len(row["exception_episode_ids"])
            for row in schemas
        )
        episode_count += len(snapshot["episodes"])
        per_conversation.append(
            {
                "conversation_id": snapshot["episodes"][0]["conversation_id"],
                "episode_count": len(snapshot["episodes"]),
                "schema_count": len(schemas),
                "active_schema_count": sum(row["status"] == "active" for row in schemas),
                "historical_schema_count": sum(
                    row["status"] == "historical" for row in schemas
                ),
                "operation_counts": dict(sorted(counts.items())),
            }
        )

    total_transitions = sum(operation_counts.values())
    aggregate = {
        "conversation_count": len(snapshots),
        "episode_count": episode_count,
        "schema_count": schema_count,
        "active_schema_count": active_count,
        "historical_schema_count": historical_count,
        "support_membership_count": support_memberships,
        "exception_membership_count": exception_count,
        "average_support_episodes_per_schema": support_memberships / schema_count,
        "average_provenance_episodes_per_schema": provenance_memberships / schema_count,
        "operation_counts": dict(sorted(operation_counts.items())),
        "operation_frequencies": {
            name: count / total_transitions
            for name, count in sorted(operation_counts.items())
        },
    }

    sample = {
        row.sample_id: row for row in load_locomo(ROOT / args.dataset)
    }[args.case_conversation]
    text_by_id = {
        turn.dia_id: {"speaker": turn.speaker, "text": turn.text}
        for session in sample.sessions
        for turn in session.turns
    }
    case_snapshot = next(
        row
        for row in snapshots
        if row["episodes"][0]["conversation_id"] == args.case_conversation
    )
    # These four records are fixed, auditable examples from one frozen conversation.
    chosen_ids = {
        "reinforce": "D3:16",
        "supersede": "D3:18",
        "split": "D18:1",
        "new": "D14:2",
    }
    transitions = {
        row["episode_id"].split(":", 1)[1]: row
        for row in case_snapshot["transitions"]
    }
    case_rows = []
    for expected_operation, dia_id in chosen_ids.items():
        row = transitions[dia_id]
        if row["operation"] != expected_operation:
            raise AssertionError(
                f"Expected {dia_id} to be {expected_operation}, got {row['operation']}"
            )
        case_rows.append(
            {
                "operation": expected_operation,
                "dia_id": dia_id,
                **text_by_id[dia_id],
                "target_schema_id": row["target_schema_id"],
                "resulting_schema_ids": row["resulting_schema_ids"],
                "signals": row["signals"],
                "candidate_energies": row["candidate_energies"],
                "selected_energy_terms": row["selected_energy_terms"],
                "hysteresis_applied": row["hysteresis_applied"],
                "stored_as_exception": row["stored_as_exception"],
            }
        )

    payload = {
        "source_run": str(run_dir.relative_to(ROOT)),
        "dataset": args.dataset,
        "aggregate": aggregate,
        "per_conversation": per_conversation,
        "transition_case_study": {
            "conversation_id": args.case_conversation,
            "records": case_rows,
        },
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
