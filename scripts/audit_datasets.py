from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindset.datasets.locomo import audit_locomo, load_locomo, locate_locomo
from mindset.datasets.subsets import (
    load_json_records,
    load_memoryagentbench_questions,
    memoryagentbench_manifest,
)
from mindset.utils.config import load_config, resolve_optional_path


def audit(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    locomo_config = config["locomo"]
    locomo_path = locate_locomo(locomo_config.get("path"), locomo_config.get("search_paths", []))
    locomo = audit_locomo(load_locomo(locomo_path), locomo_path)
    result: dict[str, object] = {"locomo": locomo}

    long_config = config["longmemeval"]
    long_path = resolve_optional_path(long_config.get("path"), "MINDSET_LONGMEMEVAL_PATH")
    if long_path and long_path.is_file():
        records = load_json_records(long_path)
        manifest_path = Path(long_config["manifest"])
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Frozen LongMemEval primary manifest is missing: {manifest_path}. "
                "Run scripts/freeze_primary_manifests.py first."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_size = int(long_config.get("target_size", 100))
        if int(manifest.get("selected_count", -1)) != expected_size:
            raise ValueError(
                f"Frozen LongMemEval manifest has {manifest.get('selected_count')} records; "
                f"expected {expected_size}"
            )
        result["longmemeval"] = {
            "status": "available",
            "role": "frozen_primary_100_question_subset",
            "source_path": str(long_path.resolve()),
            "source_record_count": len(records),
            "manifest": str(manifest_path),
            "selected_count": manifest["selected_count"],
        }
    else:
        result["longmemeval"] = {
            "status": "missing",
            "role": "frozen_primary_100_question_subset",
            "required_input": "Compact JSON or user-provided extracted subset",
        }

    mab_config = config["memoryagentbench"]
    mab_path = resolve_optional_path(mab_config.get("path"), "MINDSET_MEMORYAGENTBENCH_PATH")
    if mab_path and mab_path.is_file():
        records, source_groups = load_memoryagentbench_questions(mab_path)
        manifest = memoryagentbench_manifest(records)
        frozen_mab = json.loads(Path(mab_config["manifest"]).read_text(encoding="utf-8"))
        if manifest != frozen_mab:
            raise ValueError("Frozen MemoryAgentBench manifest differs from deterministic regeneration")
        result["memoryagentbench"] = {
            "status": "available",
            "role": "incremental_memory_transfer",
            "source_path": str(mab_path.resolve()),
            "source_group_count": source_groups,
            "question_count": len(records),
            "manifest": mab_config["manifest"],
            "selected_count": manifest["selected_count"],
        }
    else:
        result["memoryagentbench"] = {
            "status": "missing",
            "role": "incremental_memory_transfer",
            "preferred_task_family": "Test-Time Learning",
        }
    return result


def render_markdown(audit_result: dict[str, object]) -> str:
    locomo = audit_result["locomo"]
    assert isinstance(locomo, dict)
    lines = [
        "# Dataset audit",
        "",
        "This report is generated locally. No API calls or dataset downloads occurred.",
        "",
        "## LoCoMo",
        "",
        f"- Status: {locomo['status']}",
        f"- Source: `{locomo['source_path']}`",
        f"- Conversations: {locomo['conversation_count']}",
        f"- Sessions with turn arrays: {locomo['session_count']}",
        f"- Turns: {locomo['turn_count']}",
        f"- Questions: {locomo['question_count']}",
        f"- Questions missing an answer field: {locomo['missing_answer_count']}",
        f"- Multimodal source turns retained as text metadata: {locomo['multimodal_source_turn_count']}",
        f"- Orphan session timestamps: {locomo['orphan_session_timestamp_count']}",
        f"- Missing evidence references: {locomo['missing_evidence_reference_count']}",
        "",
        "### Category distribution",
        "",
        "| Category | Interpretation | Count |",
        "|---:|---|---:|",
    ]
    for category, detail in locomo["category_distribution"].items():
        lines.append(f"| {category} | {detail['name']} | {detail['count']} |")
    for name, title in (("longmemeval", "LongMemEval"), ("memoryagentbench", "MemoryAgentBench")):
        detail = audit_result[name]
        assert isinstance(detail, dict)
        lines.extend(["", f"## {title}", "", f"- Status: {detail['status']}"])
        if detail["status"] == "missing":
            lines.append("- No source data was found; selection was not fabricated.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--json-output", default="outputs/dataset_audit.json")
    parser.add_argument("--markdown-output", default="outputs/dataset_audit.md")
    args = parser.parse_args()
    result = audit(args.config)
    json_path = Path(args.json_output)
    markdown_path = Path(args.markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(f"Wrote {json_path} and {markdown_path}")


if __name__ == "__main__":
    main()
