from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindset.datasets.locomo import load_locomo
from mindset.methods.base import ConversationEvent
from mindset.methods.mindset import MindsetMemory
from mindset.methods.cache import write_json_once
from mindset.utils.hashing import stable_hash


def locomo_records():
    records = []
    samples = load_locomo(ROOT / "data/locomo/locomo10.json")
    for sample in samples:
        for index, question in enumerate(sample.questions):
            records.append(
                {
                    "id": f"{sample.sample_id}:q{index:04d}",
                    "sample_id": sample.sample_id,
                    "question_index": index,
                    "category": int(question.category),
                }
            )
    return samples, records


def stratified_disjoint(records, counts, protocol):
    by_category = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record)
    for category in by_category:
        by_category[category].sort(key=lambda row: stable_hash({"protocol": protocol, "id": row["id"]}))
    offsets = defaultdict(int)
    manifests = []
    for name, target in counts:
        base, remainder = divmod(target, len(by_category))
        selected = []
        for position, category in enumerate(sorted(by_category)):
            amount = base + int(position < remainder)
            start = offsets[category]
            chosen = by_category[category][start : start + amount]
            if len(chosen) != amount:
                raise ValueError(f"Insufficient category {category} examples for {name}")
            offsets[category] += amount
            selected.extend(chosen)
        selected.sort(key=lambda row: stable_hash({"manifest": name, "id": row["id"]}))
        manifests.append(
            (
                name,
                {
                    "manifest_version": f"{name}-v1",
                    "role": "development_gate" if name == "locomo_kill120" else "fixed_evaluation_subset",
                    "selection": "Disjoint category-stratified SHA-256 ordering; frozen before predictions.",
                    "count": len(selected),
                    "records": selected,
                },
            )
        )
    return manifests


def subset_existing(source: Path, name: str, count: int):
    manifest = json.loads(source.read_text(encoding="utf-8"))
    selected = sorted(manifest["selected"], key=lambda row: stable_hash({"manifest": name, "id": row["id"]}))[:count]
    return {
        "manifest_version": f"{name}-v1",
        "source_manifest_sha256": stable_hash(manifest),
        "selection": "SHA-256 stable subset of the previously frozen benchmark manifest.",
        "count": len(selected),
        "selected": selected,
    }


def silver_operation_manifest(samples, target=200):
    candidates = defaultdict(list)
    for sample in samples:
        method = MindsetMemory()
        events = []
        for session in sample.sessions:
            timestamp = session.date_time or session.session_id
            for turn in session.turns:
                events.append(
                    ConversationEvent(
                        event_id=f"{sample.sample_id}:{turn.dia_id}",
                        conversation_id=sample.sample_id,
                        session_id=session.session_id,
                        timestamp=timestamp,
                        speaker=turn.speaker,
                        text=turn.text,
                    )
                )
        method.ingest(events)
        for record in method.operation_log.all():
            candidates[record.operation].append(
                {
                    "id": record.episode_id,
                    "sample_id": sample.sample_id,
                    "silver_operation": record.operation,
                }
            )
    for operation, rows in candidates.items():
        rows.sort(key=lambda row: stable_hash({"silver": operation, "id": row["id"]}))
    selected = []
    labels = sorted(candidates)
    while labels and len(selected) < target:
        remaining = []
        for label in labels:
            if candidates[label] and len(selected) < target:
                selected.append(candidates[label].pop(0))
            if candidates[label]:
                remaining.append(label)
        labels = remaining
    return {
        "manifest_version": "operation-silver-energy-v1",
        "label_status": "silver_proxy_not_human_gold",
        "intended_use": "controller diagnostics and disagreement sampling only; not ground-truth operation accuracy",
        "label_provenance": {
            "source": "deterministic free-energy transition controller",
            "rule_version": "free-energy-memory-v1",
            "llm_used": False,
            "human_used": False,
        },
        "selection": "Round-robin over available deterministic operation labels with SHA-256 ordering.",
        "count": len(selected),
        "records": selected,
    }


def main():
    samples, records = locomo_records()
    output = ROOT / "data/manifests"
    for name, manifest in stratified_disjoint(
        records,
        [("locomo_kill120", 120), ("locomo_cross_model120", 120), ("locomo_repeatability100", 100)],
        "paper-protocol-v1",
    ):
        write_json_once(output / f"{name}.json", manifest)
    # The LongMemEval cross-model-60 is preserved and nested by
    # freeze_primary_manifests.py; do not reselect it after the primary scope freezes.
    write_json_once(
        output / "memoryagentbench_cross_model40.json",
        subset_existing(output / "memoryagentbench_ttl.json", "memoryagentbench_cross_model40", 40),
    )
    write_json_once(output / "operation_silver200_v2.json", silver_operation_manifest(samples))
    print("Frozen manifests written and verified immutable.")


if __name__ == "__main__":
    main()
