from __future__ import annotations

import json
from pathlib import Path

from mindset.datasets.locomo import audit_locomo, load_locomo
from mindset.datasets.subsets import focused_longmemeval_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_locomo_full_text_dataset_audit() -> None:
    path = ROOT / "data" / "locomo" / "locomo10.json"
    report = audit_locomo(load_locomo(path), path)
    assert report["conversation_count"] == 10
    assert report["question_count"] == 1986
    assert sum(item["count"] for item in report["category_distribution"].values()) == 1986
    assert report["session_count"] > 0


def test_focused_manifest_is_deterministic_and_preserves_ids() -> None:
    records = [
        {"question_id": f"q-{index}", "question_type": category}
        for index, category in enumerate(
            ["knowledge update", "temporal", "multi-session", "abstention"] * 10
        )
    ]
    first = focused_longmemeval_manifest(records, target_size=20)
    second = focused_longmemeval_manifest(list(reversed(records)), target_size=20)
    assert first == second
    assert first["selected_count"] == 20
    assert all(item["id"].startswith("q-") for item in first["selected"])


def test_frozen_primary_scope_is_exact_and_auxiliary_manifests_are_nested() -> None:
    manifest_dir = ROOT / "data" / "manifests"
    locomo = json.loads((manifest_dir / "locomo_primary700.json").read_text(encoding="utf-8"))
    longmemeval = json.loads(
        (manifest_dir / "longmemeval_primary100.json").read_text(encoding="utf-8")
    )
    mab = json.loads((manifest_dir / "memoryagentbench_ttl.json").read_text(encoding="utf-8"))
    audit = json.loads(
        (ROOT / "outputs" / "selection_audit" / "primary1500.json").read_text(encoding="utf-8")
    )

    locomo_ids = {row["id"] for row in locomo["records"]}
    long_ids = {row["id"] for row in longmemeval["selected"]}
    mab_ids = {row["id"] for row in mab["selected"]}
    assert (len(locomo_ids), len(long_ids), len(mab_ids)) == (700, 100, 700)
    assert audit["total_unique_questions"] == 1500
    assert audit["total_method_question_instances"] == 9000
    assert locomo["category_distribution"] == {"1": 151, "2": 151, "3": 96, "4": 151, "5": 151}
    assert set(locomo["conversation_distribution"].values()) == {70}
    assert set(longmemeval["stratum_distribution"].values()) == {25}
    assert all(len(row["question_sha256"]) == 64 for row in locomo["records"])
    assert all(len(row["question_sha256"]) == 64 for row in longmemeval["selected"])

    for name in ("locomo_kill120", "locomo_cross_model120", "locomo_repeatability100"):
        nested = json.loads((manifest_dir / f"{name}.json").read_text(encoding="utf-8"))
        assert {row["id"] for row in nested["records"]} <= locomo_ids
    long_cross = json.loads(
        (manifest_dir / "longmemeval_cross_model60.json").read_text(encoding="utf-8")
    )
    mab_cross = json.loads(
        (manifest_dir / "memoryagentbench_cross_model40.json").read_text(encoding="utf-8")
    )
    assert {row["id"] for row in long_cross["selected"]} <= long_ids
    assert {row["id"] for row in mab_cross["selected"]} <= mab_ids
