from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindset.datasets.locomo import load_locomo
from mindset.datasets.subsets import load_json_records, longmemeval_stratum, stable_record_id
from mindset.utils.hashing import stable_hash


LOCOMO_SOURCE = ROOT / "data/locomo/locomo10.json"
LONGMEMEVAL_SOURCE = ROOT / "data/longmemeval/longmemeval_s_cleaned.json"
MANIFEST_DIR = ROOT / "data/manifests"
LOCOMO_PRIMARY = MANIFEST_DIR / "locomo_primary700.json"
LONGMEMEVAL_PRIMARY = MANIFEST_DIR / "longmemeval_primary100.json"
SELECTION_AUDIT = ROOT / "outputs/selection_audit/primary1500.json"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def balanced_targets(capacities: dict[Any, int], target: int) -> dict[Any, int]:
    """Water-fill integer quotas without exceeding any stratum capacity."""
    if target > sum(capacities.values()):
        raise ValueError("Target exceeds available records")
    quotas = {key: 0 for key in capacities}
    while sum(quotas.values()) < target:
        eligible = [key for key in capacities if quotas[key] < capacities[key]]
        key = min(eligible, key=lambda value: (quotas[value], str(value)))
        quotas[key] += 1
    return quotas


def max_flow_allocations(
    row_needs: dict[str, int],
    column_needs: dict[int, int],
    capacities: dict[tuple[str, int], int],
) -> dict[tuple[str, int], int]:
    """Deterministic Edmonds-Karp flow for conversation/category cell quotas."""
    source, sink = "__source__", "__sink__"
    adjacency: dict[Any, list[Any]] = defaultdict(list)
    residual: dict[tuple[Any, Any], int] = {}

    def add_edge(left: Any, right: Any, capacity: int) -> None:
        adjacency[left].append(right)
        adjacency[right].append(left)
        residual[left, right] = capacity
        residual[right, left] = 0

    for row in sorted(row_needs):
        add_edge(source, ("row", row), row_needs[row])
    for row in sorted(row_needs):
        for column in sorted(column_needs):
            add_edge(("row", row), ("column", column), capacities.get((row, column), 0))
    for column in sorted(column_needs):
        add_edge(("column", column), sink, column_needs[column])

    flow = 0
    while True:
        parents: dict[Any, Any | None] = {source: None}
        queue = deque([source])
        while queue and sink not in parents:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in parents and residual[node, neighbor] > 0:
                    parents[neighbor] = node
                    queue.append(neighbor)
        if sink not in parents:
            break
        amount = 10**9
        node = sink
        while parents[node] is not None:
            parent = parents[node]
            amount = min(amount, residual[parent, node])
            node = parent
        node = sink
        while parents[node] is not None:
            parent = parents[node]
            residual[parent, node] -= amount
            residual[node, parent] += amount
            node = parent
        flow += amount

    required = sum(row_needs.values())
    if flow != required or flow != sum(column_needs.values()):
        raise ValueError(f"Could not satisfy primary LoCoMo quotas: flow={flow}, required={required}")
    return {
        (row, column): capacities.get((row, column), 0)
        - residual[("row", row), ("column", column)]
        for row in row_needs
        for column in column_needs
    }


def freeze_locomo() -> dict[str, Any]:
    samples = load_locomo(LOCOMO_SOURCE)
    records: list[dict[str, Any]] = []
    for sample in samples:
        for index, question in enumerate(sample.questions):
            records.append(
                {
                    "id": f"{sample.sample_id}:q{index:04d}",
                    "sample_id": sample.sample_id,
                    "question_index": index,
                    "category": int(question.category),
                    "question_sha256": text_sha256(question.question),
                }
            )
    by_id = {row["id"]: row for row in records}
    prerequisite_paths = [
        MANIFEST_DIR / "locomo_kill120.json",
        MANIFEST_DIR / "locomo_cross_model120.json",
        MANIFEST_DIR / "locomo_repeatability100.json",
    ]
    required_ids: set[str] = set()
    prerequisite_hashes = {}
    for path in prerequisite_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        ids = {str(row["id"]) for row in payload["records"]}
        if required_ids & ids:
            raise ValueError(f"Frozen LoCoMo prerequisites overlap at {path.name}")
        required_ids.update(ids)
        prerequisite_hashes[path.name] = stable_hash(payload)
    if len(required_ids) != 340 or not required_ids <= set(by_id):
        raise ValueError("Expected 340 valid, disjoint frozen LoCoMo prerequisite questions")

    source_category_counts = Counter(row["category"] for row in records)
    category_targets = balanced_targets(dict(source_category_counts), 700)
    conversation_targets = balanced_targets(
        dict(Counter(row["sample_id"] for row in records)), 700
    )
    required_category = Counter(by_id[item]["category"] for item in required_ids)
    required_conversation = Counter(by_id[item]["sample_id"] for item in required_ids)
    category_needs = {
        category: category_targets[category] - required_category[category]
        for category in category_targets
    }
    conversation_needs = {
        sample_id: conversation_targets[sample_id] - required_conversation[sample_id]
        for sample_id in conversation_targets
    }
    if min(category_needs.values()) < 0 or min(conversation_needs.values()) < 0:
        raise ValueError("Frozen prerequisites exceed a primary quota")

    available: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["id"] not in required_ids:
            available[row["sample_id"], row["category"]].append(row)
    for cell, rows in available.items():
        rows.sort(key=lambda row: stable_hash({"protocol": "primary1500-v1", "id": row["id"]}))
    allocations = max_flow_allocations(
        conversation_needs,
        category_needs,
        {cell: len(rows) for cell, rows in available.items()},
    )
    selected_ids = set(required_ids)
    for cell, amount in allocations.items():
        selected_ids.update(row["id"] for row in available[cell][:amount])
    selected = sorted((by_id[item] for item in selected_ids), key=lambda row: row["id"])
    if len(selected) != 700:
        raise ValueError(f"LoCoMo primary manifest has {len(selected)} records, expected 700")

    manifest = {
        "manifest_version": "locomo-primary700-v1",
        "dataset": "locomo",
        "role": "primary_paper_evaluation",
        "source_path": "data/locomo/locomo10.json",
        "source_file_sha256": file_sha256(LOCOMO_SOURCE),
        "selection_outcome_blind": True,
        "selection": (
            "Include all 340 previously frozen kill/cross-model/repeatability questions; "
            "water-fill category and conversation quotas; deterministic maximum-flow cell "
            "allocation; SHA-256 ordering within cells; no RNG, answers, evidence, predictions, "
            "or metrics used."
        ),
        "prerequisite_manifest_sha256": prerequisite_hashes,
        "target_size": 700,
        "count": len(selected),
        "category_distribution": dict(sorted(Counter(row["category"] for row in selected).items())),
        "conversation_distribution": dict(
            sorted(Counter(row["sample_id"] for row in selected).items())
        ),
        "records": selected,
    }
    write_json(LOCOMO_PRIMARY, manifest)
    return manifest


def freeze_longmemeval() -> dict[str, Any]:
    records = load_json_records(LONGMEMEVAL_SOURCE)
    indexed = {stable_record_id(record, index): record for index, record in enumerate(records)}
    eligible = {
        identifier: record
        for identifier, record in indexed.items()
        if longmemeval_stratum(record)
        in {"abstention", "knowledge_update", "multi_session", "temporal"}
    }
    cross_path = MANIFEST_DIR / "longmemeval_cross_model60.json"
    cross = json.loads(cross_path.read_text(encoding="utf-8"))
    required_ids = {str(row["id"]) for row in cross["selected"]}
    if len(required_ids) != 60 or not required_ids <= set(eligible):
        raise ValueError("LongMemEval cross-model-60 is not a valid prerequisite")

    capacities = Counter(longmemeval_stratum(record) for record in eligible.values())
    targets = balanced_targets(dict(capacities), 100)
    selected_ids = set(required_ids)
    current = Counter(longmemeval_stratum(eligible[item]) for item in selected_ids)
    for stratum in sorted(targets):
        need = targets[stratum] - current[stratum]
        if need < 0:
            raise ValueError(f"Cross-model prerequisite exceeds LongMemEval quota for {stratum}")
        candidates = [
            identifier
            for identifier, record in eligible.items()
            if identifier not in selected_ids and longmemeval_stratum(record) == stratum
        ]
        candidates.sort(
            key=lambda identifier: stable_hash(
                {"protocol": "primary1500-v1", "dataset": "longmemeval", "id": identifier}
            )
        )
        selected_ids.update(candidates[:need])
    if len(selected_ids) != 100:
        raise ValueError(f"LongMemEval primary manifest has {len(selected_ids)} records, expected 100")
    selected = []
    for identifier in sorted(selected_ids):
        record = eligible[identifier]
        selected.append(
            {
                "id": identifier,
                "stratum": longmemeval_stratum(record),
                "question_sha256": text_sha256(str(record.get("question", ""))),
            }
        )
    manifest = {
        "manifest_version": "longmemeval-primary100-v1",
        "dataset": "longmemeval",
        "role": "primary_paper_evaluation",
        "source_path": "data/longmemeval/longmemeval_s_cleaned.json",
        "source_file_sha256": file_sha256(LONGMEMEVAL_SOURCE),
        "selection_outcome_blind": True,
        "selection": (
            "Include the frozen cross-model-60; fill to 25 questions in each preregistered "
            "diagnostic stratum by SHA-256 ordering; no RNG, answers, retrieved evidence, "
            "predictions, or metrics used."
        ),
        "prerequisite_selection_sha256": stable_hash(cross["selected"]),
        "target_size": 100,
        "selected_count": len(selected),
        "stratum_distribution": dict(sorted(Counter(row["stratum"] for row in selected).items())),
        "selected": selected,
    }
    write_json(LONGMEMEVAL_PRIMARY, manifest)
    cross["source_manifest_sha256"] = stable_hash(manifest)
    cross["selection"] = (
        "Previously frozen cross-model questions, verified as a nested subset of "
        "longmemeval_primary100.json."
    )
    write_json(cross_path, cross)
    return manifest


def main() -> None:
    locomo = freeze_locomo()
    longmemeval = freeze_longmemeval()
    mab_path = MANIFEST_DIR / "memoryagentbench_ttl.json"
    mab = json.loads(mab_path.read_text(encoding="utf-8"))
    if int(mab.get("selected_count", -1)) != 700:
        raise ValueError("MemoryAgentBench primary manifest must contain exactly 700 questions")
    locomo_ids = {row["id"] for row in locomo["records"]}
    long_ids = {row["id"] for row in longmemeval["selected"]}
    mab_ids = {row["id"] for row in mab["selected"]}
    auxiliary_sets = {
        "locomo_kill120": {
            row["id"]
            for row in json.loads((MANIFEST_DIR / "locomo_kill120.json").read_text(encoding="utf-8"))["records"]
        },
        "locomo_cross_model120": {
            row["id"]
            for row in json.loads((MANIFEST_DIR / "locomo_cross_model120.json").read_text(encoding="utf-8"))["records"]
        },
        "locomo_repeatability100": {
            row["id"]
            for row in json.loads((MANIFEST_DIR / "locomo_repeatability100.json").read_text(encoding="utf-8"))["records"]
        },
        "longmemeval_cross_model60": {
            row["id"]
            for row in json.loads((MANIFEST_DIR / "longmemeval_cross_model60.json").read_text(encoding="utf-8"))["selected"]
        },
        "memoryagentbench_cross_model40": {
            row["id"]
            for row in json.loads((MANIFEST_DIR / "memoryagentbench_cross_model40.json").read_text(encoding="utf-8"))["selected"]
        },
    }
    nesting = {
        name: identifiers <= (
            long_ids if name.startswith("longmemeval") else mab_ids if name.startswith("memoryagentbench") else locomo_ids
        )
        for name, identifiers in auxiliary_sets.items()
    }
    if not all(nesting.values()):
        raise ValueError(f"An auxiliary manifest escapes the primary scope: {nesting}")
    audit = {
        "audit_version": "primary1500-v2-six-method",
        "frozen_before_primary_answer_generation": True,
        "selection_outcome_blind": True,
        "selection_inputs": "dataset identity, stable question ID, LoCoMo category/conversation, and LongMemEval question type only",
        "total_unique_questions": len(locomo_ids) + len(long_ids) + len(mab_ids),
        "total_method_question_instances": 6 * (len(locomo_ids) + len(long_ids) + len(mab_ids)),
        "datasets": {
            "locomo": {
                "count": len(locomo_ids),
                "manifest": "data/manifests/locomo_primary700.json",
                "manifest_sha256": stable_hash(locomo),
                "category_distribution": locomo["category_distribution"],
                "conversation_distribution": locomo["conversation_distribution"],
            },
            "longmemeval": {
                "count": len(long_ids),
                "manifest": "data/manifests/longmemeval_primary100.json",
                "manifest_sha256": stable_hash(longmemeval),
                "stratum_distribution": longmemeval["stratum_distribution"],
            },
            "memoryagentbench": {
                "count": len(mab_ids),
                "manifest": "data/manifests/memoryagentbench_ttl.json",
                "manifest_sha256": stable_hash(mab),
            },
        },
        "auxiliary_subset_nesting": nesting,
        "reproduction_command": ".\\.venv\\Scripts\\python.exe scripts\\freeze_primary_manifests.py",
    }
    if audit["total_unique_questions"] != 1500:
        raise ValueError(f"Primary scope is {audit['total_unique_questions']}, expected 1500")
    write_json(SELECTION_AUDIT, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
