"""Observation-order consistency diagnostic, not semantic accuracy.

Reversed presentations of the same literal source spans should not create
different directed update claims. Unknowns and structural failures stay visible.
"""
from collections import defaultdict
from mindset.utils.hashing import stable_hash


def neutral_source_ids(case, parent_pairs):
    reports = case["reports"]
    if len(reports) != 2 or {r["source_id"] for r in reports} != {"A", "B"}:
        raise ValueError("Two distinct source labels required")
    parent = parent_pairs[case["parent_pair_key"]]
    if parent["reports"] != reports:
        raise ValueError("Parent complete prepared reports changed")
    spans = {s["source_id"]: s for s in parent["source_spans"]}
    if set(spans) != {"A", "B"}:
        raise ValueError("Parent source spans differ")
    return {r["source_id"]: stable_hash({
        "episode_id": spans[r["source_id"]]["episode_id"],
        "char_start": spans[r["source_id"]]["char_start"],
        "char_end": spans[r["source_id"]]["char_end"],
        "speaker": r["speaker"], "observation_time": r["observation_time"], "text": r["text"]}) for r in reports}


def interpretation(row, ids):
    if row["error"] is not None:
        return {"valid": False, "error": row["error"], "signature": None}
    p = row["prediction"]; kind = p["kind"]; direction = p["direction"]
    edge = None
    if kind == "attribute_change" and direction in ("a_to_b", "b_to_a"):
        left, right = ("A", "B") if direction == "a_to_b" else ("B", "A")
        edge = [ids[left], ids[right]]
    return {"valid": True, "kind": kind, "direction": direction, "neutral_directed_edge": edge,
        "signature": stable_hash({"kind": kind, "edge": edge, "unknown": direction == "unknown"}),
        "update_proposal": bool(row["candidate"]["update_proposal"]),
        "variable_literal": p["variable"], "variable_semantic_equivalence_verified": False}


def consistency(rows, parent_pairs):
    groups = defaultdict(list)
    for row in rows:
        if "parent_pair_key" not in row:
            continue  # Controlled cases are evaluated separately, not natural groups.
        ids = neutral_source_ids(row, parent_pairs)
        key = stable_hash(sorted(ids.values()))
        groups[key].append({"id": row["id"], "presented_order": [ids["A"], ids["B"]],
            "interpretation": interpretation(row, ids)})
    records = []
    for key, members in sorted(groups.items()):
        valid = [m["interpretation"] for m in members if m["interpretation"]["valid"]]
        reversed_presentations = len({tuple(m["presented_order"]) for m in members}) > 1
        records.append({"id": key, "members": members, "reversed_presentations": reversed_presentations,
            "all_structurally_valid": len(valid) == len(members),
            "relation_or_direction_disagreement": len({v["signature"] for v in valid}) > 1,
            "update_admission_disagreement": len({v["update_proposal"] for v in valid}) > 1,
            "variable_literal_disagreement": len({v["variable_literal"] for v in valid if v["kind"] == "attribute_change"}) > 1,
            "any_update_admission": any(v["update_proposal"] for v in valid),
            "semantic_correctness_verified": False})
    mirrored = [r for r in records if r["reversed_presentations"]]
    return {"natural_rows": sum(len(r["members"]) for r in records), "unordered_source_span_groups": len(records),
        "mirrored_groups": len(mirrored), "mirrored_structural_failure_groups": sum(not r["all_structurally_valid"] for r in mirrored),
        "mirrored_relation_direction_disagreement_groups": sum(r["relation_or_direction_disagreement"] for r in mirrored),
        "mirrored_update_admission_disagreement_groups": sum(r["update_admission_disagreement"] for r in mirrored),
        "update_admitted_source_groups": sum(r["any_update_admission"] for r in records),
        "scope": "source-span presentation consistency; not independent samples, natural accuracy or world updates",
        "groups": records}
