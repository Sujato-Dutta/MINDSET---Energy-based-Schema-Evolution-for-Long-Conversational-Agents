"""Independent replay/certificate validation and tie-aware influence. DGX only.

No reingestion, inference, fitting, answers or historical artifact overwrites.
Counterfactual ties are separated from strictly preferred changed decisions.
"""
import os
import json
import hashlib
from dataclasses import asdict
from pathlib import Path
from collections import Counter
from mindset.acl.pipeline import ROOT, write_csv
from mindset.acl.joint_iteration import protocol, groups_for
from mindset.acl.io import read_json, digest_file, write_json, unique_rows
from mindset.evaluation.energy_identifiability import attribute, contributions, choose, signature, TERMS
from mindset.utils.hashing import stable_hash

TOLERANCE = 1e-8


def margins(certificate, term):
    if term not in TERMS:
        raise ValueError("Only single-term removals have comparable numerical margins")
    plans = certificate["feasible_candidates"]
    original = choose(plans); alternative = choose(plans, (term,))
    def cost(p):
        return float(p["adjusted_energy"]) - contributions(p)[term]
    best = cost(alternative)
    result = {"selected_decision_cost_advantage": cost(original)-best,
              "decision_changed": signature(original) != signature(alternative)}
    for field in ("action", "mutation", "schema_id"):
        retained = min(cost(p) for p in plans if p[field] == original[field])
        result[field + "_advantage"] = retained - best
        result[field + "_changed"] = alternative[field] != original[field]
    return result


def run(parent):
    if os.name == "nt" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("DGX Slurm only")
    parent = Path(parent); audit = read_json(parent / "audit.json")
    if read_json(parent / "status.json")["status"] != "complete_mechanism_audit":
        raise ValueError("Parent audit is not terminal complete")
    manifest = ROOT / "data/manifests/mindset_v2_development100.json"
    calibration = ROOT / "outputs/acl/mindset_joint_v2_27058/calibration/calibration.json"
    cfg, cohort, frozen = protocol(manifest, calibration)
    for key, value in {"manifest_sha256": digest_file(manifest), "calibration_sha256": digest_file(calibration),
        "worker_sha256": digest_file(ROOT / "scripts/audit_energy_identifiability.py"),
        "checkpoint_lock_sha256": digest_file(ROOT / "configs/checkpoints.lock.json"),
        "parameters": frozen["selected_parameters"]}.items():
        if audit[key] != value:
            raise ValueError("Frozen identity mismatch: " + key)
    for source, digest in audit["modules"].items():
        if digest_file(ROOT / "src/mindset" / source) != digest:
            raise ValueError("Sealed module changed")
    histories = unique_rows(parent / "histories.jsonl"); expected = set(); summary = Counter(); detailed = []
    for dataset, cluster, queries in groups_for(cohort, cfg):
        key = f"{dataset}/{cluster}"; expected.add(key); history = histories[key]
        if history["question_ids"] != [q.id for q in queries]:
            raise ValueError("Original100 question membership differs")
        source_hash = hashlib.sha256(json.dumps([asdict(e) for e in queries[0].events],
            sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if history["events_sha256"] != source_hash:
            raise ValueError("Underlying history differs")
        rows = read_json(parent / "decisions" / f"{dataset}-{cluster}.json")
        if len(rows) != history["episodes"] or [r["episode_id"] for r in rows] != [e.event_id for e in queries[0].events]:
            raise ValueError("Missing/duplicate/reordered decisions")
        for row in rows:
            summary[(dataset, "episodes")] += 1
            if row["certificate"] is None:
                if row["operation"] != "new" or row["attribution"] is not None:
                    raise ValueError("Missing non-initial certificate")
                summary[(dataset, "initial_creations")] += 1
                continue
            derived = attribute(row["certificate"])
            if stable_hash(derived) != stable_hash(row["attribution"]):
                raise ValueError("Stored attribution mismatch")
            if derived["selected"][1] != row["operation"]:
                raise ValueError("Stored operation differs from certificate")
            summary[(dataset, "certificates_verified")] += 1
            for term in TERMS:
                effect = margins(row["certificate"], term)
                record = {"dataset": dataset, "cluster": cluster, "episode_id": row["episode_id"],
                          "removed_term": term, **effect}
                detailed.append(record)
                if effect["decision_changed"]:
                    summary[(dataset, term, "raw_decision_changes")] += 1
                    label = "strict_decision_changes" if effect["selected_decision_cost_advantage"] > TOLERANCE else "tie_only_decision_changes"
                    summary[(dataset, term, label)] += 1
                for field in ("action", "mutation", "schema_id"):
                    if effect[field + "_changed"] and effect[field + "_advantage"] > TOLERANCE:
                        summary[(dataset, term, "strict_" + field + "_changes")] += 1
    if set(histories) != expected:
        raise ValueError("Unexpected history")
    reports = []
    for dataset in ("locomo", "longmemeval"):
        for term in TERMS:
            reports.append({"dataset": dataset, "removed_term": term,
                "episodes": summary[(dataset, "episodes")],
                "initial_creations": summary[(dataset, "initial_creations")],
                "certificates_verified": summary[(dataset, "certificates_verified")],
                **{k: summary[(dataset, term, k)] for k in ("raw_decision_changes", "strict_decision_changes",
                   "tie_only_decision_changes", "strict_action_changes", "strict_mutation_changes", "strict_schema_id_changes")}})
    output = ROOT / f"outputs/acl/energy_identifiability_verified_{os.environ['SLURM_JOB_ID']}"
    write_json(output / "audit.json", {"parent_audit_sha256": digest_file(parent / "audit.json"),
        "parent_analysis_sha256": digest_file(parent / "analysis.json"),
        "parent_histories_sha256": digest_file(parent / "histories.jsonl"), "parent": str(parent),
        "verifier_sha256": digest_file(__file__), "tolerance": TOLERANCE,
        "energy_off_control": "all costs tied by construction, not a strong rule or benefit test"}, immutable=True)
    write_csv(output / "component_influence.csv", reports)
    write_json(output / "margins.json", detailed, immutable=True)
    write_json(output / "analysis.json", {"reports": reports, "histories_verified": len(expected),
        "useful_components_or_QA_or_novelty_proven": False,
        "scope": "independent identity/certificate validation and numerical necessity only"}, immutable=True)
    write_json(output / "status.json", {"status": "complete_verified_mechanism_audit"})
    print(reports, flush=True)


if __name__ == "__main__":
    run(ROOT / "outputs/acl/energy_identifiability_27673")
