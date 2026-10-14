"""Full-trajectory original100 equivalence, not performance tuning. DGX only.

Compare actual states and exact reranker inputs against unmodified Joint. If
inputs are identical, an identical pinned reader has no new evidence to answer.
No reranker/LLM calls are needed to establish that input equivalence.
"""
import os
import json
import hashlib
from dataclasses import asdict
from pathlib import Path
import numpy as np
from mindset.acl.pipeline import ROOT, record_environment
from mindset.acl.joint_iteration import protocol, groups_for
from mindset.acl.io import read_json, write_json, unique_rows, append_row, digest_file
from mindset.acl.runtime import CachedEncoder
from mindset.methods.joint import JointMindsetMemory
from mindset.methods.joint_one_price import OnePriceJointMemory
from mindset.methods.compact import CompactConfig


def compare_state(a, b):
    if a.operation_log.all() != b.operation_log.all():
        raise ValueError("Full trajectory operations changed")
    if set(a.schemas) != set(b.schemas):
        raise ValueError("Schema IDs changed")
    for sid in a.schemas:
        if a.schemas[sid].snapshot() != b.schemas[sid].snapshot() or not np.array_equal(a.schemas[sid].centroid, b.schemas[sid].centroid):
            raise ValueError("Schema post-state changed")
        if not np.array_equal(a._sums[sid], b._sums[sid]):
            raise ValueError("Support sum changed")
    if a._pending != b._pending or a._latest != b._latest or a._split_links != b._split_links:
        raise ValueError("Pending/time/branch state changed")
    if a._episode_order != b._episode_order or a._documents != b._documents:
        raise ValueError("Immutable evidence/index changed")
    if set(a._decision_certificates) != set(b._decision_certificates):
        raise ValueError("Certificate coverage changed")
    for eid, cert in a._decision_certificates.items():
        other = b._decision_certificates[eid]
        for field in ("selected_schema_id", "selected_mutation"):
            if cert[field] != other[field]:
                raise ValueError("Intermediate target/mutation changed")
    if a.validate_invariants() != b.validate_invariants():
        raise ValueError("Invariant summaries changed")


def run():
    if os.name == "nt" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("DGX Slurm only")
    manifest = ROOT / "data/manifests/mindset_v2_development100.json"
    calibration = ROOT / "outputs/acl/mindset_joint_v2_27058/calibration/calibration.json"
    cfg, cohort, frozen = protocol(manifest, calibration)
    out = ROOT / f"outputs/acl/one_price_joint_{os.environ['SLURM_JOB_ID']}"; out.mkdir(parents=True, exist_ok=True)
    lock = read_json(ROOT / "configs/checkpoints.lock.json")["encoder"]
    sources = ["methods/joint.py", "methods/joint_one_price.py", "methods/compact.py", "methods/mindset.py", "memory/energy.py"]
    write_json(out / "audit.json", {"manifest_sha256": digest_file(manifest), "calibration_sha256": digest_file(calibration),
        "worker_sha256": digest_file(__file__), "modules": {p: digest_file(ROOT / "src/mindset" / p) for p in sources},
        "parameters": frozen["selected_parameters"], "candidate_change": "only switch_margin forced to zero",
        "calibration_not_refitted": True, "encoder": lock, "query_panel": "original100 unchanged",
        "scope": "full-state/reranker-input equivalence on exposed development, not universal equivalence or novelty",
        "new_answer_calls": 0, "full_runs_authorized": False}, immutable=True)
    record_environment(out)
    from sentence_transformers import SentenceTransformer
    encoder = CachedEncoder(SentenceTransformer(lock["checkpoint"], revision=lock["revision"], device=cfg["retrieval"]["device"]), enabled=True)
    complete = unique_rows(out / "histories.jsonl")
    expected_histories = set(); expected_queries = set()
    for dataset, cluster, cases in groups_for(cohort, cfg):
        key = f"{dataset}/{cluster}"; expected_histories.add(key)
        expected_queries.update(f"{dataset}/{case.id}" for case in cases)
        if key in complete:
            continue
        config = CompactConfig(**{**{k:v for k,v in cfg["mindset"].items() if k != "profile"}, **frozen["selected_parameters"]})
        original = JointMindsetMemory(config, encoder=encoder, local_reranker=None)
        reduced = OnePriceJointMemory(config, encoder=encoder, local_reranker=None)
        original.ingest(cases[0].events); reduced.ingest(cases[0].events)
        compare_state(original, reduced)
        query_rows = []
        for case in cases:
            a, am = original.candidate_pool(case.question); b, bm = reduced.candidate_pool(case.question)
            if a != b or am != bm:
                raise ValueError("Candidate membership/order/provenance changed")
            ce_a = [(case.question, original._documents[i]) for i in a]
            ce_b = [(case.question, reduced._documents[i]) for i in b]
            if ce_a != ce_b:
                raise ValueError("Exact CE input pairs changed")
            query_rows.append({"id": f"{dataset}/{case.id}", "dataset": dataset, "cluster": cluster,
                "candidate_ids": [original._episode_order[i] for i in a], "exact_CE_inputs_identical": True,
                "CE_input_sha256": hashlib.sha256(json.dumps(ce_a, ensure_ascii=False).encode()).hexdigest()})
        # One immutable per-history record prevents duplicate query rows on resume.
        write_json(out / "queries" / f"{dataset}-{cluster}.json", query_rows, immutable=True)
        row = {"id": key, "dataset": dataset, "cluster": cluster, "episodes": len(cases[0].events),
            "queries": len(cases), "full_trajectory_state_identical": True, "exact_CE_inputs_identical": True,
            "operation_counts": original.telemetry()["operation_counts"]}
        append_row(out / "histories.jsonl", row); complete[key] = row
        encoder.cache.clear()
        print(f"ONE_PRICE {len(complete)} histories {key}: identical state and {len(cases)} query inputs", flush=True)
    if set(complete) != expected_histories:
        raise ValueError("History panel differs")
    queries = [r for path in sorted((out / "queries").glob("*.json")) for r in read_json(path)]
    if len(queries) != 100 or {r["id"] for r in queries} != expected_queries:
        raise ValueError("Original100 query coverage differs")
    write_json(out / "analysis.json", {"histories": len(complete), "episodes": sum(r["episodes"] for r in complete.values()),
        "queries": len(queries), "full_trajectory_state_identical": True, "exact_CE_inputs_identical": True,
        "new_CE_calls": 0, "new_answer_calls": 0,
        "scope": "cohort-only state/CE-input equivalence; not a new QA score or proof all components useful",
        "native_answer_scores_not_recomputed": True}, immutable=True)
    write_json(out / "status.json", {"status": "complete_equivalence_check", "questions": 100, "histories": len(complete)})


if __name__ == "__main__":
    run()
