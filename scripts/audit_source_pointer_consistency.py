"""Independent, zero-inference paired audit of old quotes vs new source IDs."""
import os
from mindset.acl.pipeline import ROOT, record_environment
from mindset.acl.io import read_json, unique_rows, write_json, digest_file
from mindset.evaluation.source_pair_consistency import consistency


def run():
    if os.name == "nt" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("DGX Slurm only")
    new = ROOT / "outputs/acl/source_pointers_r01"; old = ROOT / "outputs/acl/bound_attribute_contrasts_r01"
    if read_json(new / "status.json")["status"] != "complete_source_pointers":
        raise ValueError("Full source-pointer measurements not complete; no partial-panel comparison")
    if read_json(old / "status.json")["status"] != "complete_bound_contrasts":
        raise ValueError("Quote parent is not complete")
    na = read_json(new / "audit.json"); oa = read_json(old / "audit.json")
    manifest = ROOT / "data/manifests/mindset_v2_development100.json"
    if na["manifest_sha256"] != digest_file(manifest) or oa["manifest_sha256"] != digest_file(manifest):
        raise ValueError("Development scope changed")
    for name, expected in na["modules_sha256"].items():
        if digest_file(ROOT / "src/mindset/memory" / name) != expected:
            raise ValueError("Instrument module changed")
    if na["checker_sha256"] != digest_file(ROOT / "scripts/check_source_pointer_outputs.py"):
        raise ValueError("Pointer validation code changed")
    parents = unique_rows(ROOT / "outputs/acl/natural_source_relations_r01/measurements/pairs.jsonl")
    inputs = read_json(old / "inputs.json")
    expected = {c["id"]:c for c in inputs}
    old_rows = read_json(old / "complete_validation/records.json")
    new_rows = read_json(new / "complete_validation/records.json")
    for rows in (old_rows, new_rows):
        if len(rows) != 53 or {r["id"] for r in rows} != set(expected):
            raise ValueError("Paired input membership changed")
        for row in rows:
            if any(row[k] != v for k,v in expected[row["id"]].items()):
                raise ValueError("Paired source reports/labels changed")
    out = ROOT / f"outputs/acl/source_pointer_consistency_{os.environ['SLURM_JOB_ID']}"
    out.mkdir(parents=True, exist_ok=False)
    record_environment(out)
    write_json(out / "audit.json", {"worker_sha256": digest_file(__file__),
        "audit_module_sha256": digest_file(ROOT / "src/mindset/evaluation/source_pair_consistency.py"),
        "manifest_sha256": digest_file(manifest), "new_records_sha256": digest_file(new / "complete_validation/records.json"),
        "old_records_sha256": digest_file(old / "complete_validation/records.json"),
        "parent_pair_measurements_sha256": digest_file(ROOT / "outputs/acl/natural_source_relations_r01/measurements/pairs.jsonl"),
        "no_new_model_calls": True, "scope": "consistency, NOT natural accuracy or energy/QA usefulness"}, immutable=True)
    reports = {"quote_contract": consistency(old_rows, parents), "pointer_contract": consistency(new_rows, parents)}
    old_by_id = {r["id"]:r for r in old_rows}
    changes = []
    for n in new_rows:
        o = old_by_id[n["id"]]
        old_admission = None if o["error"] is not None else o["candidate"]["update_proposal"]
        new_admission = None if n["error"] is not None else n["candidate"]["update_proposal"]
        old_kind = None if o["error"] is not None else o["prediction"]["kind"]
        new_kind = None if n["error"] is not None else n["prediction"]["kind"]
        if (old_admission, old_kind, o["error"]) != (new_admission, new_kind, n["error"]):
            changes.append({"id": n["id"], "reports": n["reports"], "old": o, "new": n,
                "semantic_correctness_verified": False})
    write_json(out / "comparison.json", reports, immutable=True)
    write_json(out / "changed_interpretations.json", changes, immutable=True)
    write_json(out / "analysis.json", {"status": "complete_consistency_audit", "paired_cases": 53,
        "changes": len(changes), "arms": {arm:{k:v for k,v in report.items() if k != "groups"} for arm,report in reports.items()},
        "LLM_QA_or_hosted_calls": 0, "scope": "presentation consistency; natural truth and energy gains unverified"}, immutable=True)
    print(read_json(out / "analysis.json"), flush=True)


if __name__ == "__main__": run()
