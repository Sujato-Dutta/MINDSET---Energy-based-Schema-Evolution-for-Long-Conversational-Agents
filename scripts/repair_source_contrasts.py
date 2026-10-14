"""Post-hoc source-binding candidate; pinned syntax env, DGX Slurm only.

Saved outputs/failed gate remain unchanged. No inference, labels as inputs to
repair, new questions, memory updates, energy decisions or answer calls.
"""
import os
import sys
import importlib.util
from pathlib import Path
from time import perf_counter
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mindset.acl.io import read_json, write_json, digest_file, append_row, unique_rows
from mindset.utils.hashing import stable_hash
spec = importlib.util.spec_from_file_location("source_binding", ROOT / "src/mindset/memory/source_binding.py")
binding_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(binding_module)


def apply_candidate(payload, reports, nlp):
    binding = binding_module.normalize_binding(payload, reports)
    docs = {r["source_id"]: nlp(r["text"]) for r in reports}
    uncertainty = binding_module.reference_uncertainty(binding, reports, docs)
    return {"binding": binding, "reference_check": uncertainty,
        "update_proposal": uncertainty["update_proposal"],
        "semantic_correctness_verified": False, "world_truth_or_update_verified": False}


def functional_checks(nlp):
    checks = [
        ("I lead exactly three analysts.", "I now lead exactly five analysts.", True),
        ("My only regular commute was by train.", "I replaced that commute; I now only cycle to work.", True),
        ("I prefer that option.", "I prefer the other one now.", False),
        ("It was my usual choice.", "Now the other one is my usual choice.", False),
        ("Someone had three of those.", "They now have four of those.", False),
        ("My weekday commute is by train. My weekend commute is by bus.", "That commute is now by bicycle.", False),
        ("I lead three workers.", "They lead five workers.", False),
        ("My regular commute is by train.", "Her commute is now by bicycle.", False),
    ]
    results = []
    for i, (a, b, expected) in enumerate(checks):
        reports = [{"source_id": "A", "speaker": "user", "observation_time": "2025-01-01", "text": a},
                   {"source_id": "B", "speaker": "user", "observation_time": "2025-06-01", "text": b}]
        # Forced positive proposal probes the guard independently of LLM accuracy.
        payload = {"kind": "attribute_change", "direction": "a_to_b", "variable": "controlled_variable",
                   "a_quote": a, "b_quote": b}
        result = apply_candidate(payload, reports, nlp)
        if result["update_proposal"] != expected:
            raise ValueError(f"Reference-functional check failed: {i}: {result}")
        results.append({"id": str(i), "reports": reports, "expected_proposal": expected, "candidate": result})
    return results


def run():
    if os.name == "nt" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("DGX Slurm only")
    parent = ROOT / "outputs/acl/attribute_contrasts_r01"
    if read_json(parent / "status.json")["status"] != "controls_failed_natural_not_run":
        raise ValueError("Expected preserved terminal failed control gate")
    audit = read_json(parent / "audit.json")
    expected = {
        "manifest_sha256": digest_file(ROOT / "data/manifests/mindset_v2_development100.json"),
        "calibration_sha256": digest_file(ROOT / "outputs/acl/mindset_joint_v2_27058/calibration/calibration.json"),
        "worker_sha256": digest_file(ROOT / "scripts/probe_attribute_contrasts.py"),
        "fixture_module_sha256": digest_file(ROOT / "scripts/attribute_contrast_fixtures.py"),
        "contract_module_sha256": digest_file(ROOT / "src/mindset/memory/attribute_contrast_contract.py"),
        "client_module_sha256": digest_file(ROOT / "src/mindset/acl/clients.py"),
    }
    if any(audit.get(k) != v for k, v in expected.items()):
        raise ValueError("Parent identity changed")
    controls = [r for r in read_json(parent / "inputs.json") if "expected_kind" in r]
    if stable_hash(controls) != audit["controls_hash"]:
        raise ValueError("Control cohort changed")
    rows = unique_rows(parent / "measurements.jsonl")
    if set(rows) != {r["id"] for r in controls} or len(rows) != 32:
        raise ValueError("Incomplete or changed failed controls")
    import spacy
    nlp = spacy.load("en_core_web_sm")
    if spacy.__version__ != "3.8.7" or nlp.meta["version"] != "3.8.0":
        raise ValueError("Parser revision mismatch")
    output = ROOT / f"outputs/acl/source_binding_repair_{os.environ['SLURM_JOB_ID']}"
    output.mkdir(parents=True, exist_ok=False)
    import hashlib
    parser_digest = hashlib.sha256(nlp.to_bytes()).hexdigest()
    write_json(output / "audit.json", {"parent_audit_sha256": digest_file(parent / "audit.json"),
        "parent_measurements_sha256": digest_file(parent / "measurements.jsonl"),
        "parent_failed_analysis_sha256": digest_file(parent / "control_analysis.json"),
        "binding_module_sha256": digest_file(ROOT / "src/mindset/memory/source_binding.py"),
        "worker_sha256": digest_file(__file__), "parser_bytes_sha256": parser_digest,
        "spacy": spacy.__version__, "parser": nlp.meta,
        "policy_selected_after_inspecting_failures_not_independent_confirmation": True,
        "semantic_or_world_truth_verified": False, "new_LLMinference_calls": 0,
        "full_runs_authorized": False}, immutable=True)
    functional = functional_checks(nlp)
    write_json(output / "functional_checks.json", functional, immutable=True)
    completed = []; started = perf_counter()
    for original in controls:
        row = rows[original["id"]]
        if any(row[k] != v for k, v in original.items()):
            raise ValueError("Stored control source/label differs")
        result = None; error = None
        try:
            if row["incomplete"]:
                raise ValueError("Original capped/empty output retained")
            # Functions receive ONLY raw model output and complete source reports.
            # Expected kind/direction are used strictly below for evaluation.
            result = apply_candidate(row["prediction"], row["reports"], nlp)
        except (ValueError, KeyError, TypeError) as exc:
            error = str(exc)
        record = {**original, "original_prediction": row["prediction"],
            "original_validation_error": row["validation_error"], "candidate": result, "candidate_error": error}
        append_row(output / "controls.jsonl", record); completed.append(record)
    valid = [r for r in completed if r["candidate_error"] is None]
    positives = [r for r in completed if r["expected_kind"] == "attribute_change"]
    correct = sum(r["candidate_error"] is None and r["candidate"]["update_proposal"] and
        r["original_prediction"]["direction"] == r["expected_direction"] for r in positives)
    false = sum(r["expected_kind"] != "attribute_change" and r["candidate_error"] is None and
        r["candidate"]["update_proposal"] for r in completed)
    report = {"cases": len(completed), "valid_candidate_shapes": len(valid),
        "candidate_errors": len(completed)-len(valid), "positive_controls": len(positives),
        "positive_proposals_correct_direction": correct, "negative_false_proposals": false,
        "unique_rebindings": sum(r["candidate"]["binding"]["binding_status"] == "unique_literal_source_rebinding" for r in valid),
        "nonchange_extra_fields_preserved_not_used_as_state": sum(r["candidate"]["binding"]["nonchange_extra_fields_ignored_for_state"] for r in valid),
        "uncertainty_holds": sum(bool(r["candidate"]["reference_check"]["warnings"]) for r in valid),
        "warning_reasons": dict(Counter(w["reason"] for r in valid for w in r["candidate"]["reference_check"]["warnings"])),
        "new_LLMinference_calls": 0, "world_updates": 0, "QA_calls": 0,
        "observed_candidate_control_gate": len(valid) == 32 and correct == len(positives) and false == 0,
        "single_pass_candidate_seconds_not_paper_latency": perf_counter()-started,
        "scope": "post-hoc replay feasibility; not fresh confirmation/natural accuracy/useful energy or novelty"}
    write_json(output / "analysis.json", report, immutable=True)
    write_json(output / "status.json", {"status": "complete_candidate_replay", "cases": len(completed),
        "new_LLMinference_calls": 0, "natural_source_measurements": 0, "world_updates": 0, "full_runs_authorized": False})
    print(report, flush=True)


if __name__ == "__main__":
    run()
