"""Interpret saved ID-only outputs; literal provenance is not semantic truth."""
import argparse
import os
import sys
import importlib.util
import hashlib
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mindset.acl.io import read_json, write_json, unique_rows, digest_file


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src/mindset/memory" / filename)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def run(out, mode):
    if os.name == "nt" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("DGX Slurm only")
    import spacy
    nlp = spacy.load("en_core_web_sm")
    if spacy.__version__ != "3.8.7" or nlp.meta["version"] != "3.8.0":
        raise ValueError("Parser identity differs")
    pointing = load("pointing_validator", "source_pointing.py")
    guard = load("pointing_reference_guard", "source_reference_guard.py")
    out = Path(out); cases = read_json(out / "inputs.json"); raw = unique_rows(out / "measurements.jsonl")
    expected = cases if mode == "complete" else [c for c in cases if "expected_kind" in c]
    if set(raw) != {c["id"] for c in expected}:
        raise ValueError("Missing or extra source outputs")
    records = []
    for case in expected:
        row = raw[case["id"]]
        if any(row[k] != v for k,v in case.items()):
            raise ValueError("Measured source inputs changed")
        error = None; result = None
        try:
            if row["incomplete"] or row["decode_error"] is not None:
                raise ValueError("Incomplete/invalid JSON output retained")
            decoded = pointing.decode_pointer(row["prediction"], case["registry"], case["reports"])
            docs = {r["source_id"]: nlp(r["text"]) for r in case["reports"]}
            check = guard.reference_uncertainty(decoded["binding"], case["reports"], docs)
            result = {**decoded, "reference_check": check, "update_proposal": check["update_proposal"]}
        except (ValueError, KeyError, TypeError) as exc:
            error = str(exc)
        records.append({**case, "prediction": row["prediction"], "candidate": result, "error": error})
    controls = [r for r in records if "expected_kind" in r]
    positive = [r for r in controls if r["expected_kind"] == "attribute_change"]
    correct = sum(r["error"] is None and r["candidate"]["update_proposal"] and
        r["prediction"]["direction"] == r["expected_direction"] for r in positive)
    false = sum(r["expected_kind"] != "attribute_change" and r["error"] is None and
        r["candidate"]["update_proposal"] for r in controls)
    natural = [r for r in records if "expected_kind" not in r]
    report = {"controls": len(controls), "positive_controls": len(positive),
        "positive_proposals_correct_direction": correct, "negative_false_proposals": false,
        "control_invalid": sum(r["error"] is not None for r in controls),
        "control_kind_correct": sum(r["error"] is None and r["prediction"]["kind"] == r["expected_kind"] for r in controls),
        "fresh_output_control_gate": len(controls) == 12 and len(positive) == 6 and correct == 6 and false == 0
            and all(r["error"] is None for r in controls),
        "natural_pairs": len(natural), "natural_invalid": sum(r["error"] is not None for r in natural),
        "provisional_natural_types": dict(Counter(r["prediction"]["kind"] for r in natural if r["error"] is None)),
        "natural_update_proposals": sum(r["error"] is None and r["candidate"]["update_proposal"] for r in natural),
        "natural_uncertainty_holds": sum(r["error"] is None and bool(r["candidate"]["reference_check"]["warnings"]) for r in natural),
        "generated_quote_text_allowed": False,
        "scope": "same-family source-contract feasibility; no natural semantic truth, QA/energy/novelty proof"}
    target = out / ("control_validation" if mode == "controls" else "complete_validation")
    write_json(target / "audit.json", {"worker_sha256": digest_file(__file__), "input_sha256": digest_file(out / "inputs.json"),
        "measurements_sha256": digest_file(out / "measurements.jsonl"), "parser_sha256": hashlib.sha256(nlp.to_bytes()).hexdigest(),
        "mode": mode, "modules_sha256": {n:digest_file(ROOT / "src/mindset/memory" / n)
            for n in ("source_pointing.py", "source_reference_guard.py", "source_binding.py", "source_binding_candidate.py")}}, immutable=True)
    write_json(target / "records.json", records, immutable=True)
    write_json(target / "analysis.json", report, immutable=True)
    print(report, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=("controls", "complete"), required=True)
    a = p.parse_args(); run(a.output, a.mode)
