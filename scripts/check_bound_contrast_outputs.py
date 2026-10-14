"""Pinned syntax subprocess in the same Slurm allocation; no LLM calls."""
import os
import sys
import argparse
import importlib.util
import hashlib
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from mindset.acl.io import read_json, write_json, unique_rows, digest_file


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def run(out, mode):
    if os.name == "nt" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("DGX Slurm only")
    import spacy
    nlp = spacy.load("en_core_web_sm")
    if spacy.__version__ != "3.8.7" or nlp.meta["version"] != "3.8.0":
        raise ValueError("Parser revision mismatch")
    candidate = load("binding_candidate", ROOT / "src/mindset/memory/source_binding_candidate.py")
    worker = load("binding_worker", ROOT / "scripts/repair_source_contrasts.py")
    worker.binding_module = candidate
    out = Path(out); inputs = read_json(out / "inputs.json"); raw = unique_rows(out / "measurements.jsonl")
    expected = inputs if mode == "complete" else [r for r in inputs if "expected_kind" in r]
    if set(raw) != {r["id"] for r in expected}:
        raise ValueError("Incomplete or extra measurement cohort")
    target = out / ("control_validation" if mode == "controls" else "complete_validation")
    write_json(target / "audit.json", {"worker_sha256": digest_file(__file__),
        "original_worker_sha256": digest_file(ROOT / "scripts/repair_source_contrasts.py"),
        "binding_candidate_sha256": digest_file(ROOT / "src/mindset/memory/source_binding_candidate.py"),
        "binding_base_sha256": digest_file(ROOT / "src/mindset/memory/source_binding.py"),
        "input_sha256": digest_file(out / "inputs.json"), "raw_measurements_sha256": digest_file(out / "measurements.jsonl"),
        "parser_bytes_sha256": hashlib.sha256(nlp.to_bytes()).hexdigest(), "parser": nlp.meta,
        "spacy": spacy.__version__, "mode": mode, "semantic_truth_verified": False}, immutable=True)
    write_json(target / "functional_checks.json", worker.functional_checks(nlp), immutable=True)
    records = []
    for case in expected:
        original = raw[case["id"]]
        if any(original[k] != v for k, v in case.items()):
            raise ValueError("Measurement inputs differ")
        error = None; result = None
        try:
            if original["incomplete"]:
                raise ValueError("Original capped/empty output retained")
            if original["decode_error"] is not None:
                raise ValueError("JSON decoding failed")
            result = worker.apply_candidate(original["prediction"], case["reports"], nlp)
        except (ValueError, KeyError, TypeError) as exc:
            error = str(exc)
        records.append({**case, "prediction": original["prediction"], "candidate": result, "error": error})
    write_json(target / "records.json", records, immutable=True)
    controls = [r for r in records if "expected_kind" in r]
    positives = [r for r in controls if r["expected_kind"] == "attribute_change"]
    correct = sum(r["error"] is None and r["candidate"]["update_proposal"] and
        r["prediction"]["direction"] == r["expected_direction"] for r in positives)
    false = sum(r["expected_kind"] != "attribute_change" and r["error"] is None and
        r["candidate"]["update_proposal"] for r in controls)
    natural = [r for r in records if "expected_kind" not in r]
    report = {"controls": len(controls), "positive_controls": len(positives),
        "positive_proposals_correct_direction": correct, "negative_false_proposals": false,
        "control_invalid": sum(r["error"] is not None for r in controls),
        "control_kind_correct": sum(r["error"] is None and r["prediction"]["kind"] == r["expected_kind"] for r in controls),
        "fresh_output_control_gate": len(controls) == 12 and correct == len(positives) and false == 0 and all(r["error"] is None for r in controls),
        "natural_pairs": len(natural), "natural_invalid": sum(r["error"] is not None for r in natural),
        "provisional_natural_types": dict(Counter(r["prediction"]["kind"] for r in natural if r["error"] is None)),
        "natural_update_proposals": sum(r["error"] is None and r["candidate"]["update_proposal"] for r in natural),
        "natural_uncertainty_holds": sum(r["error"] is None and bool(r["candidate"]["reference_check"]["warnings"]) for r in natural),
        "scope": "fresh same-family source-instrument feasibility; no natural semantic accuracy/energy/QA/novelty proof"}
    write_json(target / "analysis.json", report, immutable=True)
    print(report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("controls", "complete"), required=True)
    args = parser.parse_args(); run(args.output, args.mode)
