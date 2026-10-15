"""DGX-only source pointing/within-report checks, followed by source-only coverage.

No model calls, answers, gold matching, world updates or parameter fitting.
"""
import os
import sys
import importlib.util
import hashlib
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mindset.acl.io import read_json, unique_rows, write_json, append_row, digest_file
from mindset.utils.hashing import stable_hash


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src/mindset/memory" / filename)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def run():
    if os.name == "nt" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("DGX Slurm only")
    import spacy
    nlp = spacy.load("en_core_web_sm")
    if spacy.__version__ != "3.8.7" or nlp.meta["version"] != "3.8.0":
        raise ValueError("Parser identity differs")
    pointing = load("source_pointing_check", "source_pointing.py")
    guard = load("source_reference_check", "source_reference_guard.py")
    numeric = load("numeric_microtransition_check", "numeric_microtransitions.py")
    out = ROOT / f"outputs/acl/source_representation_{os.environ['SLURM_JOB_ID']}"
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / "outputs/acl/mindset_syntax_27302/inputs"
    audit = read_json(source / "audit.json")
    manifest = ROOT / "data/manifests/mindset_v2_development100.json"
    if audit["manifest_sha256"] != digest_file(manifest) or audit["input_sha256"] != digest_file(source / "inputs.jsonl"):
        raise ValueError("Source export/manifest differs")
    rows = unique_rows(source / "inputs.jsonl")
    episodes = [r for r in rows.values() if r["kind"] == "episode"]
    queries = [r for r in rows.values() if r["kind"] == "query"]
    if len(episodes) != 30595 or len(queries) != 100:
        raise ValueError("Bounded original100 source scope differs")
    for r in rows.values():
        if stable_hash({k:v for k,v in r.items() if k != "input_sha256"}) != r["input_sha256"]:
            raise ValueError("Raw source fingerprint differs")
    write_json(out / "audit.json", {"manifest_sha256": digest_file(manifest), "input_sha256": audit["input_sha256"],
        "worker_sha256": digest_file(__file__), "modules_sha256": {n:digest_file(ROOT / "src/mindset/memory" / n)
            for n in ("source_pointing.py", "source_reference_guard.py", "source_binding.py", "source_binding_candidate.py",
                      "numeric_microtransitions.py", "report_frames.py")},
        "parser_sha256": hashlib.sha256(nlp.to_bytes()).hexdigest(), "spacy": spacy.__version__, "parser_meta": nlp.meta,
        "selected_after_previous_failures": True, "fit_or_QA_calls": 0,
        "scope": "constructed functionality and source candidate coverage; NOT accuracy/energy usefulness/novelty"}, immutable=True)
    checks = [
        ("My Twitter follower count has jumped from 420 to 540 over the past month.", ("420", "540")),
        ("My budget increased from 120 to 180.", ("120", "180")),
        ("I changed my budget from 120 dollars to 80 dollars.", ("120", "80")),
        ("My rent decreased from 900 to 700.", ("900", "700")),
        ("My budget did not increase from 120 to 180.", None),
        ("My budget might increase from 120 to 180.", None),
        ("If my budget increases from 120 to 180, I will buy a bicycle.", None),
        ("Did my budget increase from 120 to 180?", None),
        ('"My budget increased from 120 to 180."', None),
        ("I hope my budget increases from 120 to 180.", None),
        ("Your budget increased from 120 to 180.", None),
        ("My budget increased from 120 to 120.", None),
        ("My budget increased from about 120 to 180.", None),
        ("Web search results: My budget increased from 120 to 180.", None),
    ]
    for i, (text, expected) in enumerate(checks):
        doc = nlp(text)
        result = numeric.compile_numeric_reports(doc, text, "user", episode_id="functional", observation_time="2025-05-29")
        observed = [(r["before_value"]["numeric_normalization_alternative"], r["after_value"]["numeric_normalization_alternative"]) for r in result]
        passed = observed == ([] if expected is None else [expected])
        record = {"id": str(i), "text": text, "expected": expected, "result": result, "passed": passed,
            "parse": [{"text": t.text, "head": t.head.text, "dep": t.dep_, "pos": t.pos_, "entity": t.ent_type_} for t in doc]}
        append_row(out / "numeric_functional.jsonl", record)
        if not passed:
            raise ValueError(f"Numeric functional check failed: {i}; inspect saved literal parse")
    parent = ROOT / "outputs/acl/bound_attribute_contrasts_r01"
    if read_json(parent / "status.json")["status"] != "complete_bound_contrasts":
        raise ValueError("Parent source contrast output is not terminal")
    pairs = read_json(parent / "inputs.json")
    if len(pairs) != 53:
        raise ValueError("Parent bounded source cases differ")
    for case in pairs:
        reports = case["reports"]; docs = {r["source_id"]: nlp(r["text"]) for r in reports}
        reg = pointing.registry(reports, docs); pointing.validate_registry(reg, reports)
        # Force a complete-source positive solely to test conservative reference
        # handling. This is NOT the model's interpretation or natural truth.
        ids = {u["source_id"]: u["id"] for u in reg["units"] if u["kind"] == "full_report"}
        forced = None; error = None
        try:
            decoded = pointing.decode_pointer({"kind": "attribute_change", "variable": "forced_guard_probe",
                "direction": "a_to_b", "a_unit": ids["A"], "b_unit": ids["B"]}, reg, reports)
            forced = guard.reference_uncertainty(decoded["binding"], reports, docs)
        except ValueError as exc:
            error = str(exc)
        append_row(out / "source_registries.jsonl", {"id": case["id"], "reports": reports, "registry": reg,
            "forced_reference_probe": forced, "forced_probe_error": error})
    counts = Counter(); total = 0; source_hits = 0
    for doc, row in nlp.pipe(((r["text"], r) for r in episodes), as_tuples=True, batch_size=64, n_process=1):
        candidates = numeric.compile_numeric_reports(doc, row["text"], row["speaker"],
            episode_id=row["source_id"], observation_time=row["timestamp"])
        for c in candidates:
            for key in ("property_span", "predicate_span", "before_value", "after_value", "scope_span"):
                s = c[key]
                if row["text"][s["char_start"]:s["char_end"]] != s["text"]:
                    raise ValueError("Nonliteral compiled interval")
        if candidates:
            append_row(out / "numeric_candidates.jsonl", {"id": row["id"], "input_sha256": row["input_sha256"],
                "text": row["text"], "candidates": candidates})
            source_hits += 1; counts[row["dataset"]] += len(candidates)
        total += 1
        if total % 5000 == 0: print(f"source episodes {total}/{len(episodes)}", flush=True)
    write_json(out / "analysis.json", {"status": "complete_source_checks", "episodes": total, "queries_not_parsed": len(queries),
        "functional_numeric_checks": len(checks), "source_pair_registries": len(pairs), "episodes_with_candidates": source_hits,
        "numeric_candidate_counts": dict(counts), "model_calls": 0, "memory_updates": 0,
        "scope": "candidate coverage only; not natural accuracy, QA gain, energy benefit or novelty"})


if __name__ == "__main__": run()
