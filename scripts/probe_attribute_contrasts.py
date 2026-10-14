"""Grounded contrast feasibility; source-only local calls, no memory updates/QA."""
import os
import json
from pathlib import Path
from collections import Counter
from mindset.acl.pipeline import ROOT, server_client, request_json, record_environment
from mindset.acl.joint_iteration import protocol, groups_for
from mindset.acl.io import read_json, write_json, unique_rows, append_row, digest_file
from mindset.acl.clients import prompt_tokens
from mindset.acl.source_relation_validation import verify_prepared_pair
from mindset.memory.source_relation_contract import CONTRACT_HASH as PARENT_HASH, validate_relation
from mindset.memory.attribute_contrast_contract import SCHEMA, CONTRACT_HASH, contrast_messages, validate_contrast
from mindset.utils.hashing import stable_hash
from attribute_contrast_fixtures import contrast_fixtures


def control_summary(rows):
    valid = [r for r in rows if r["validation_error"] is None]
    positives = [r for r in rows if r["expected_kind"] == "attribute_change"]
    correct_positive = sum(r["validation_error"] is None and r["prediction"]["kind"] == "attribute_change"
        and r["prediction"]["direction"] == r["expected_direction"] for r in positives)
    false_admissions = sum(r["expected_kind"] != "attribute_change" and r["validation_error"] is None
        and r["certificate"]["provisional_update_proposal"] for r in rows)
    return {"cases": len(rows), "valid": len(valid), "invalid_or_incomplete": len(rows)-len(valid),
        "positive_controls": len(positives), "correct_positive_direction": correct_positive,
        "false_update_proposals": false_admissions,
        "kind_correct": sum(r["prediction"]["kind"] == r["expected_kind"] for r in valid),
        "natural_measurement_gate": len(rows) == 32 and len(valid) == 32 and
            correct_positive == len(positives) and false_admissions == 0,
        "not_semantic_accuracy_on_natural_sources": True}


def run():
    if os.name == "nt" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("DGX Slurm only")
    manifest = ROOT / "data/manifests/mindset_v2_development100.json"
    calibration = ROOT / "outputs/acl/mindset_joint_v2_27058/calibration/calibration.json"
    cfg, cohort, _ = protocol(manifest, calibration)
    checkpoint = read_json(ROOT / "configs/checkpoints.lock.json")[cfg["primary_generator"]]
    parent = ROOT / "outputs/acl/natural_source_relations_r01"
    measured = parent / "measurements"
    if read_json(measured / "status.json")["status"] != "complete_source_measurements":
        raise ValueError("Parent source measurements incomplete")
    pa = read_json(measured / "audit.json")
    for key, expected in {"manifest_sha256": digest_file(manifest), "calibration_sha256": digest_file(calibration),
        "checkpoint": checkpoint, "contract_hash": PARENT_HASH, "seed": cfg["seed"],
        "source_queries_sha256": digest_file(parent / "queries.jsonl"),
        "source_pairs_sha256": digest_file(parent / "unique_pairs.json"),
        "worker_sha256": digest_file(ROOT / "scripts/measure_natural_source_relations.py")}.items():
        if pa.get(key) != expected:
            raise ValueError("Parent measurement identity differs: " + key)
    prepared = unique_rows(parent / "queries.jsonl"); pairs = unique_rows(measured / "pairs.jsonl")
    expected_queries = {f"{r['dataset']}/{r['id']}" for r in cohort["records"]}
    if set(prepared) != expected_queries:
        raise ValueError("Original100 query panel changed")
    seen = set()
    for dataset, cluster, cases in groups_for(cohort, cfg):
        for case in cases:
            for pair in prepared[f"{dataset}/{case.id}"]["measurement_pairs"]:
                verify_prepared_pair(pair, case.events, question_date=case.question_date)
                row = pairs[pair["pair_key"]]
                if any(row[k] != v for k, v in pair.items()):
                    raise ValueError("Measured parent source differs")
                if row["validation_error"] is None:
                    if validate_relation(row["prediction"], row["reports"]) != row["certificate"]:
                        raise ValueError("Invalid parent relation certificate")
                seen.add(pair["pair_key"])
    if seen != set(pairs):
        raise ValueError("Parent pair panel differs")
    proposals = [{"id": "natural:" + r["id"], "split": "natural_proposals_unlabelled",
                  "reports": r["reports"], "parent_pair_key": r["id"]} for r in pairs.values()
                 if r["validation_error"] is None and r["prediction"]["relationship"] == "temporal_change"]
    proposals.sort(key=lambda r: r["id"])
    controls = contrast_fixtures()
    if len(proposals) != 41 or len(controls) != 32:
        raise ValueError("Bounded input cohort differs")
    out = ROOT / "outputs/acl/attribute_contrasts_r01"; out.mkdir(parents=True, exist_ok=True)
    identity = {"manifest_sha256": digest_file(manifest), "calibration_sha256": digest_file(calibration),
        "parent_pairs_sha256": digest_file(measured / "pairs.jsonl"), "checkpoint": checkpoint,
        "seed": cfg["seed"], "contract_hash": CONTRACT_HASH, "max_output_tokens": 256,
        "worker_sha256": digest_file(__file__),
        "contract_module_sha256": digest_file(ROOT / "src/mindset/memory/attribute_contrast_contract.py"),
        "fixture_module_sha256": digest_file(ROOT / "scripts/attribute_contrast_fixtures.py"),
        "client_module_sha256": digest_file(ROOT / "src/mindset/acl/clients.py"),
        "controls_hash": stable_hash(controls), "natural_inputs_hash": stable_hash(proposals),
        "selection": "all 41 valid prior temporal proposals, no answer/outcome filtering",
        "new_controls_authored_after_inspecting_failures_not_independent_confirmation": True,
        "no_question_reference_gold_category_in_model_inputs": True,
        "no_world_updates_or_energy_or_QA": True, "full_runs_authorized": False}
    import fcntl
    with (out / "worker.lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(out / "audit.json", identity, immutable=True)
        write_json(out / "inputs.json", controls + proposals, immutable=True)
        record_environment(out / "continuations" / os.environ["SLURM_JOB_ID"])
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint["checkpoint"], revision=checkpoint["revision"],
            cache_dir=str(Path.home() / "MINDSET_v2/.cache/gpu_prototype/huggingface/hub"), local_files_only=True)
        url = os.environ.get("MINDSET_LOCAL_BASE_URL", "")
        if url.split("://")[-1].split(":")[0] not in ("localhost", "127.0.0.1"):
            raise ValueError("Local model only")
        if {r["id"] for r in request_json(url.rstrip("/") + "/models")["data"]} != {checkpoint["checkpoint"]}:
            raise ValueError("Wrong served model")
        client = server_client(cfg["primary_generator"], cfg, cache_path=out / "completion_cache.sqlite")
        client.allow_incomplete = True
        client.options = {**client.options, "response_format": {"type": "json_schema",
            "json_schema": {"name": "attribute_contrast", "strict": True, "schema": SCHEMA}}}
        done = unique_rows(out / "measurements.jsonl")
        if set(done) - {r["id"] for r in controls + proposals}:
            raise ValueError("Unexpected resumed cases")
        def measure(case):
            if case["id"] in done:
                return
            prompt = contrast_messages(case["reports"])
            c = client.complete(prompt, max_tokens=256, seed=cfg["seed"])
            if c.input_tokens != prompt_tokens(tokenizer, prompt, thinking=False):
                raise ValueError("Tokenization differs")
            payload = certificate = None; error = None
            try:
                if c.incomplete:
                    raise ValueError("Capped/empty output")
                payload = json.loads(c.text); certificate = validate_contrast(payload, case["reports"])
            except (ValueError, KeyError, TypeError) as exc:
                error = str(exc)
            row = {**case, "prediction": payload, "certificate": certificate,
                "validation_error": error, "raw_response": c.text, "request_id": c.request_id,
                "prompt_sha256": stable_hash(prompt), "input_tokens": c.input_tokens,
                "output_tokens": c.output_tokens, "cache_hit": c.cache_hit, "incomplete": c.incomplete,
                "request_latency_seconds": c.latency_seconds, "source_job_id": os.environ["SLURM_JOB_ID"]}
            append_row(out / "measurements.jsonl", row); done[case["id"]] = row
            write_json(out / "status.json", {"status": "partial_contrast_measurements", "completed_cases": len(done),
                "maximum_cases": 73, "world_updates": 0, "qa_calls": 0, "hosted_API_calls": 0})
            print(f"CONTRAST {len(done)}/73 {case['id']} valid={error is None}", flush=True)
        for case in controls:
            measure(case)
        summary = control_summary([done[r["id"]] for r in controls])
        write_json(out / "control_analysis.json", summary, immutable=True)
        if not summary["natural_measurement_gate"]:
            write_json(out / "status.json", {"status": "controls_failed_natural_not_run",
                "completed_cases": len(done), "world_updates": 0, "qa_calls": 0, "hosted_API_calls": 0})
            print("CONTROL_GATE_FAILED", summary, flush=True); return
        for case in proposals:
            measure(case)
        natural = [done[r["id"]] for r in proposals]
        write_json(out / "analysis.json", {"controls": summary, "natural_cases": len(natural),
            "natural_invalid": sum(r["validation_error"] is not None for r in natural),
            "provisional_natural_kinds": dict(Counter(r["prediction"]["kind"] for r in natural if r["validation_error"] is None)),
            "natural_update_proposals": sum(r["validation_error"] is None and r["certificate"]["provisional_update_proposal"] for r in natural),
            "fresh_calls": sum(not r["cache_hit"] for r in done.values()),
            "fresh_input_tokens": sum(r["input_tokens"] for r in done.values() if not r["cache_hit"]),
            "fresh_output_tokens": sum(r["output_tokens"] for r in done.values() if not r["cache_hit"]),
            "natural_semantic_accuracy_or_useful_energy_or_novelty_proven": False}, immutable=True)
        write_json(out / "status.json", {"status": "complete_contrast_measurements", "completed_cases": len(done),
            "world_updates": 0, "qa_calls": 0, "hosted_API_calls": 0, "full_runs_authorized": False})


if __name__ == "__main__":
    run()
