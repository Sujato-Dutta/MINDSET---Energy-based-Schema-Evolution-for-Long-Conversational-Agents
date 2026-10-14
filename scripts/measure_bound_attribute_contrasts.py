"""Original100 source contrasts, unchanged prompt; fresh controls gate natural calls."""
import os
import json
import subprocess
from pathlib import Path
from mindset.acl.pipeline import ROOT, server_client, request_json, record_environment
from mindset.acl.joint_iteration import protocol, groups_for
from mindset.acl.io import read_json, write_json, digest_file, unique_rows, append_row
from mindset.acl.clients import prompt_tokens
from mindset.acl.source_relation_validation import verify_prepared_pair
from mindset.memory.attribute_contrast_contract import SCHEMA, CONTRACT_HASH, contrast_messages
from mindset.utils.hashing import stable_hash
from bound_contrast_controls import fresh_controls


def run():
    if os.name == "nt" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("DGX Slurm only")
    manifest = ROOT / "data/manifests/mindset_v2_development100.json"
    calibration = ROOT / "outputs/acl/mindset_joint_v2_27058/calibration/calibration.json"
    cfg, cohort, _ = protocol(manifest, calibration)
    checkpoint = read_json(ROOT / "configs/checkpoints.lock.json")[cfg["primary_generator"]]
    replay = ROOT / "outputs/acl/source_binding_repair_27689"
    adapter = ROOT / "outputs/acl/source_binding_adapter_27689"
    ra = read_json(replay / "audit.json"); aa = read_json(adapter / "audit.json")
    if not read_json(replay / "analysis.json")["observed_candidate_control_gate"] or any([
        aa["active_binding_candidate_sha256"] != digest_file(ROOT / "src/mindset/memory/source_binding_candidate.py"),
        aa["preserved_binding_sha256"] != digest_file(ROOT / "src/mindset/memory/source_binding.py"),
        ra["worker_sha256"] != digest_file(ROOT / "scripts/repair_source_contrasts.py")]):
        raise ValueError("Repaired candidate identity/gate changed")
    parent = ROOT / "outputs/acl/natural_source_relations_r01"
    prepared = unique_rows(parent / "queries.jsonl"); measured = unique_rows(parent / "measurements/pairs.jsonl")
    parent_audit = read_json(parent / "measurements/audit.json")
    if read_json(parent / "measurements/status.json")["status"] != "complete_source_measurements" or any([
        parent_audit["manifest_sha256"] != digest_file(manifest),
        parent_audit["calibration_sha256"] != digest_file(calibration),
        parent_audit["source_queries_sha256"] != digest_file(parent / "queries.jsonl"),
        parent_audit["source_pairs_sha256"] != digest_file(parent / "unique_pairs.json")]):
        raise ValueError("Prepared source cohort changed")
    if set(prepared) != {f"{r['dataset']}/{r['id']}" for r in cohort["records"]}:
        raise ValueError("Original100 membership changed")
    seen = set()
    for dataset, cluster, cases in groups_for(cohort, cfg):
        for case in cases:
            for pair in prepared[f"{dataset}/{case.id}"]["measurement_pairs"]:
                verify_prepared_pair(pair, case.events, question_date=case.question_date)
                row = measured[pair["pair_key"]]
                if any(row[k] != v for k,v in pair.items()):
                    raise ValueError("Parent quote source differs")
                seen.add(pair["pair_key"])
    if seen != set(measured):
        raise ValueError("Pair cohort differs")
    proposals = [{"id": "natural:" + r["id"], "split": "natural_unlabelled_proposals",
        "reports": r["reports"], "parent_pair_key": r["id"]} for r in measured.values()
        if r["validation_error"] is None and r["prediction"]["relationship"] == "temporal_change"]
    proposals.sort(key=lambda r: r["id"]); controls = fresh_controls()
    if len(proposals) != 41 or len(controls) != 12:
        raise ValueError("Frozen bounded scope changed")
    out = ROOT / "outputs/acl/bound_attribute_contrasts_r01"; out.mkdir(parents=True, exist_ok=True)
    identity = {"manifest_sha256": digest_file(manifest), "calibration_sha256": digest_file(calibration),
        "parent_measurements_sha256": digest_file(parent / "measurements/pairs.jsonl"),
        "repaired_candidate_audit_sha256": digest_file(replay / "audit.json"),
        "repaired_adapter_sha256": digest_file(adapter / "audit.json"), "checkpoint": checkpoint,
        "seed": cfg["seed"], "contract_hash": CONTRACT_HASH, "max_output_tokens": 256,
        "worker_sha256": digest_file(__file__), "checker_sha256": digest_file(ROOT / "scripts/check_bound_contrast_outputs.py"),
        "controls_module_sha256": digest_file(ROOT / "scripts/bound_contrast_controls.py"),
        "client_module_sha256": digest_file(ROOT / "src/mindset/acl/clients.py"),
        "source_binding_sha256": aa["preserved_binding_sha256"],
        "binding_candidate_sha256": aa["active_binding_candidate_sha256"],
        "inputs_hash": stable_hash(controls + proposals),
        "selection": "all 41 earlier temporal proposals; no question outcomes/answers used",
        "prompt_unchanged_from_failed_first_contrast_gate": True,
        "controls_are_fresh_outputs_same_families_not_OOD": True,
        "no_memory_updates_energy_QA_or_hosted_calls": True, "full_runs_authorized": False}
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
            raise ValueError("Local-only source inference")
        if {r["id"] for r in request_json(url.rstrip("/") + "/models")["data"]} != {checkpoint["checkpoint"]}:
            raise ValueError("Wrong served model")
        client = server_client(cfg["primary_generator"], cfg, cache_path=out / "completion_cache.sqlite")
        client.allow_incomplete = True
        client.options = {**client.options, "response_format": {"type": "json_schema",
            "json_schema": {"name": "attribute_contrast", "strict": True, "schema": SCHEMA}}}
        done = unique_rows(out / "measurements.jsonl")
        if set(done) - {r["id"] for r in controls + proposals}:
            raise ValueError("Unexpected resumed rows")
        def measure(case):
            if case["id"] in done:
                return
            prompt = contrast_messages(case["reports"])
            c = client.complete(prompt, max_tokens=256, seed=cfg["seed"])
            if c.input_tokens != prompt_tokens(tokenizer, prompt, thinking=False):
                raise ValueError("Tokenization differs")
            payload = None; error = None
            try:
                payload = json.loads(c.text)
            except (ValueError, TypeError) as exc:
                error = str(exc)
            row = {**case, "prediction": payload, "decode_error": error, "incomplete": c.incomplete,
                "raw_response": c.text, "request_id": c.request_id, "cache_hit": c.cache_hit,
                "input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
                "request_latency_seconds": c.latency_seconds, "prompt_sha256": stable_hash(prompt),
                "source_job_id": os.environ["SLURM_JOB_ID"]}
            append_row(out / "measurements.jsonl", row); done[case["id"]] = row
            write_json(out / "status.json", {"status": "partial_bound_contrasts", "completed_cases": len(done),
                "maximum_cases": 53, "QA_calls": 0, "world_updates": 0})
            print(f"BOUND_CONTRAST {len(done)}/53 {case['id']}", flush=True)
        def check(mode):
            subprocess.run([str(ROOT / ".venv-syntax/bin/python"), str(ROOT / "scripts/check_bound_contrast_outputs.py"),
                "--output", str(out), "--mode", mode], check=True, cwd=ROOT)
        for case in controls:
            measure(case)
        if not (out / "control_validation/analysis.json").exists():
            check("controls")
        if not read_json(out / "control_validation/analysis.json")["fresh_output_control_gate"]:
            write_json(out / "status.json", {"status": "fresh_controls_failed_natural_not_run", "completed_cases": len(done),
                "QA_calls": 0, "world_updates": 0, "hosted_calls": 0})
            return
        for case in proposals:
            measure(case)
        if not (out / "complete_validation/analysis.json").exists():
            check("complete")
        report = read_json(out / "complete_validation/analysis.json")
        report.update(fresh_calls=sum(not r["cache_hit"] for r in done.values()),
            fresh_input_tokens=sum(r["input_tokens"] for r in done.values() if not r["cache_hit"]),
            fresh_output_tokens=sum(r["output_tokens"] for r in done.values() if not r["cache_hit"]))
        write_json(out / "analysis.json", report, immutable=True)
        write_json(out / "status.json", {"status": "complete_bound_contrasts", "completed_cases": len(done),
            "QA_calls": 0, "world_updates": 0, "hosted_calls": 0, "full_runs_authorized": False})


if __name__ == "__main__":
    run()
