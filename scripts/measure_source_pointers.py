"""Frozen 12+41 source cases with a new ID-only contract, local DGX reader only."""
import os
import json
import subprocess
from pathlib import Path
from mindset.acl.pipeline import ROOT, server_client, request_json, record_environment
from mindset.acl.joint_iteration import protocol
from mindset.acl.io import read_json, write_json, digest_file, unique_rows, append_row
from mindset.acl.clients import prompt_tokens
from mindset.memory.source_pointing import CONTRACT_HASH, pointing_messages, schema, validate_registry
from mindset.utils.hashing import stable_hash


def run():
    if os.name == "nt" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("DGX Slurm only")
    manifest = ROOT / "data/manifests/mindset_v2_development100.json"
    calibration = ROOT / "outputs/acl/mindset_joint_v2_27058/calibration/calibration.json"
    cfg, _, _ = protocol(manifest, calibration)
    source = ROOT / "outputs/acl/source_representation_27701"
    parent = ROOT / "outputs/acl/bound_attribute_contrasts_r01"
    a = read_json(source / "audit.json"); status = read_json(source / "analysis.json")
    if status["status"] != "complete_source_checks" or a["manifest_sha256"] != digest_file(manifest):
        raise ValueError("Source-only preflight not passed")
    if read_json(parent / "status.json")["status"] != "complete_bound_contrasts":
        raise ValueError("Original source cases incomplete")
    for filename, expected in a["modules_sha256"].items():
        if digest_file(ROOT / "src/mindset/memory" / filename) != expected:
            raise ValueError("Checked source representation changed")
    registries = unique_rows(source / "source_registries.jsonl")
    originals = read_json(parent / "inputs.json")
    if len(originals) != 53 or set(registries) != {c["id"] for c in originals}:
        raise ValueError("Bounded source membership differs")
    cases = []
    for case in originals:
        r = registries[case["id"]]
        if r["reports"] != case["reports"]:
            raise ValueError("Original complete reports changed")
        validate_registry(r["registry"], case["reports"])
        cases.append({**case, "registry": r["registry"]})
    checkpoint = read_json(ROOT / "configs/checkpoints.lock.json")[cfg["primary_generator"]]
    out = ROOT / "outputs/acl/source_pointers_r01"; out.mkdir(parents=True, exist_ok=True)
    audit = {"manifest_sha256": digest_file(manifest), "calibration_sha256": digest_file(calibration),
        "source_preflight_audit_sha256": digest_file(source / "audit.json"),
        "source_preflight_analysis_sha256": digest_file(source / "analysis.json"),
        "registries_sha256": digest_file(source / "source_registries.jsonl"),
        "parent_inputs_sha256": digest_file(parent / "inputs.json"), "inputs_hash": stable_hash(cases),
        "checkpoint": checkpoint, "seed": cfg["seed"], "max_output_tokens": 256, "contract_hash": CONTRACT_HASH,
        "worker_sha256": digest_file(__file__), "checker_sha256": digest_file(ROOT / "scripts/check_source_pointer_outputs.py"),
        "client_sha256": digest_file(ROOT / "src/mindset/acl/clients.py"), "modules_sha256": a["modules_sha256"],
        "selection": "same 12 constructed controls then all 41 previous temporal proposals; no QA filtering",
        "new_contract_not_same_prompt_ablation": True, "no_hosted_QA_calls_or_memory_updates": True,
        "source_only_no_query_gold_answer_in_prompt": True, "full_runs_authorized": False}
    import fcntl
    with (out / "worker.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(out / "audit.json", audit, immutable=True); write_json(out / "inputs.json", cases, immutable=True)
        record_environment(out / "continuations" / os.environ["SLURM_JOB_ID"])
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint["checkpoint"], revision=checkpoint["revision"],
            cache_dir=str(Path.home() / "MINDSET_v2/.cache/gpu_prototype/huggingface/hub"), local_files_only=True)
        url = os.environ.get("MINDSET_LOCAL_BASE_URL", "")
        if url.split("://")[-1].split(":")[0] not in ("localhost", "127.0.0.1"):
            raise ValueError("Local DGX inference only")
        if {r["id"] for r in request_json(url.rstrip("/") + "/models")["data"]} != {checkpoint["checkpoint"]}:
            raise ValueError("Served checkpoint differs")
        client = server_client(cfg["primary_generator"], cfg, cache_path=out / "completion_cache.sqlite")
        client.allow_incomplete = True
        base_options = dict(client.options); done = unique_rows(out / "measurements.jsonl")
        if set(done) - {c["id"] for c in cases}:
            raise ValueError("Unexpected resumed source cases")
        for c in cases:
            if c["id"] in done and any(done[c["id"]][k] != v for k,v in c.items()):
                raise ValueError("Resumed input changed")
        def measure(case):
            if case["id"] in done: return
            prompt = pointing_messages(case["registry"], case["reports"])
            client.options = {**base_options, "response_format": {"type": "json_schema", "json_schema":
                {"name": "source_pointer", "strict": True, "schema": schema(case["registry"], case["reports"])}}}
            completion = client.complete(prompt, max_tokens=256, seed=cfg["seed"])
            if completion.input_tokens != prompt_tokens(tokenizer, prompt, thinking=False):
                raise ValueError("Exact tokenization differs")
            prediction = None; error = None
            try: prediction = json.loads(completion.text)
            except (ValueError, TypeError) as exc: error = str(exc)
            row = {**case, "prediction": prediction, "decode_error": error, "raw_response": completion.text,
                "incomplete": completion.incomplete, "cache_hit": completion.cache_hit, "request_id": completion.request_id,
                "input_tokens": completion.input_tokens, "output_tokens": completion.output_tokens,
                "prompt_sha256": stable_hash(prompt), "request_latency_seconds": completion.latency_seconds,
                "source_job_id": os.environ["SLURM_JOB_ID"]}
            append_row(out / "measurements.jsonl", row); done[case["id"]] = row
            write_json(out / "status.json", {"status": "partial_source_pointers", "completed_cases": len(done), "maximum_cases": 53})
            print(f"SOURCE_POINTER {len(done)}/53 {case['id']}", flush=True)
        def check(mode):
            subprocess.run([str(ROOT / ".venv-syntax/bin/python"), str(ROOT / "scripts/check_source_pointer_outputs.py"),
                "--output", str(out), "--mode", mode], check=True, cwd=ROOT)
        for case in cases:
            if "expected_kind" in case: measure(case)
        if not (out / "control_validation/analysis.json").exists(): check("controls")
        if not read_json(out / "control_validation/analysis.json")["fresh_output_control_gate"]:
            write_json(out / "status.json", {"status": "controls_failed_natural_not_run", "completed_cases": len(done),
                "QA_calls": 0, "memory_updates": 0, "hosted_calls": 0})
            return
        for case in cases:
            if "expected_kind" not in case: measure(case)
        if not (out / "complete_validation/analysis.json").exists(): check("complete")
        report = read_json(out / "complete_validation/analysis.json")
        report.update(fresh_calls=sum(not r["cache_hit"] for r in done.values()),
            fresh_input_tokens=sum(r["input_tokens"] for r in done.values() if not r["cache_hit"]),
            fresh_output_tokens=sum(r["output_tokens"] for r in done.values() if not r["cache_hit"]))
        write_json(out / "analysis.json", report, immutable=True)
        write_json(out / "status.json", {"status": "complete_source_pointers", "completed_cases": len(done),
            "QA_calls": 0, "memory_updates": 0, "hosted_calls": 0, "full_runs_authorized": False})


if __name__ == "__main__": run()
