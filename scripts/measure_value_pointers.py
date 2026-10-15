"""Bounded value-backed post-state experiment; pinned local DGX model only."""
import os
import json
import argparse
import subprocess
from pathlib import Path
from mindset.acl.pipeline import ROOT,server_client,request_json,record_environment
from mindset.acl.joint_iteration import protocol
from mindset.acl.io import read_json,write_json,unique_rows,digest_file,append_row
from mindset.acl.clients import prompt_tokens
from mindset.memory.value_pointing import messages,schema,validate_registry,CONTRACT_HASH
from mindset.utils.hashing import stable_hash


def run(preflight):
    if os.name=="nt" or not os.environ.get("SLURM_JOB_ID"):raise RuntimeError("DGX Slurm only")
    manifest=ROOT/"data/manifests/mindset_v2_development100.json"
    calibration=ROOT/"outputs/acl/mindset_joint_v2_27058/calibration/calibration.json"
    cfg,_,_=protocol(manifest,calibration)
    source=Path(preflight);a=read_json(source/"audit.json")
    if read_json(source/"analysis.json")["status"]!="complete_value_preflight" or a["manifest_sha256"]!=digest_file(manifest):
        raise ValueError("Checked source preflight incomplete/different")
    for n,h in a["modules_sha256"].items():
        if digest_file(ROOT/"src/mindset/memory"/n)!=h:raise ValueError("Preflight module changed")
    cases=read_json(source/"inputs.json")
    if len(cases)!=53 or len({c["id"] for c in cases})!=53:raise ValueError("Bounded source scope differs")
    for c in cases:validate_registry(c["registry"],c["reports"])
    checkpoint=read_json(ROOT/"configs/checkpoints.lock.json")[cfg["primary_generator"]]
    out=ROOT/"outputs/acl/value_pointers_r01";out.mkdir(parents=True,exist_ok=True)
    identity={"manifest_sha256":digest_file(manifest),"calibration_sha256":digest_file(calibration),
        "source_audit_sha256":digest_file(source/"audit.json"),"source_inputs_sha256":digest_file(source/"inputs.json"),
        "inputs_hash":stable_hash(cases),"contract_hash":CONTRACT_HASH,"modules_sha256":a["modules_sha256"],
        "worker_sha256":digest_file(__file__),"checker_sha256":digest_file(ROOT/"scripts/check_value_pointer_outputs.py"),
        "client_sha256":digest_file(ROOT/"src/mindset/acl/clients.py"),"checkpoint":checkpoint,"seed":cfg["seed"],
        "max_output_tokens":256,"max_context_tokens":16384,"decision_coefficients_fitted":False,
        "controlled_context_coexistence_requires_split_not_revision":True,
        "new_contract_and_objective_not_single_factor_ablation":True,"no_queries_gold_answers_in_inference":True,
        "same_family_exposed_controls_not_independent_confirmation":True,"full_runs_authorized":False}
    import fcntl
    with (out/"worker.lock").open("a+b") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(out/"audit.json",identity,immutable=True);write_json(out/"inputs.json",cases,immutable=True)
        record_environment(out/"continuations"/os.environ["SLURM_JOB_ID"])
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(checkpoint["checkpoint"],revision=checkpoint["revision"],
            cache_dir=str(Path.home()/"MINDSET_v2/.cache/gpu_prototype/huggingface/hub"),local_files_only=True)
        url=os.environ.get("MINDSET_LOCAL_BASE_URL","")
        if url.split("://")[-1].split(":")[0] not in ("localhost","127.0.0.1"):raise ValueError("Local DGX only")
        if {r["id"] for r in request_json(url.rstrip("/")+"/models")["data"]}!={checkpoint["checkpoint"]}:
            raise ValueError("Served model differs")
        client=server_client(cfg["primary_generator"],cfg,cache_path=out/"completion_cache.sqlite")
        client.allow_incomplete=True;base_options=dict(client.options);done=unique_rows(out/"measurements.jsonl")
        if set(done)-{c["id"] for c in cases}:raise ValueError("Unexpected resumed source rows")
        for c in cases:
            if c["id"] in done and any(done[c["id"]][k]!=v for k,v in c.items()):raise ValueError("Resumed inputs differ")
        def measure(case):
            if case["id"] in done:return
            prompt=messages(case["registry"],case["reports"]);count=prompt_tokens(tokenizer,prompt,thinking=False)
            if count+256>16384:
                row={**case,"prediction":None,"decode_error":"context_limit_no_silent_trimming","incomplete":True,
                    "raw_response":"","request_id":None,"cache_hit":False,"no_request":True,
                    "input_tokens":0,"output_tokens":0,"request_latency_seconds":None,"prompt_sha256":stable_hash(prompt)}
            else:
                client.options={**base_options,"response_format":{"type":"json_schema","json_schema":
                    {"name":"value_pointer","strict":True,"schema":schema(case["registry"],case["reports"])}}}
                c=client.complete(prompt,max_tokens=256,seed=cfg["seed"])
                if c.input_tokens!=count:raise ValueError("Tokenization mismatch")
                prediction=None;error=None
                try:prediction=json.loads(c.text)
                except (ValueError,TypeError) as exc:error=str(exc)
                row={**case,"prediction":prediction,"decode_error":error,"incomplete":c.incomplete,"raw_response":c.text,
                    "request_id":c.request_id,"cache_hit":c.cache_hit,"no_request":False,"input_tokens":c.input_tokens,
                    "output_tokens":c.output_tokens,"request_latency_seconds":c.latency_seconds,"prompt_sha256":stable_hash(prompt)}
            row["source_job_id"]=os.environ["SLURM_JOB_ID"]
            append_row(out/"measurements.jsonl",row);done[case["id"]]=row
            write_json(out/"status.json",{"status":"partial_value_pointers","completed_cases":len(done),"maximum_cases":53,
                "QA_hosted_calls_or_actual_memory_updates":0})
            print(f"VALUE_POINTER {len(done)}/53 {case['id']}",flush=True)
        def check(mode):
            subprocess.run([str(ROOT/".venv-syntax/bin/python"),str(ROOT/"scripts/check_value_pointer_outputs.py"),
                "--output",str(out),"--mode",mode],check=True,cwd=ROOT)
        for case in cases:
            if "expected_kind" in case:measure(case)
        if not (out/"control_validation/analysis.json").exists():check("controls")
        if not read_json(out/"control_validation/analysis.json")["control_gate"]:
            write_json(out/"status.json",{"status":"controls_failed_natural_not_run","completed_cases":len(done),
                "QA_hosted_calls_or_actual_memory_updates":0})
            return
        for case in cases:
            if "expected_kind" not in case:measure(case)
        if not (out/"complete_validation/analysis.json").exists():check("complete")
        report=read_json(out/"complete_validation/analysis.json")
        fresh=[r for r in done.values() if not r["cache_hit"] and not r["no_request"]]
        report.update(fresh_calls=len(fresh),fresh_input_tokens=sum(r["input_tokens"] for r in fresh),
            fresh_output_tokens=sum(r["output_tokens"] for r in fresh))
        write_json(out/"analysis.json",report,immutable=True)
        write_json(out/"status.json",{"status":"complete_value_pointers","completed_cases":len(done),
            "QA_hosted_calls_or_actual_memory_updates":0,"full_runs_authorized":False})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--preflight",required=True);a=p.parse_args();run(a.preflight)
