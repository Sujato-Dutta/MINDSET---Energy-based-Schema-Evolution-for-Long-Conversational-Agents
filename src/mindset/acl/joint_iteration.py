"""Audited development-100 evaluation; isolated from paper-frozen execution."""
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import os
import subprocess
import sys
import numpy as np
from mindset.acl.pipeline import (ROOT,config,cases_for,local_stack,record_environment,source_fingerprint,
                                 rank_metrics,write_csv,retrieval_request_k,server_client)
from mindset.acl.calibration import validate_calibration
from mindset.acl.joint_calibration import validate_joint
from mindset.acl.iteration import verify_iteration,candidate_variants,variant_hits,METRICS
from mindset.acl.io import read_json,write_json,digest_file,unique_rows,append_row
from mindset.acl.runtime import CachedEncoder,make_method
from mindset.acl.evidence import EvidenceCache
from mindset.acl.clients import load_tokenizer,messages,prompt_tokens,ANSWER_PROMPT_HASH
from mindset.acl.latency import latency_summary
from mindset.methods.compact import CompactConfig
from mindset.methods.joint import JointMindsetMemory
from mindset.evaluation.locomo import locomo_native_score
from mindset.evaluation.metrics import token_f1
from mindset.datasets.locomo import CATEGORY_NAMES

ARMS=("joint","joint_preselected","compact","flat48","lightmem","vector_rag","nemori")
LIGHTMEM_SHA="55071abd26e1058e7bdece2c8ec61f062a7c878634f11e8ebe89cfdff483b82e"

def protocol(manifest_path,calibration_path):
    cfg=config(); validate_calibration(cfg,ROOT)
    manifest=read_json(manifest_path); verify_iteration(manifest,cfg)
    calibration=validate_joint(calibration_path)
    if digest_file(ROOT/"src/mindset/baselines/lightmem.py")!=LIGHTMEM_SHA:
        raise ValueError("LightMem implementation is not the preserved version")
    return cfg,manifest,calibration

def identity(cfg,manifest_path,calibration_path,scope,methods):
    return {"method_name":"mindset","implementation":"v2_joint_prototype","configuration":cfg,
        "source_fingerprint":source_fingerprint(),"worker_source_sha256":digest_file(Path(__file__)),
        "manifest_sha256":digest_file(manifest_path),"joint_calibration_sha256":digest_file(calibration_path),
        "lightmem_source_sha256":LIGHTMEM_SHA,"scope":scope,"methods":list(methods),
        "checkpoint_lock_sha256":digest_file(ROOT/"configs/checkpoints.lock.json"),
        "budget":{"mindset_ce_candidates":48,"raw_evidence_episodes":8,"evidence_tokens":8000,
                  "lightmem_returned_hits":retrieval_request_k("lightmem",cfg),"lightmem_ce_candidates":"up to 16",
                  "compute_matched_baselines":False},"full_runs_authorized":False}

def build(arm,events,cfg,parameters,encoder,reranker):
    if arm in ("joint","joint_preselected"):
        params={**{k:v for k,v in cfg["mindset"].items() if k!="profile"},**parameters}
        m=JointMindsetMemory(CompactConfig(**params),encoder=encoder,local_reranker=reranker,
                            joint_targets=arm=="joint")
        m.ingest(events); return m
    return make_method("mindset" if arm in ("compact","flat48") else arm,events,cfg,encoder,reranker)

def get_hits(arm,memory,query,cfg):
    if arm=="flat48":
        return variant_hits(memory,query,{"flat_hybrid48":candidate_variants(memory,query)["flat_hybrid48"]})["flat_hybrid48"]
    return memory.retrieve(query,8 if arm=="lightmem" else 40)

def groups_for(manifest,cfg):
    for dataset in ("locomo","longmemeval"):
        ids={r["id"] for r in manifest["records"] if r["dataset"]==dataset}; groups=defaultdict(list)
        for case in cases_for(dataset,cfg):
            if case.id in ids: groups[case.cluster].append(case)
        for cluster,queries in sorted(groups.items()): yield dataset,cluster,queries

def summarize(output,manifest,methods):
    out=Path(output); rows=list(unique_rows(out/"retrieval.jsonl").values()); report=[]
    for dataset in ("locomo","longmemeval"):
        for method in methods:
            selected=[r for r in rows if r["dataset"]==dataset and r["method"]==method]
            if not selected: continue
            row={"dataset":dataset,"method":method,"questions":len(selected),"expected":50,
                 "status":"complete_development" if len(selected)==50 else "partial_development"}
            for metric in METRICS: row[metric]=float(np.mean([r["metrics"][metric] for r in selected]))
            row["mean_evidence_tokens"]=float(np.mean([r["evidence_tokens"] for r in selected]))
            report.append(row)
    if report: write_csv(out/"retrieval.csv",report)
    write_json(out/"status.json",{"status":"complete_development" if len(rows)==100*len(methods) else "partial_development",
        "completed_cells":len(rows),"expected_cells":100*len(methods),"unique_questions":len({(r["dataset"],r["question_id"]) for r in rows}),
        "qa_calls":0,"judge_calls":0,"full_runs_authorized":False,
        "claim_scope":"descriptive repeated-look development retrieval; not confirmation of answer superiority or novelty"})
    return report

def quality(manifest_path,calibration_path,output,methods=ARMS):
    cfg,manifest,calibration=protocol(manifest_path,calibration_path)
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    # OS lock, released on crash. Resumption never overwrites completed cells.
    import fcntl
    with (out/"worker.lock").open("a+b") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(out/"audit.json",identity(cfg,manifest_path,calibration_path,"quality_only_not_latency",methods),immutable=True)
        record_environment(out); done=unique_rows(out/"retrieval.jsonl")
        expected={f"{r['dataset']}/{m}/{r['id']}" for r in manifest["records"] for m in methods}
        if set(done)-expected: raise ValueError("Unexpected development cells")
        encoder,reranker=local_stack(cfg,cache=False); encoder=CachedEncoder(encoder,enabled=True)
        tokenizer=load_tokenizer(read_json(ROOT/"configs/checkpoints.lock.json")[cfg["evidence_tokenizer"]])
        for dataset,cluster,queries in groups_for(manifest,cfg):
            for arm in methods:
                pending=[c for c in queries if f"{dataset}/{arm}/{c.id}" not in done]
                if not pending: continue
                print(f"Build {dataset}/{cluster}/{arm}: {len(pending)} queries",flush=True)
                memory=build(arm,queries[0].events,cfg,calibration["selected_parameters"],encoder,reranker)
                if hasattr(memory,"validate_invariants"):
                    write_json(out/"structure"/f"{dataset}-{cluster}-{arm}.json",{
                        "invariants":memory.validate_invariants(),"telemetry":memory.telemetry()})
                index=EvidenceCache(queries[0].events,encoder,tokenizer,cfg)
                for case in pending:
                    hits=get_hits(arm,memory,case.question,cfg); selected,packet=index.packet(hits,case.question)
                    row={"id":f"{dataset}/{arm}/{case.id}","dataset":dataset,"question_id":case.id,"method":arm,
                        "cluster":cluster,"category":case.category,"metrics":rank_metrics(selected,case,8),
                        "selected":selected,"evidence":packet,"evidence_tokens":len(tokenizer.encode(packet,add_special_tokens=False)),
                        "status":"development_not_paper"}
                    if arm in ("joint","joint_preselected"):
                        unit={e.event_id:e.session_id if case.retrieval_unit=="session" else e.event_id for e in case.events}
                        candidates={unit[memory._episode_order[i]] for i in memory.last_candidate_pool}
                        row["candidate_count"]=len(memory.last_candidate_pool)
                        row["candidate_pool_recall@48"]=len(candidates&set(case.gold))/len(set(case.gold))
                    append_row(out/"retrieval.jsonl",row); done[row["id"]]=row
                    summarize(out,manifest,methods)
                    print(f"{len(done)}/{len(expected)} completed",flush=True)
                encoder.cache.clear(); del memory,index
        return summarize(out,manifest,methods)

def latency_worker(manifest_path,calibration_path,output,methods,repetition,answers=False):
    cfg,manifest,calibration=protocol(manifest_path,calibration_path); out=Path(output)
    if out.exists() and any(out.iterdir()): raise ValueError("Latency requires fresh repetition outputs, never resumed/cache-hit rows")
    out.mkdir(parents=True,exist_ok=True)
    audit=identity(cfg,manifest_path,calibration_path,"fresh_process_latency_with_answers" if answers else "fresh_process_retrieval_latency",methods)
    audit.update(repetition=repetition,cache_policy="encoder_score_packet_and_completion_caches_disabled",
                 warmup="one discarded neutral query per history/method; one discarded answer request per repetition if enabled",
                 latency_scope="retrieval+evidence; E2E additionally prompt preparation and actual serialized answer request; excludes ingestion/judge/startup",
                 answer_prompt_hash=ANSWER_PROMPT_HASH,qa_metric="LoCoMo native F1; LME token F1 is secondary, not native judge accuracy")
    write_json(out/"audit.json",audit,immutable=True); record_environment(out)
    encoder,reranker=local_stack(cfg,cache=False)
    tokenizer=load_tokenizer(read_json(ROOT/"configs/checkpoints.lock.json")[cfg["evidence_tokenizer"]])
    client=server_client(cfg["primary_generator"],cfg) if answers else None
    if client: client.complete(messages("What information is available?","No conversation evidence."),max_tokens=512,seed=cfg["seed"])
    rng=np.random.default_rng(cfg["seed"]+repetition)
    for dataset,cluster,queries in groups_for(manifest,cfg):
        for arm in rng.permutation(methods).tolist():
            memory=build(arm,queries[0].events,cfg,calibration["selected_parameters"],encoder,reranker)
            # Discarded warm-up; packet caches are fresh on every measured query.
            get_hits(arm,memory,"Conversation facts and events",cfg)
            for case in queries:
                started=perf_counter(); hits=get_hits(arm,memory,case.question,cfg)
                selected,packet=EvidenceCache(case.events,encoder,tokenizer,cfg).packet(hits,case.question)
                retrieved=perf_counter(); completion=None; native=None; e2e=None
                if client:
                    prompt=messages(case.question,packet,case.question_date)
                    if prompt_tokens(tokenizer,prompt,thinking=False)+cfg["generation"]["max_output_tokens"]>16384:
                        raise ValueError("Bounded development server context exceeded; no silent truncation")
                    completion=client.complete(prompt,max_tokens=cfg["generation"]["max_output_tokens"],seed=cfg["seed"])
                    e2e=perf_counter()-started
                    if completion.cache_hit: raise ValueError("Cached answer is not an efficiency observation")
                    if dataset=="locomo":
                        category_id=next(int(k) for k,v in CATEGORY_NAMES.items() if v==case.category)
                        native=locomo_native_score(completion.text,case.answer,category_id)
                row={"id":f"{dataset}/{arm}/{case.id}","dataset":dataset,"question_id":case.id,"method":arm,
                    "category":case.category,"repetition":repetition,"retrieval_seconds":retrieved-started,
                    "end_to_end_seconds":e2e,"cache_hit":False,"retrieval":rank_metrics(selected,case,8),
                    "native_f1":native,"token_f1":token_f1(completion.text,case.answer) if completion else None,
                    "prediction":completion.text if completion else None,
                    "input_tokens":completion.input_tokens if completion else 0,"output_tokens":completion.output_tokens if completion else 0,
                    "answer_request_id":completion.request_id if completion else None,"status":"development_latency_not_paper"}
                append_row(out/"queries.jsonl",row)
                print(f"Latency rep {repetition}: {dataset}/{arm}/{case.id}",flush=True)
            del memory
    return out

def latency(manifest_path,calibration_path,output,methods=("joint","compact","lightmem"),repetitions=3,answers=False):
    if repetitions<3: raise ValueError("At least three fresh process repetitions")
    out=Path(output)
    if out.exists() and any(out.iterdir()): raise ValueError("Use a fresh latency directory, do not time resumed artifacts")
    report=[]; all_rows=[]; hardware=None
    for rep in range(repetitions):
        folder=out/f"rep-{rep}"
        cmd=[sys.executable,str(ROOT/"scripts/iterate_joint.py"),"latency-worker","--manifest",str(manifest_path),
             "--calibration",str(calibration_path),"--output",str(folder),"--repetition",str(rep),"--methods",*methods]
        if answers: cmd.append("--answers")
        subprocess.run(cmd,cwd=ROOT,check=True)
        env=read_json(folder/"environment.json")
        signature={k:env[k] for k in ("hostname","cpu_affinity","dependencies")}
        signature["threads"]={k:env["allocation"].get(k) for k in ("SLURM_CPUS_PER_TASK","OMP_NUM_THREADS","MKL_NUM_THREADS")}
        if hardware is None: hardware=signature
        elif hardware!=signature: raise ValueError("Fresh repetitions differ in hardware/software/thread settings")
        rows=list(unique_rows(folder/"queries.jsonl").values()); all_rows.extend(rows)
        for dataset in ("locomo","longmemeval"):
            for method in methods:
                subset=[r for r in rows if r["dataset"]==dataset and r["method"]==method]
                if len(subset)!=50: raise ValueError("Incomplete latency panel")
                report.append({"dataset":dataset,"method":method,"repetition":rep,**latency_summary(subset)})
        write_csv(out/"efficiency_repetitions.csv",report)
    aggregate=[]
    for dataset in ("locomo","longmemeval"):
        for method in methods:
            subset=[r for r in all_rows if r["dataset"]==dataset and r["method"]==method]
            aggregate.append({"dataset":dataset,"method":method,"repetitions":repetitions,
                "scope":"development100_same_node_fresh_processes_serialized_queries",
                **latency_summary(subset),"total_answer_tokens":sum(r["input_tokens"]+r["output_tokens"] for r in subset),
                "mean_native_f1":float(np.mean([r["native_f1"] for r in subset])) if dataset=="locomo" and answers else None})
    write_csv(out/"efficiency.csv",aggregate); write_json(out/"hardware.json",hardware)
    write_json(out/"status.json",{"status":"complete_development_latency","answers":answers,"judge_calls":0,"full_runs_authorized":False})
    return aggregate
