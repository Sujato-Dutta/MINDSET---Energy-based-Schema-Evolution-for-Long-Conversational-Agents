from __future__ import annotations
import ast
import csv
import json
import time
import subprocess
import platform
import sys
import urllib.request
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import numpy as np
import yaml
from mindset.acl.io import read_json, write_json, digest_file, unique_rows, append_row
from mindset.acl.data import manifest, verified_cases
from mindset.acl.runtime import CachedEncoder, ResourceMonitor, make_method, snapshot, distribution
from mindset.acl.clients import LocalClient, load_tokenizer, messages, prompt_tokens, ANSWER_PROMPT_HASH
from mindset.evaluation.metrics import token_f1, exact_match
from mindset.evaluation.locomo import locomo_native_score
from mindset.datasets.locomo import CATEGORY_NAMES
from mindset.utils.hashing import stable_hash

ROOT = Path(__file__).resolve().parents[3]

def config(path="configs/acl.yaml"):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))

def source_fingerprint():
    paths=list((ROOT/"src").rglob("*.py"))+list((ROOT/"configs").glob("*.yaml"))
    return stable_hash({p.relative_to(ROOT).as_posix():digest_file(p) for p in sorted(paths)})

def record_environment(target):
    target=Path(target)
    if (target/"environment.json").exists(): return
    resolved=subprocess.run([sys.executable,"-m","pip","freeze"],capture_output=True,text=True,check=True)
    import os
    write_json(target/"environment.json",{
        "python":sys.version,"platform":platform.platform(),"machine":platform.machine(),
        "hostname":platform.node(),"processor":platform.processor(),
        "logical_cpus":os.cpu_count(),"cpu_affinity":sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
        "allocation":{k:os.getenv(k) for k in ("SLURM_JOB_ID","SLURM_CPUS_PER_TASK","SLURM_MEM_PER_NODE",
            "SLURM_JOB_GPUS","CUDA_VISIBLE_DEVICES","OMP_NUM_THREADS","MKL_NUM_THREADS")},
        "dependencies":resolved.stdout.splitlines()})

def request_json(url):
    req=urllib.request.Request(url,headers={"User-Agent":"MINDSET-reproducibility"})
    with urllib.request.urlopen(req,timeout=60) as response: return json.load(response)

def freeze(resolve_checkpoints=True):
    cfg=config()
    if cfg.get("judge",{}).get("locomo_protocol") in {"published_semantic_reference","lightmem_published"}:
        from mindset.acl.evaluation_protocol import pin_evaluation
        pin_evaluation()
    for name,spec in cfg["benchmarks"].items():
        value=manifest(name,ROOT/spec["path"],spec["expected_count"],categories=spec.get("include_categories"))
        write_json(ROOT/spec["manifest"],value,immutable=True)
        if name=="longmemeval":
            from mindset.acl.data import longmemeval_session_integrity
            write_json(ROOT/"data/manifests/longmemeval_session_integrity.json",
                       longmemeval_session_integrity(ROOT/spec["path"]),immutable=True)
        print(f"Frozen {name}: {value['count']} questions")
    if resolve_checkpoints:
        models=yaml.safe_load((ROOT/"configs/models.yaml").read_text())["models"]
        ids={k:v["checkpoint"] for k,v in models.items()}
        ids.update({"encoder":cfg["retrieval"]["encoder"],"reranker":cfg["retrieval"]["reranker"]})
        lock_path=ROOT/"configs/checkpoints.lock.json"
        if lock_path.exists():
            # A published lock is authoritative: never resolve moving HEADs on resume.
            lock=read_json(lock_path)
            for key,repo in ids.items():
                if lock.get(key,{}).get("checkpoint")!=repo:
                    raise ValueError("Checkpoint config differs from frozen lock; use a new protocol version")
            judge=lock["longmemeval_judge"]
            reference=ROOT/"external/reference/longmemeval_evaluate_qa.py"
            if not reference.exists():
                with urllib.request.urlopen(judge["url"],timeout=60) as response: source=response.read().decode()
                reference.parent.mkdir(parents=True,exist_ok=True)
                reference.write_text(source,encoding="utf-8")
            if digest_file(reference)!=judge["sha256"]:
                raise ValueError("Official judging source differs from frozen hash")
            license_path=reference.parent/"LICENSE.LongMemEval"
            if not license_path.exists():
                url=f"https://raw.githubusercontent.com/xiaowu0162/LongMemEval/{judge['commit']}/LICENSE"
                with urllib.request.urlopen(url,timeout=60) as response: license_text=response.read().decode()
                license_path.write_text(license_text,encoding="utf-8")
            return
        lock={}
        for key,repo in ids.items():
            data=request_json(f"https://huggingface.co/api/models/{repo}")
            lock[key]={"checkpoint":repo,"revision":data["sha"]}
        commit=request_json("https://api.github.com/repos/xiaowu0162/LongMemEval/commits/main")["sha"]
        url=f"https://raw.githubusercontent.com/xiaowu0162/LongMemEval/{commit}/src/evaluation/evaluate_qa.py"
        with urllib.request.urlopen(url,timeout=60) as response: source=response.read().decode()
        reference=ROOT/"external/reference/longmemeval_evaluate_qa.py"
        if reference.exists() and reference.read_text(encoding="utf-8")!=source:
            raise ValueError("Official judge source changed; use a new protocol version")
        reference.parent.mkdir(parents=True,exist_ok=True)
        reference.write_text(source,encoding="utf-8")
        license_url=f"https://raw.githubusercontent.com/xiaowu0162/LongMemEval/{commit}/LICENSE"
        with urllib.request.urlopen(license_url,timeout=60) as response: license_text=response.read().decode()
        (reference.parent/"LICENSE.LongMemEval").write_text(license_text,encoding="utf-8")
        lock["longmemeval_judge"]={"commit":commit,"sha256":digest_file(reference),"url":url}
        write_json(ROOT/"configs/checkpoints.lock.json",lock,immutable=True)

def local_stack(cfg, cache=True):
    from sentence_transformers import SentenceTransformer, CrossEncoder
    lock=read_json(ROOT/"configs/checkpoints.lock.json"); opts=cfg["retrieval"]
    enc=SentenceTransformer(lock["encoder"]["checkpoint"],revision=lock["encoder"]["revision"],device=opts["device"])
    rer=CrossEncoder(lock["reranker"]["checkpoint"],revision=lock["reranker"]["revision"],device=opts["device"])
    if cache:
        from mindset.methods.rerank_cache import CachedReranker
        rer=CachedReranker(rer,lock["reranker"]["checkpoint"]+"@"+lock["reranker"]["revision"],
                          ROOT/"outputs/acl/rerank_cache.sqlite")
    return CachedEncoder(enc,enabled=cache),rer

def server_client(model_key,cfg,cache_path=None):
    import os
    modelcfg=yaml.safe_load((ROOT/"configs/models.yaml").read_text())
    lock=read_json(ROOT/"configs/checkpoints.lock.json")
    item=modelcfg["models"][model_key]
    options={"chat_template_kwargs":{"enable_thinking":False}} if "enable_thinking" in item else {}
    url=os.getenv("MINDSET_LOCAL_BASE_URL",modelcfg["server"]["base_url"])
    return LocalClient(url,lock[model_key]["checkpoint"],options=options,
                       timeout=modelcfg["server"]["timeout_seconds"],cache_path=cache_path,
                       revision=lock[model_key]["revision"])

def cases_for(dataset,cfg):
    spec=cfg["benchmarks"][dataset]
    return verified_cases(dataset,ROOT/spec["path"],ROOT/spec["manifest"])

def seal_preparation(target,expected,count):
    ingestion_rows=[read_json(p) for p in sorted((target/"ingestion").glob("*.json"))]
    destination=target/"ingestion.jsonl"; temporary=destination.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in ingestion_rows),encoding="utf-8")
    temporary.replace(destination)
    write_json(target/"status.json",{"status":"complete" if count==expected else "smoke",
                                    "count":count,"expected":expected})

def packet(hits,events,query,encoder,tokenizer,cfg,cache=None):
    from mindset.acl.evidence import EvidenceCache
    index=cache if cache is not None else EvidenceCache(events,encoder,tokenizer,cfg)
    return index.packet(hits,query)

def rank_metrics(selected,case,k=8):
    if not case.gold: return None
    field="session_id" if case.retrieval_unit=="session" else "event_id"
    ranked=list(dict.fromkeys(c[field] for c in selected))[:k]; gold=set(case.gold)
    hits=[int(i in gold) for i in ranked]
    dcg=sum(x/np.log2(i+2) for i,x in enumerate(hits))
    ideal=sum(1/np.log2(i+2) for i in range(min(k,len(gold))))
    return {f"recall@{k}":sum(hits)/len(gold),f"precision@{k}":sum(hits)/k,
            f"mrr@{k}":next((1/(i+1) for i,x in enumerate(hits) if x),0),
            f"ndcg@{k}":float(dcg/ideal) if ideal else 0}

def retrieval_request_k(method,cfg):
    """Returned-hit budget, distinct from the method's internal scored pool."""
    return cfg["retrieval"].get("lightmem_hits",8) if method=="lightmem" else cfg["retrieval"]["candidate_k"]

def prepare(dataset,method,output,controller="energy",learned_path=None,limit=None,overrides=None,cache=True):
    cfg=config()
    if method=="mindset" and cfg.get("calibration"):
        from mindset.acl.calibration import validate_calibration
        validate_calibration(cfg,ROOT)
    if overrides: cfg["mindset"].update(overrides)
    if method not in cfg["methods"]: raise ValueError("Method outside frozen study")
    if method!="mindset" and controller!="energy": raise ValueError("Controllers apply only to MINDSET")
    cases=cases_for(dataset,cfg); expected=len(cases)
    if limit: cases=cases[:limit]
    target=Path(output); lock=read_json(ROOT/"configs/checkpoints.lock.json")
    identity={"dataset":dataset,"method":method,"controller":controller,"protocol":cfg,
              "checkpoints":lock,"source_fingerprint":source_fingerprint(),
              "manifest_sha256":digest_file(ROOT/cfg["benchmarks"][dataset]["manifest"]),
              "learned_sha256":digest_file(learned_path) if learned_path else None,
              "calibration_sha256":digest_file(ROOT/cfg["calibration"]["artifact"])
                  if cfg.get("calibration") and (ROOT/cfg["calibration"]["artifact"]).exists() else None,
              "limit":limit,"encoder_cache":cache}
    write_json(target/"run_config.json",identity,immutable=True)
    record_environment(target)
    done=unique_rows(target/"evidence.jsonl")
    if set(done)-{c.id for c in cases}: raise ValueError("Unexpected resumed IDs")
    if len(done)==len(cases):
        seal_preparation(target,expected,len(done))
        return target
    encoder,reranker=local_stack(cfg,cache=cache)
    tokenizer=load_tokenizer(lock[cfg["evidence_tokenizer"]])
    client=server_client(cfg["construction_generator"],cfg,
        ROOT/"outputs/acl/construction_cache.sqlite" if cache else None) if method=="zep" else None
    groups=defaultdict(list)
    for case in cases: groups[case.cluster].append(case)
    for cluster,group in groups.items():
        pending=[c for c in group if c.id not in done]
        if not pending: continue
        events=group[0].events
        from mindset.acl.evidence import EvidenceCache
        evidence_index=EvidenceCache(events,encoder,tokenizer,cfg)
        with ResourceMonitor() as monitor:
            started=time.perf_counter()
            memory=make_method(method,events,cfg,encoder,reranker,client,controller,learned_path)
            ingest_seconds=time.perf_counter()-started
            snap=target/"memory"/stable_hash(cluster)[:20]/"snapshot.json"
            serialization_started=time.perf_counter()
            snapshot(memory,events,snap)
            write_json(snap.with_name("raw_events.json"),[asdict(e) for e in events])
            serialization_seconds=time.perf_counter()-serialization_started
            ingestion_path=target/"ingestion"/(stable_hash(cluster)[:20]+".json")
            ingestion_record={"id":cluster,"episodes":len(events),"ingestion_seconds":ingest_seconds,
                "seconds_per_episode":ingest_seconds/max(1,len(events)),"serialization_seconds":serialization_seconds,
                "serialized_memory_bytes":sum(p.stat().st_size for p in snap.parent.iterdir() if p.is_file()),
                **monitor.report(),"telemetry":memory.telemetry()
                if memory is not None and hasattr(memory,"telemetry") else {}}
            write_json(ingestion_path,ingestion_record)
            if memory is not None and cfg["efficiency"]["warmup_queries"]:
                memory.retrieve(group[0].question,retrieval_request_k(method,cfg))
            for case in pending:
                started=time.perf_counter()
                native_seconds=0
                if memory is None:
                    selected=[{"event_id":e.event_id,"session_id":e.session_id,"text":
                        f"[{e.timestamp}] | Speaker: {e.speaker}\n{e.text}"} for e in events]
                    text="\n\n".join(f"[{i}] {e['text']}" for i,e in enumerate(selected,1))
                else:
                    hits=memory.retrieve(case.question,retrieval_request_k(method,cfg))
                    native_seconds=time.perf_counter()-started
                    selected,text=packet(hits,events,case.question,encoder,tokenizer,cfg,cache=evidence_index)
                elapsed=time.perf_counter()-started
                ranks=None if memory is None else rank_metrics(selected,case,cfg["retrieval"]["top_k"])
                append_row(target/"evidence.jsonl",{
                    "id":case.id,"cluster":case.cluster,"category":case.category,"question":case.question,
                    "answer":case.answer,"question_date":case.question_date,"evidence":text,
                    "selected":selected,"evidence_sha256":stable_hash(text),"retrieval":ranks,
                    "history_sessions":len({e.session_id for e in events}),"history_episodes":len(events),
                    "history_bytes":sum(len(e.text.encode()) for e in events),
                    "retrieval_seconds":elapsed,"retrieval_unit":case.retrieval_unit,
                    "profile_native_retrieval_seconds":native_seconds,
                    "profile_shared_evidence_seconds":elapsed-native_seconds})
                print(f"{method}/{controller} {case.id}",flush=True)
        evidence_index.snapshot(snap.parent)
        ingestion_record["serialized_memory_bytes"]=sum(p.stat().st_size for p in snap.parent.iterdir() if p.is_file())
        ingestion_record["shared_tokenization_calls"]=evidence_index.tokenization_calls
        ingestion_record["shared_chunk_embedding_calls"]=evidence_index.chunk_embedding_calls
        ingestion_record.update(monitor.report())
        write_json(ingestion_path,ingestion_record)
    seal_preparation(target,expected,len(unique_rows(target/"evidence.jsonl")))
    return target

def check_prepared_protocol(cfg,prepared):
    """Permit only prespecified ablations, never test-tuned scalar overrides."""
    from copy import deepcopy
    actual=deepcopy(prepared); expected=deepcopy(cfg)
    ablation=actual.get("mindset",{}).pop("ablation","full")
    expected.get("mindset",{}).pop("ablation",None)
    if ablation!="full" and ablation not in cfg.get("ablations",[]):
        raise ValueError("Unregistered MINDSET ablation")
    if actual!=expected:
        raise ValueError("Protocol changed after memory freeze")
    return ablation


def answer(prepared,model_key,output):
    cfg=config()
    if cfg.get("status","frozen")!="frozen":
        raise ValueError("Paper answer generation requires a frozen calibrated protocol")
    prepared=Path(prepared); target=Path(output)
    frozen=read_json(prepared/"run_config.json")
    ablation=check_prepared_protocol(cfg,frozen["protocol"])
    if ablation!="full" and (frozen["method"]!="mindset" or model_key!=cfg["primary_generator"]):
        raise ValueError("Prespecified ablations require MINDSET and the primary answer model")
    if frozen.get("source_fingerprint") and frozen["source_fingerprint"]!=source_fingerprint():
        raise ValueError("Source code changed after memory freeze")
    lock=read_json(ROOT/"configs/checkpoints.lock.json")
    if lock!=frozen["checkpoints"]: raise ValueError("Checkpoint lock changed")
    rows=unique_rows(prepared/"evidence.jsonl")
    manifest_rows=read_json(ROOT/cfg["benchmarks"][frozen["dataset"]]["manifest"])["records"]
    if set(rows)!={r["id"] for r in manifest_rows}:
        raise ValueError("Full benchmark evidence required for paper generation")
    settings=yaml.safe_load((ROOT/"configs/models.yaml").read_text())["models"][model_key]
    tokenizer=load_tokenizer(lock[model_key]); client=server_client(model_key,cfg)
    write_json(target/"run_config.json",{"prepared_config":frozen,"model":model_key,
        "checkpoint":lock[model_key],"prompt_hash":ANSWER_PROMPT_HASH,"generation":cfg["generation"],
        "prepared_evidence_sha256":digest_file(prepared/"evidence.jsonl"),
        "prepared_path":str(prepared.resolve())},immutable=True)
    record_environment(target)
    done=unique_rows(target/"predictions.jsonl")
    if set(done)-set(rows): raise ValueError("Unexpected resumed prediction IDs")
    for row in rows.values():
        if row["id"] in done: continue
        evidence=row["evidence"]; selected=list(row["selected"]); omitted=0
        if frozen["method"]=="long_context":
            from mindset.baselines.long_context import fit_long_context
            selected,evidence,omitted=fit_long_context(row["question"],row["question_date"],selected,
                tokenizer,cfg["generation"]["common_context_tokens"],
                cfg["generation"]["max_output_tokens"],settings.get("enable_thinking"))
        msgs=messages(row["question"],evidence,row["question_date"])
        n=prompt_tokens(tokenizer,msgs,thinking=settings.get("enable_thinking"))
        if n+cfg["generation"]["max_output_tokens"]>cfg["generation"]["common_context_tokens"]:
            raise ValueError("Bounded retrieval packet exceeds common context")
        result=client.complete(msgs,cfg["generation"]["max_output_tokens"],cfg["seed"])
        category_id=next((int(k) for k,v in CATEGORY_NAMES.items() if v==row["category"]),None)
        append_row(target/"predictions.jsonl",{**row,"prediction":result.text,
            "selected":selected,"evidence":evidence,"evidence_sha256":stable_hash(evidence),
            "omitted_oldest_events":omitted,"model":model_key,
            "token_f1":token_f1(result.text,row["answer"]),
            "native_f1":locomo_native_score(result.text,row["answer"],category_id)
                if frozen["dataset"]=="locomo" else None,
            "exact_match":exact_match(result.text,row["answer"]),"generation":asdict(result)})
        print(f"{model_key} answer {row['id']}",flush=True)
    write_json(target/"status.json",{"status":"answered","count":len(rows)})

def official_lme_prompt():
    lock=read_json(ROOT/"configs/checkpoints.lock.json")["longmemeval_judge"]
    p=ROOT/"external/reference/longmemeval_evaluate_qa.py"
    if digest_file(p)!=lock["sha256"]: raise ValueError("Official judge source hash mismatch")
    tree=ast.parse(p.read_text(encoding="utf-8"))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="get_anscheck_prompt")
    namespace={}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),"exec"),namespace)
    return namespace["get_anscheck_prompt"]

def judge(run):
    from mindset.api.backend import APIBackend
    from mindset.api.models import CompletionRequest
    cfg=config(); target=Path(run); identity=read_json(target/"run_config.json")
    write_json(target/"judge_config.json",{"judge":cfg["judge"],
        "source_sha256":digest_file(ROOT/"configs/judge.yaml"),
        "evaluation_lock":read_json(ROOT/"configs/evaluation.lock.json") if (ROOT/"configs/evaluation.lock.json").exists() else None,
        "protocol":"official_LongMemEval_prompt_Gemini_substitution; external_published_LoCoMo_semantic_reference_not_ACL_LightMem"},
        immutable=True)
    dataset=identity["prepared_config"]["dataset"]
    upstream=official_lme_prompt() if dataset=="longmemeval" else None
    backend=APIBackend.from_file(ROOT/"configs/judge.yaml")
    done=unique_rows(target/"judgements.jsonl")
    versions={r.get("provider_model_version") for r in done.values() if r.get("provider_model_version")}
    for row in unique_rows(target/"predictions.jsonl").values():
        if row["id"] in done: continue
        if upstream:
            prompt=upstream(row["category"],row["question"],row["answer"],row["prediction"],
                            abstention="_abs" in row["id"])
        else:
            if cfg.get("judge",{}).get("locomo_protocol") in {"published_semantic_reference","lightmem_published"}:
                from mindset.acl.evaluation_protocol import published_locomo_prompt
                prompt=published_locomo_prompt(row["question"],str(row["answer"]),row["prediction"])
            else:
                prompt=("Judge whether the candidate answer correctly and completely answers the question, "
                    "accepting equivalent paraphrases. For unanswerable questions require abstention. "
                    "Return yes or no only.\nQuestion: "+row["question"]+"\nReference: "+
                    str(row["answer"])+"\nCandidate: "+row["prediction"])
        published=dataset=="locomo" and cfg.get("judge",{}).get("locomo_protocol") in {"published_semantic_reference","lightmem_published"}
        result=backend.complete(CompletionRequest(model_name="gemini_judge",system_prompt="",
            user_prompt=prompt,dataset_example_id=stable_hash((row["id"],prompt)),method="blind_judge",
            code_version=cfg["judge"]["prompt_version"],temperature=0,max_output_tokens=128 if published else 10),max_retries=2)
        raw=result.text.strip().lower().rstrip(".")
        if published:
            from mindset.acl.evaluation_protocol import parse_published_label
            label=parse_published_label(result.text)
        else:
            if raw not in {"yes","no"}: raise ValueError(f"Malformed judge answer: {result.text!r}")
            label=raw=="yes"
        if result.provider_model_version:
            versions.add(result.provider_model_version)
            if len(versions)>1: raise ValueError("Judge underlying model version changed; evaluation stopped")
        append_row(target/"judgements.jsonl",{"id":row["id"],"label":label,
            "judge_model":result.model_id,"prompt_sha256":stable_hash(prompt),"raw":result.text,
            "provider_model_version":result.provider_model_version,
            "input_tokens":result.usage.input_tokens,"output_tokens":result.usage.output_tokens,
            "cache_hit":result.cache_hit,"latency_seconds":result.latency_seconds})
        print(f"Judged {row['id']}",flush=True)
    write_json(target/"status.json",{"status":"judged","count":len(unique_rows(target/"judgements.jsonl"))})

def write_csv(path,rows):
    if not rows: raise ValueError("No completed runs to export")
    keys=list(dict.fromkeys(k for r in rows for k in r))
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)


def complete_judge_labels(dataset,predictions,judged):
    if judged and set(judged)!=set(predictions):
        raise ValueError("Partial or misaligned judge labels cannot be used")
    if dataset=="longmemeval" and set(judged)!=set(predictions):
        raise ValueError("Official LongMemEval accuracy requires complete judge labels")
    return bool(judged)

def summarize(run_paths,output):
    from mindset.acl.analysis import paired_stats
    quality=[]; efficiency=[]; per_question=[]; categories=[]
    for path in run_paths:
        p=Path(path); identity=read_json(p/"run_config.json"); prep=identity["prepared_config"]
        predictions=unique_rows(p/"predictions.jsonl"); judged=unique_rows(p/"judgements.jsonl")
        expected=read_json(ROOT/prep["protocol"]["benchmarks"][prep["dataset"]]["manifest"])["records"]
        if set(predictions)!={x["id"] for x in expected}:
            raise ValueError(f"Incomplete or misaligned predictions/judgements: {p}")
        has_judge=complete_judge_labels(prep["dataset"],predictions,judged)
        base={"dataset":prep["dataset"],"method":prep["method"],"controller":prep["controller"],
              "ablation":prep["protocol"].get("mindset",{}).get("ablation","full") if prep["method"]=="mindset" else "not_applicable",
              "model":identity["model"],"question_count":len(predictions),"run_dir":str(p)}
        row={**base,"judge_count":len(judged),"judge_accuracy":float(np.mean([j["label"] for j in judged.values()])) if has_judge else None,
             "token_f1":float(np.mean([r["token_f1"] for r in predictions.values()]))}
        ids=list(predictions)
        if has_judge:
            labels=[int(judged[i]["label"]) for i in ids]
            interval=paired_stats(labels,[0]*len(labels),
                [predictions[i]["cluster"] for i in ids] if prep["dataset"]=="locomo" else None,
                resamples=prep["protocol"]["statistics"]["resamples"])
            row["accuracy_ci_low"]=interval["ci_low"]; row["accuracy_ci_high"]=interval["ci_high"]
        native=[r["native_f1"] for r in predictions.values() if r.get("native_f1") is not None]
        row["native_f1"]=float(np.mean(native)) if native else None
        row["exact_match"]=float(np.mean([r["exact_match"] for r in predictions.values()]))
        row["primary_metric"]=prep["protocol"].get("primary_metrics",{}).get(prep["dataset"],"judge_accuracy")
        retrieval=[r["retrieval"] for r in predictions.values() if r["retrieval"] is not None]
        row["retrieval_count"]=len(retrieval)
        for k in retrieval[0] if retrieval else []: row[k]=float(np.mean([r[k] for r in retrieval]))
        quality.append(row)
        grouped=defaultdict(list)
        for id,r in predictions.items(): grouped[r["category"]].append(id)
        for category,category_ids in grouped.items():
            categories.append({**base,"category":category,"category_count":len(category_ids),
                "judge_accuracy":float(np.mean([judged[i]["label"] for i in category_ids])) if has_judge else None,
                "native_f1":float(np.mean([predictions[i]["native_f1"] for i in category_ids])) if prep["dataset"]=="locomo" else None,
                "token_f1":float(np.mean([predictions[i]["token_f1"] for i in category_ids]))})
        for id,r in predictions.items():
            per_question.append({**base,"id":id,"cluster":r["cluster"],"category":r["category"],
                "judge_accuracy":int(judged[id]["label"]) if has_judge else None,
                "native_f1":r.get("native_f1"),"exact_match":r["exact_match"],"token_f1":r["token_f1"],**(r["retrieval"] or {}),
                "history_sessions":r["history_sessions"],"history_episodes":r["history_episodes"],
                "history_bytes":r["history_bytes"]})
        e={**base,"answer_calls":len(predictions),"judge_logical_calls":len(judged),
           "judge_uncached_calls":sum(not j["cache_hit"] for j in judged.values()),
           "answer_input_tokens":sum(r["generation"]["input_tokens"] for r in predictions.values()),
           "answer_output_tokens":sum(r["generation"]["output_tokens"] for r in predictions.values()),
           "judge_logical_tokens":sum(j["input_tokens"]+j["output_tokens"] for j in judged.values()),
           "judge_billed_tokens":sum(j["input_tokens"]+j["output_tokens"] for j in judged.values() if not j["cache_hit"]),
           "construction_model":prep["protocol"]["construction_generator"]}
        ingestion=list(unique_rows(Path(identity["prepared_path"])/"ingestion.jsonl").values())
        e["ingestion_episodes"]=sum(r["episodes"] for r in ingestion)
        e["weighted_ingestion_seconds_per_episode"]=sum(r["ingestion_seconds"] for r in ingestion)/max(1,e["ingestion_episodes"])
        e["serialized_memory_total_bytes"]=sum(r["serialized_memory_bytes"] for r in ingestion)
        e["peak_process_rss_bytes"]=max((r["peak_process_rss_bytes"] for r in ingestion),default=None)
        vram=[r["peak_device_vram_bytes"] for r in ingestion if r["peak_device_vram_bytes"] is not None]
        e["peak_device_vram_bytes"]=max(vram) if vram else None
        e["construction_uncached_local_llm_calls"]=sum(r["telemetry"].get("write_local_llm_calls",0) for r in ingestion)
        e["construction_input_tokens"]=sum(r["telemetry"].get("construction_input_tokens",0) for r in ingestion)
        e["construction_output_tokens"]=sum(r["telemetry"].get("construction_output_tokens",0) for r in ingestion)
        for field,values in {"retrieval_seconds":[r["retrieval_seconds"] for r in predictions.values()],
             "answer_latency_seconds":[r["generation"]["latency_seconds"] for r in predictions.values()],
             "evidence_bytes":[len(r["evidence"].encode()) for r in predictions.values()]}.items():
            for stat,value in distribution(values).items(): e[field+"_"+stat]=value
        efficiency.append(e)
    out=Path(output)
    write_csv(out/"results.csv",quality); write_csv(out/"efficiency.csv",efficiency)
    write_csv(out/"per_question.csv",per_question)
    write_csv(out/"category_results.csv",categories)
    panels=defaultdict(list)
    for r in quality: panels[(r["dataset"],r["model"])].append(r)
    ranked=defaultdict(list)
    for (dataset,model),rows in panels.items():
        panel=tuple(sorted(r["method"]+":"+r["controller"]+":"+r["ablation"] for r in rows))
        for r in rows:
            metric=r["primary_metric"]
            if r[metric] is None: raise ValueError("Missing primary answer metric")
            rank=1+sum(x[metric]>r[metric] for x in rows)
            ranked[(dataset,panel,r["method"],r["controller"],r["ablation"])].append(rank)
    write_csv(out/"mean_ranks.csv",[{"dataset":k[0],"panel":"|".join(k[1]),"method":k[2],
        "controller":k[3],"ablation":k[4],"generator_count":len(v),"mean_rank":float(np.mean(v))}
        for k,v in ranked.items()])

