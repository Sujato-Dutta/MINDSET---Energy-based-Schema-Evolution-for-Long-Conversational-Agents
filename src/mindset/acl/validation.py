"""CPU-only diagnostic retrieval validation; never a paper QA result.

Question IDs are chosen before scoring and preserved verbatim. This command
does not tune parameters, generate answers, call a judge, or run a GPU model.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from dataclasses import asdict
from time import perf_counter
import numpy as np
from mindset.acl.pipeline import (ROOT,config,cases_for,local_stack,record_environment,
                                  source_fingerprint,rank_metrics,write_csv)
from mindset.acl.calibration import validate_calibration
from mindset.acl.clients import load_tokenizer
from mindset.acl.evidence import EvidenceCache
from mindset.acl.runtime import make_method
from mindset.acl.io import read_json,write_json,append_row,digest_file


def diagnostic_cases(cases,dataset,per_group):
    """Round-robin categories within each LoCoMo cluster; LME by category.

    Selection depends only on IDs, category and evidence annotation availability,
    never relevance scores, answers or a method's outputs.
    """
    if per_group<1: raise ValueError("Positive diagnostic group size required")
    groups=defaultdict(lambda:defaultdict(list))
    for case in cases:
        if case.gold:
            group=case.cluster if dataset=="locomo" else case.category
            groups[group][case.category].append(case)
    selected=[]
    for group in sorted(groups):
        queues={k:sorted(v,key=lambda c:c.id) for k,v in sorted(groups[group].items())}
        picked=[]
        while len(picked)<per_group and any(queues.values()):
            for queue in queues.values():
                if queue and len(picked)<per_group: picked.append(queue.pop(0))
        selected.extend(picked)
    return selected


def validate_mindset(output,locomo_per_cluster=2,lme_per_category=1,datasets=None,categories=None):
    cfg=config(); validate_calibration(cfg,ROOT)
    if cfg["retrieval"]["device"]!="cpu": raise ValueError("Diagnostic validation is CPU only")
    out=Path(output)
    if out.exists() and any(out.iterdir()): raise ValueError("Use a fresh validation output directory")
    out.mkdir(parents=True,exist_ok=True); record_environment(out)
    cohorts={d:diagnostic_cases(cases_for(d,cfg),d,n) for d,n in
             (("locomo",locomo_per_cluster),("longmemeval",lme_per_category))}
    if datasets: cohorts={d:c for d,c in cohorts.items() if d in datasets}
    if categories: cohorts={d:[c for c in cohort if c.category in categories] for d,cohort in cohorts.items()}
    if not cohorts or any(not cohort for cohort in cohorts.values()):
        raise ValueError("Diagnostic dataset/category selection is empty")
    audit={"method":"mindset","implementation_version":"v2","status":"diagnostic_not_paper",
        "selection":"ID/category round robin; annotated evidence required; no score-based selection",
        "configuration":cfg,"source_fingerprint":source_fingerprint(),
        "calibration_sha256":digest_file(ROOT/cfg["calibration"]["artifact"]),
        "benchmarks":{d:{"manifest_sha256":digest_file(ROOT/cfg["benchmarks"][d]["manifest"]),
            "records":[{"id":c.id,"cluster":c.cluster,"category":c.category} for c in cohort]}
             for d,cohort in cohorts.items()},
        "lightmem_source_sha256":digest_file(ROOT/"src/mindset/baselines/lightmem.py"),
        "qa_generation_calls":0,"judge_calls":0,"gpu_calls":0,
        "limitations":"Small known-benchmark diagnostic, not unseen/OOD QA or a full efficiency experiment"}
    write_json(out/"audit.json",audit,immutable=True)
    # In-memory reuse accelerates quality diagnostics; these timings are NOT
    # valid paper efficiency measurements. No cross-encoder score cache is used.
    encoder,reranker=local_stack(cfg,cache=False)
    from mindset.acl.runtime import CachedEncoder
    encoder=CachedEncoder(encoder,enabled=True)
    lock=read_json(ROOT/"configs/checkpoints.lock.json")
    tokenizer=load_tokenizer(lock[cfg["evidence_tokenizer"]])
    rows=[]; structures=[]; batching=[]
    for dataset,cohort in cohorts.items():
        groups=defaultdict(list)
        for case in cohort: groups[case.cluster].append(case)
        for cluster,queries in groups.items():
            for method in ("mindset","vector_rag","lightmem","nemori"):
                started=perf_counter()
                memory=make_method(method,queries[0].events,cfg,encoder,reranker)
                if hasattr(memory,"validate_invariants"):
                    invariants=memory.validate_invariants()
                    structures.append({"dataset":dataset,"cluster":cluster,"method":method,
                                       "invariants":invariants,"telemetry":memory.telemetry()})
                index=EvidenceCache(queries[0].events,encoder,tokenizer,cfg)
                for case in queries:
                    budget=cfg["mindset"]["retrieval_candidate_k"] if method=="mindset" else cfg["retrieval"]["candidate_k"]
                    hits=memory.retrieve(case.question,budget)
                    selected,_=index.packet(hits,case.question)
                    metric=rank_metrics(selected,case,8)
                    row={"id":f"{dataset}/{method}/{case.id}","question_id":case.id,
                        "dataset":dataset,"method":method,"cluster":case.cluster,"category":case.category,
                        "selected":selected,"metrics":metric,"status":"diagnostic_not_paper"}
                    if method=="mindset":
                        # This is the union actually scored, not unscored histories.
                        candidates={h.memory_id for h in hits}
                        field="session_id" if case.retrieval_unit=="session" else "event_id"
                        event_map={e.event_id:e for e in case.events}
                        native={event_map[i].session_id if field=="session_id" else i for i in candidates}
                        row["candidate_pool_recall@48"]=len(native&set(case.gold))/len(set(case.gold))
                        row["candidate_count"]=len(candidates)
                        if not batching:
                            pairs=[(case.question,h.text) for h in hits]
                            reference=np.asarray(reranker.predict(pairs,batch_size=64,show_progress_bar=False))
                            optimized=np.asarray(reranker.predict(pairs,batch_size=8,show_progress_bar=False))
                            np.testing.assert_allclose(reference,optimized,rtol=1e-5,atol=1e-5)
                            same_order=bool(np.array_equal(np.argsort(-reference,kind="stable"),
                                                         np.argsort(-optimized,kind="stable")))
                            if not same_order: raise ValueError("CPU batching changed the candidate ranking")
                            batching.append({"pairs":len(pairs),"max_absolute_score_difference":
                                float(np.max(np.abs(reference-optimized))),"same_ranking":same_order})
                    append_row(out/"retrieval.jsonl",row); rows.append(row)
                print(f"CPU diagnostic: {dataset} {cluster} {method} {len(queries)} queries "
                      f"({perf_counter()-started:.1f}s; timings include reuse)",flush=True)
    summary=[]
    for dataset in cohorts:
        for method in ("mindset","vector_rag","lightmem","nemori"):
            subset=[r for r in rows if r["dataset"]==dataset and r["method"]==method]
            entry={"dataset":dataset,"method":method,"questions":len(subset),"status":"diagnostic_not_paper"}
            for name in ("recall@8","precision@8","mrr@8","ndcg@8"):
                entry[name]=float(np.mean([r["metrics"][name] for r in subset]))
            if method=="mindset":
                entry["candidate_pool_recall@48"]=float(np.mean([r["candidate_pool_recall@48"] for r in subset]))
            summary.append(entry)
    write_csv(out/"retrieval.csv",summary)
    write_json(out/"structural_validation.json",structures)
    write_json(out/"batching_validation.json",batching)
    if digest_file(ROOT/"src/mindset/baselines/lightmem.py")!=audit["lightmem_source_sha256"]:
        raise ValueError("LightMem baseline changed during diagnostic validation")
    write_json(out/"status.json",{"status":"complete_diagnostic","qa_calls":0,"judge_calls":0,
        "lightmem_unchanged":True,"skipped_methods":{"zep":"Requires the local construction LLM server",
        "long_context":"No native top-k retrieval; answer generation is not part of this CPU diagnostic"}})
    return summary
