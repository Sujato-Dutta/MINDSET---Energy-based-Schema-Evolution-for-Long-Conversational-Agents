"""Bounded, resumable 100-question development experiments, never paper tests.

No coefficient search or answer/gold access inside retrieval variants. The
production controller and its calibration remain unchanged until a candidate
passes independent evaluation. Standard RRF and log-volume diversity are
controls/building blocks, not claims of newly invented algorithms.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import numpy as np
from mindset.acl.io import read_json, write_json, digest_file, append_row, unique_rows
from mindset.acl.pipeline import ROOT, config, cases_for, local_stack, record_environment, source_fingerprint, rank_metrics, write_csv
from mindset.acl.calibration import validate_calibration
from mindset.acl.runtime import CachedEncoder, make_method
from mindset.acl.evidence import EvidenceCache
from mindset.acl.clients import load_tokenizer
from mindset.acl.pipeline import retrieval_request_k
from mindset.methods.base import MemoryHit
from mindset.memory.energy import unit_vector
from mindset.retrieval.hybrid import ranked_indices, reciprocal_rank_fusion

VARIANTS = ("current", "flat_hybrid48", "balanced_provenance", "provenance_logvolume")
BASELINES = ("vector_rag", "lightmem", "nemori")
METRICS = ("recall@8", "precision@8", "mrr@8", "ndcg@8")


def seeded_key(seed, value):
    return hashlib.sha256(f"{seed}/{value}".encode()).hexdigest()


def stratified_cases(cases, dataset, count=50, seed=0, excluded=()):
    """ID-hash sampling, equal cluster/category quotas, never method scores.

    Gold presence is used only to require evaluable retrieval annotations;
    annotation contents and answers never determine the sampled IDs.
    """
    groups=defaultdict(lambda:defaultdict(list))
    for c in cases:
        if c.gold and c.id not in excluded:
            groups[c.cluster if dataset=="locomo" else c.category][c.category].append(c)
    if count<=0 or count>sum(len(v) for g in groups.values() for v in g.values()):
        raise ValueError("Invalid development sample count")
    ordered=sorted(groups,key=lambda g:seeded_key(seed,g))
    queues={}
    for group in ordered:
        categories=sorted(groups[group],key=lambda t:seeded_key(seed,f"{group}/{t}"))
        buckets={t:sorted(groups[group][t],key=lambda c:seeded_key(seed,c.id)) for t in categories}
        queue=[]
        while any(buckets.values()):
            for t in categories:
                if buckets[t]: queue.append(buckets[t].pop(0))
        queues[group]=queue
    selected=[]
    while len(selected)<count:
        for group in ordered:
            if queues[group] and len(selected)<count: selected.append(queues[group].pop(0))
    return selected


def freeze_iteration(path):
    cfg=config(); validate_calibration(cfg,ROOT)
    records=[]; sources={}; excluded=defaultdict(set)
    # Avoid reusing the immediately preceding scored pilot IDs. Older study
    # exposure still exists: this does NOT turn either benchmark into unseen data.
    previous=(ROOT/"outputs/acl/mindset_v2_retrieval_verified/audit.json",
              ROOT/"outputs/acl/mindset_v2_integrity_recheck/audit.json")
    for p in previous:
        if p.exists():
            for d,b in read_json(p)["benchmarks"].items(): excluded[d].update(r["id"] for r in b["records"])
    for d in ("locomo","longmemeval"):
        full=ROOT/cfg["benchmarks"][d]["manifest"]
        native=read_json(full); lookup={r["id"]:r for r in native["records"]}
        cohort=stratified_cases(cases_for(d,cfg),d,50,cfg["seed"],excluded[d])
        records.extend({"dataset":d,**lookup[c.id]} for c in cohort)
        sources[d]={"manifest_path":full.relative_to(ROOT).as_posix(),"manifest_sha256":digest_file(full),
                    "source_path":cfg["benchmarks"][d]["path"],"source_sha256":native["source_sha256"]}
    value={"schema_version":"mindset-v2-development100-v1","count":100,"seed":cfg["seed"],
        "selection":"50 per benchmark; seeded ID hashes; balanced LoCoMo clusters and LME categories",
        "role":"development_diagnostic_not_confirmatory_test","prior_benchmark_exposure":True,
        "parameter_selection":"independent synthetic transition histories; never this cohort's scores",
        "calibration_sha256":digest_file(ROOT/cfg["calibration"]["artifact"]),
        "excluded_previous_pilot_ids":{d:sorted(v) for d,v in excluded.items()},
        "sources":sources,"records":records,"variants":list(VARIANTS),
        "qa_gate":"No automatic full runs; retrieval probes cannot establish answer superiority or oral acceptance"}
    write_json(path,value,immutable=True)
    return value


def verify_iteration(value,cfg):
    if value["count"]!=100 or len(value["records"])!=100:
        raise ValueError("Iteration must contain exactly 100 questions")
    identities=[(r["dataset"],r["id"]) for r in value["records"]]
    if len(set(identities))!=100: raise ValueError("Duplicate iteration question IDs")
    if Counter(d for d,_ in identities)!={"locomo":50,"longmemeval":50}:
        raise ValueError("Iteration requires 50 questions per benchmark")
    if value["calibration_sha256"]!=digest_file(ROOT/cfg["calibration"]["artifact"]):
        raise ValueError("Iteration calibration changed")
    for d,s in value["sources"].items():
        if s["manifest_path"]!=cfg["benchmarks"][d]["manifest"] or s["source_path"]!=cfg["benchmarks"][d]["path"]:
            raise ValueError("Iteration source paths changed")
        if digest_file(ROOT/s["manifest_path"])!=s["manifest_sha256"] or digest_file(ROOT/s["source_path"])!=s["source_sha256"]:
            raise ValueError("Iteration source checksum changed")
        lookup={r["id"]:r for r in read_json(ROOT/s["manifest_path"])["records"]}
        for r in value["records"]:
            if r["dataset"]==d and {k:v for k,v in r.items() if k!="dataset"}!=lookup.get(r["id"]):
                raise ValueError("Iteration question/history audit mismatch")


def round_robin_unique(queues, selected, limit):
    """Allocate the remaining unique budget fairly to provenance paths."""
    selected=list(dict.fromkeys(selected))[:limit]; seen=set(selected)
    queues=[iter(q) for q in queues]
    while queues and len(selected)<limit:
        active=[]
        for queue in queues:
            for candidate in queue:
                if candidate not in seen:
                    seen.add(candidate); selected.append(candidate); active.append(queue); break
            if len(selected)>=limit: break
        queues=active
    return selected


def candidate_variants(memory,query):
    """Construct equal-budget controls; no gold, answer or benchmark ID inputs.

    The balanced variant follows predecessor/successor version paths, allocates
    provenance round-robin, then fills unused slots with hybrid episodes. This
    addresses duplicate channel overlap without introducing feature weights.
    """
    q=unit_vector(memory._encode([query])[0]); n=len(memory._episode_order)
    dense=memory._episode_vectors@q; lexical=memory._bm25.scores(query)
    budget=memory.config.retrieval_candidate_k
    all_fused=reciprocal_rank_fusion([ranked_indices(dense),ranked_indices(lexical)])
    full=sorted(all_fused,key=lambda i:(-all_fused[i],i))
    partial=reciprocal_rank_fusion([ranked_indices(dense,memory.config.direct_pool_k),
                                  ranked_indices(lexical,memory.config.direct_pool_k)])
    direct=sorted(partial,key=lambda i:(-partial[i],i))[:memory.config.direct_pool_k]
    roots=[memory._schema_ids[i] for i in ranked_indices(memory._schema_matrix@q,memory.config.nearest_schema_k)]
    expanded={memory._index_by_id[e] for s in roots for e in memory._schema_provenance[s]}
    old_expanded=sorted(expanded,key=lambda i:(-float(dense[i]),i))[:memory.config.schema_pool_k]
    current=list(dict.fromkeys(direct+old_expanded))[:budget]
    # Only version links are traversed: unrelated same-speaker schemas do not
    # become evidence merely because they share an owner or discourse slot.
    versions=list(roots); seen=set(versions)
    for sid in versions:
        schema=memory.schemas[sid]
        links=[schema.predecessor_schema_id,*schema.successor_schema_ids]
        for link in links:
            if link and link not in seen: seen.add(link); versions.append(link)
    queues=[sorted((memory._index_by_id[e] for e in memory._schema_provenance[s]),
                   key=lambda i:(-all_fused[i],i)) for s in versions]
    balanced=round_robin_unique(queues,direct,budget)
    balanced=round_robin_unique([full],balanced,budget)
    return {"current":current,"flat_hybrid48":full[:min(budget,n)],
            "balanced_provenance":balanced,"provenance_logvolume":balanced}


def logvolume_order(scores,vectors,identifiers,top_k=8):
    """Greedy quality/diversity log determinant, with best CE result first.

    L = diag(exp(score/2)) V V^T diag(exp(score/2)). A candidate's
    marginal gain is score + log(squared orthogonal residual). No fitted
    trade-off coefficient. This is a standard DPP-style control, not new math.
    Remaining evidence is appended in CE order for bounded packet backfilling.
    """
    scores=np.asarray(scores,dtype=np.float64); vectors=np.asarray(vectors,dtype=np.float64)
    if not len(scores): return []
    if len(scores)!=len(vectors) or len(scores)!=len(identifiers) or not np.isfinite(scores).all() or not np.isfinite(vectors).all():
        raise ValueError("Malformed log-volume inputs")
    vectors=vectors/np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),np.finfo(float).eps)
    ce=sorted(range(len(scores)),key=lambda i:(-scores[i],identifiers[i]))
    chosen=[]; residual=vectors.copy(); remaining=set(ce)
    for step in range(min(top_k,len(ce))):
        norm2=np.sum(residual*residual,axis=1)
        best=ce[0] if step==0 else min(remaining,key=lambda i:(
            -(scores[i]+np.log(max(norm2[i],np.finfo(float).eps))),identifiers[i]))
        chosen.append(best); remaining.remove(best)
        if norm2[best]>np.finfo(float).eps:
            axis=residual[best]/np.sqrt(norm2[best]); residual-=np.outer(residual@axis,axis)
    return chosen+[i for i in ce if i in remaining]


def variant_hits(memory,query,pools):
    """Score the union once locally; reuse identical pair scores across arms.

    CPU timing/calls here are diagnostic and cannot enter an efficiency table.
    Every arm receives <=48 candidates and <=8 final raw episodes.
    """
    union=list(dict.fromkeys(i for pool in pools.values() for i in pool))
    values=np.asarray(memory.local_reranker.predict([(query,memory._documents[i]) for i in union],
        batch_size=memory.config.rerank_batch_size,show_progress_bar=False),dtype=np.float32)
    if len(values)!=len(union) or not np.isfinite(values).all(): raise ValueError("Invalid cross-encoder scores")
    score=dict(zip(union,map(float,values))); results={}
    for name,pool in pools.items():
        ids=[memory._episode_order[i] for i in pool]
        order=logvolume_order([score[i] for i in pool],memory._episode_vectors[pool],ids) if name=="provenance_logvolume" else sorted(
            range(len(pool)),key=lambda j:(-score[pool[j]],ids[j]))
        results[name]=[MemoryHit(ids[j],memory._documents[pool[j]],score[pool[j]],(ids[j],),
            "development_raw_episode",{"variant":name}) for j in order]
    return results


def summarize_iteration(out,manifest):
    rows=list(unique_rows(out/"retrieval.jsonl").values()); summary=[]
    for d in ("locomo","longmemeval"):
        for method in (*VARIANTS,*BASELINES):
            selected=[r for r in rows if r["dataset"]==d and r["method"]==method]
            if not selected: continue
            entry={"dataset":d,"method":method,"completed":len(selected),"expected":50,
                   "status":"complete_development" if len(selected)==50 else "partial_development"}
            for k in METRICS: entry[k]=float(np.mean([r["metrics"][k] for r in selected]))
            entry["candidate_pool_recall@48"]=float(np.mean([r["candidate_pool_recall@48"] for r in selected])) if method in VARIANTS else None
            entry["mean_candidates"]=float(np.mean([r["candidate_count"] for r in selected]))
            summary.append(entry)
    if summary: write_csv(out/"retrieval.csv",summary)
    complete=len(rows)==manifest["count"]*(len(VARIANTS)+len(BASELINES))
    gate={"status":"complete_development" if complete else "partial_development",
          "completed_question_method_pairs":len(rows),"expected_question_method_pairs":700,
          "completed_unique_questions":len({(r["dataset"],r["question_id"]) for r in rows}),
          "expected_unique_questions":100,"full_runs_authorized":False,"qa_calls":0,"judge_calls":0,
          "reason":"Repeated-look development retrieval only; no answer superiority, mechanism novelty or oral-level conclusion",
          "skipped_methods":{"zep":"Needs authorized local construction LLM",
                             "long_context":"Has no native top-k ranking; requires actual answer evaluation"}}
    if complete:
        # A descriptive screen only, not a significance test or automatic promotion.
        scores={}
        for method in VARIANTS:
            comparisons=[]
            for d in ("locomo","longmemeval"):
                candidate=next(r for r in summary if r["dataset"]==d and r["method"]==method)
                baselines=[r for r in summary if r["dataset"]==d and r["method"] in BASELINES]
                comparisons.append(all(candidate[k]>=max(r[k] for r in baselines) for k in METRICS))
            scores[method]={"matches_or_beats_all_three_cpu_baselines_on_all_four_metrics_in_both_datasets":all(comparisons)}
        gate["descriptive_retrieval_screen"]=scores
    write_json(out/"status.json",gate)
    return summary


def run_iteration(manifest_path,output,methods=None):
    """Resume exact completed question/arm cells; rebuild native memory as needed."""
    cfg=config(); validate_calibration(cfg,ROOT)
    if cfg["retrieval"]["device"]!="cpu": raise ValueError("Development iteration is CPU only")
    manifest=read_json(manifest_path); verify_iteration(manifest,cfg)
    methods=tuple(methods or (*VARIANTS,*BASELINES))
    if not set(methods)<=set((*VARIANTS,*BASELINES)): raise ValueError("Unknown development arm")
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    # Advisory OS lock is released automatically on crash; no artifact deletion.
    with (out/"worker.lock").open("a+b") as lockfile:
        lockfile.seek(0,2)
        if lockfile.tell()==0: lockfile.write(b"0"); lockfile.flush()
        lockfile.seek(0)
        import os
        if os.name=="nt":
            import msvcrt
            msvcrt.locking(lockfile.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lockfile.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity={"manifest_sha256":digest_file(manifest_path),"configuration":cfg,
            "source_fingerprint":source_fingerprint(),"experiment_source_sha256":digest_file(Path(__file__)),
            "calibration_sha256":digest_file(ROOT/cfg["calibration"]["artifact"]),
            "lightmem_source_sha256":digest_file(ROOT/"src/mindset/baselines/lightmem.py"),
            "checkpoint_lock_sha256":digest_file(ROOT/"configs/checkpoints.lock.json"),
            "qa_calls":0,"judge_calls":0,"gpu_calls":0,"scope":"development_not_paper_or_latency",
            "variants":list(VARIANTS),"baselines":list(BASELINES),
            "budgets":{"candidates_per_arm":48,"raw_evidence_episodes":8,"evidence_tokens":8000,
                       "lightmem_requested_hits":retrieval_request_k("lightmem",cfg),
                       "lightmem_internal_pool":"up to twice requested hits; not compute matched to MINDSET"},
            "score_reuse":"identical query/episode scores within development arms only; never efficiency data"}
        write_json(out/"audit.json",identity,immutable=True); record_environment(out)
        completed=unique_rows(out/"retrieval.jsonl")
        expected={f"{r['dataset']}/{m}/{r['id']}" for r in manifest["records"] for m in (*VARIANTS,*BASELINES)}
        if set(completed)-expected: raise ValueError("Unexpected development prediction IDs")
        summarize_iteration(out,manifest)
        encoder,reranker=local_stack(cfg,cache=False); encoder=CachedEncoder(encoder,enabled=True)
        checkpoint=read_json(ROOT/"configs/checkpoints.lock.json")
        tokenizer=load_tokenizer(checkpoint[cfg["evidence_tokenizer"]])
        try:
            for dataset in ("locomo","longmemeval"):
                ids={r["id"] for r in manifest["records"] if r["dataset"]==dataset}
                groups=defaultdict(list)
                for case in cases_for(dataset,cfg):
                    if case.id in ids: groups[case.cluster].append(case)
                for cluster,queries in sorted(groups.items()):
                    for family in ("mindset",*BASELINES):
                        arms=[m for m in methods if m in VARIANTS] if family=="mindset" else [family] if family in methods else []
                        pending=[c for c in queries if any(f"{dataset}/{m}/{c.id}" not in completed for m in arms)]
                        if not pending: continue
                        print(f"Building {dataset}/{cluster}/{family}: {len(pending)} pending questions",flush=True)
                        memory=make_method(family,queries[0].events,cfg,encoder,reranker)
                        if family=="mindset":
                            write_json(out/"structure"/f"{dataset}-{cluster}.json",{
                                "invariants":memory.validate_invariants(),"telemetry":memory.telemetry()})
                        index=EvidenceCache(queries[0].events,encoder,tokenizer,cfg)
                        native_ids={e.event_id:e.session_id if pending[0].retrieval_unit=="session" else e.event_id for e in queries[0].events}
                        for case in pending:
                            todo=[m for m in arms if f"{dataset}/{m}/{case.id}" not in completed]
                            if family=="mindset":
                                pools={k:v for k,v in candidate_variants(memory,case.question).items() if k in todo}
                                results=variant_hits(memory,case.question,pools)
                            else:
                                results={family:memory.retrieve(case.question,retrieval_request_k(family,cfg))}
                            for arm,hits in results.items():
                                selected,packet=index.packet(hits,case.question)
                                candidates={native_ids[e] for h in hits for e in h.source_event_ids if e in native_ids}
                                row={"id":f"{dataset}/{arm}/{case.id}","dataset":dataset,"question_id":case.id,
                                    "cluster":cluster,"category":case.category,"method":arm,
                                    "metrics":rank_metrics(selected,case,8),"selected":selected,
                                    "candidate_ids":[h.memory_id for h in hits],"candidate_count":len(hits),
                                    "candidate_pool_recall@48":len(candidates&set(case.gold))/len(set(case.gold)),
                                    "evidence_tokens":len(tokenizer.encode(packet,add_special_tokens=False)),
                                    "status":"development_not_paper"}
                                append_row(out/"retrieval.jsonl",row); completed[row["id"]]=row
                            summarize_iteration(out,manifest)
                            print(f"Completed {dataset}/{case.id}: {','.join(todo)}; {len(completed)}/700 cells",flush=True)
                        # Histories differ for every LME question. Bound reuse to
                        # the current history so encoder cache cannot grow forever.
                        encoder.cache.clear()
                        del memory,index
            if digest_file(ROOT/"src/mindset/baselines/lightmem.py")!=identity["lightmem_source_sha256"]:
                raise ValueError("LightMem source changed")
        except Exception as exc:
            summarize_iteration(out,manifest)
            state=read_json(out/"status.json"); state.update(status="failed_resumable_development",error=f"{type(exc).__name__}: {exc}")
            write_json(out/"status.json",state)
            raise
        return summarize_iteration(out,manifest)
