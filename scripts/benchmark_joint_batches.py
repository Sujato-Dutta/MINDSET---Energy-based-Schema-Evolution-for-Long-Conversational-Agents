"""DGX-only CE batching microbenchmark: operational budgets, no QA tuning."""
from pathlib import Path
import argparse
import os
import sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--calibration",required=True); p.add_argument("--output",required=True)
    args=p.parse_args()
    if os.name=="nt" or not os.environ.get("SLURM_JOB_ID"): p.error("DGX Slurm only")
    sys.path.insert(0,str(ROOT/"src"))
    import numpy as np
    from mindset.acl.joint_iteration import protocol,build,groups_for
    from mindset.acl.pipeline import local_stack,record_environment,write_csv
    from mindset.acl.io import write_json
    manifest=ROOT/"data/manifests/mindset_v2_development100.json"
    cfg,cohort,calibration=protocol(manifest,args.calibration)
    out=Path(args.output)
    if out.exists() and any(out.iterdir()): raise ValueError("Fresh batching benchmark directory required")
    out.mkdir(parents=True); record_environment(out)
    encoder,reranker=local_stack(cfg,cache=False); pairs=[]
    # Four manifest-order questions per benchmark, not score-selected questions.
    ids={d:[r["id"] for r in cohort["records"] if r["dataset"]==d][:4] for d in ("locomo","longmemeval")}
    for dataset,cluster,queries in groups_for(cohort,cfg):
        selected=[q for q in queries if q.id in ids[dataset]]
        if not selected: continue
        memory=build("joint",queries[0].events,cfg,calibration["selected_parameters"],encoder,reranker)
        for q in selected:
            candidates,_=memory.candidate_pool(q.question)
            pairs.append((dataset,q.id,[(q.question,memory._documents[i]) for i in candidates]))
        del memory
    rows=[]
    for dataset,id,query_pairs in pairs:
        reference=np.asarray(reranker.predict(query_pairs,batch_size=8,show_progress_bar=False))
        top8=np.argsort(-reference,kind="stable")[:8].tolist()
        for batch in (8,16,32,48):
            reranker.predict(query_pairs,batch_size=batch,show_progress_bar=False)
            for repetition in range(5):
                start=perf_counter(); scores=np.asarray(reranker.predict(query_pairs,batch_size=batch,show_progress_bar=False)); elapsed=perf_counter()-start
                rows.append({"dataset":dataset,"id":id,"batch_size":batch,"repetition":repetition,
                    "pairs":len(query_pairs),"cross_encoder_ms":elapsed*1000,
                    "max_abs_score_change":float(np.max(np.abs(scores-reference))),
                    "identical_top8_order":np.argsort(-scores,kind="stable")[:8].tolist()==top8})
        print(f"Benchmarked CE batching: {dataset}/{id}",flush=True)
    write_csv(out/"cross_encoder_batches.csv",rows)
    summary=[]
    for batch in (8,16,32,48):
        subset=[r for r in rows if r["batch_size"]==batch]; times=[r["cross_encoder_ms"] for r in subset]
        summary.append({"batch_size":batch,"measurements":len(times),"cross_encoder_p50_ms":float(np.median(times)),
            "cross_encoder_p95_ms":float(np.quantile(times,.95)),
            "all_top8_orders_identical":all(r["identical_top8_order"] for r in subset),
            "max_abs_score_change":max(r["max_abs_score_change"] for r in subset)})
    write_csv(out/"summary.csv",summary)
    write_json(out/"audit.json",{"scope":"CE-only microbenchmark; not retrieval or end-to-end paper latency",
        "question_ids":ids,"selection":"first four frozen-manifest IDs per benchmark, no outcome-based selection",
        "calibration":str(args.calibration),"cache_policy":"no score cache; discarded per-batch warmup",
        "parameter_changes":False,"answers":0,"source_update_during_running_pilot":False})

if __name__=="__main__": main()
