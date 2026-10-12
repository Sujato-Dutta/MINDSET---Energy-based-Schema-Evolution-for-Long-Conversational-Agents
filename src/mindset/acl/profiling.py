"""Local paired v1/v2 profiling; timing diagnostics, never QA claims."""
from __future__ import annotations
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.stats import t
from mindset.acl.pipeline import ROOT,config,cases_for,local_stack,record_environment,write_csv
from mindset.acl.io import write_json
from mindset.acl.runtime import ResourceMonitor,distribution,snapshot
from mindset.methods.mindset import MindsetMemory,MindsetConfig
from mindset.methods.compact import CompactMindsetMemory,CompactConfig

def mean_ci(values):
    values=np.asarray(values,dtype=float)
    if len(values)<2 or not np.isfinite(values).all(): raise ValueError("At least two finite repetitions required")
    mean=float(values.mean()); sd=float(values.std(ddof=1))
    half=float(t.ppf(.975,len(values)-1)*sd/np.sqrt(len(values)))
    return {"mean":mean,"sample_std":sd,"ci95_low":mean-half,"ci95_high":mean+half}

def profile_controller(output,questions=20,repetitions=3):
    if repetitions<3 or questions<2: raise ValueError("Use >=3 repetitions and >=2 questions")
    cfg=config()
    out=Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Timing measurements require a fresh output directory, not resumed/cache-hit results")
    cases=cases_for("locomo",cfg)[:questions]
    out.mkdir(parents=True,exist_ok=True); record_environment(out)
    encoder,reranker=local_stack(cfg,cache=False)
    groups={}
    for case in cases: groups.setdefault(case.cluster,[]).append(case)
    rows=[]
    for rep in range(repetitions):
        order=["v1","v2"] if rep%2==0 else ["v2","v1"]
        for version in order:
            ingest=0.; retrieval=[]; total_episodes=0; peaks=[]; storage=0
            for cluster,queries in groups.items():
                with ResourceMonitor() as monitor:
                    if version=="v1":
                        m=MindsetMemory(MindsetConfig(),encoder=encoder,local_reranker=reranker)
                    else:
                        params=dict(cfg["mindset"]); params.pop("profile",None)
                        m=CompactMindsetMemory(CompactConfig(**params),encoder=encoder,local_reranker=reranker)
                    started=perf_counter(); m.ingest(queries[0].events); ingest+=perf_counter()-started
                    total_episodes+=len(queries[0].events); m.validate_invariants()
                    m.retrieve(queries[0].question,8)  # Same single untimed warmup.
                    for case in queries:
                        started=perf_counter(); m.retrieve(case.question,8)
                        retrieval.append(perf_counter()-started)
                    folder=out/version/f"rep-{rep}"/cluster
                    snapshot(m,queries[0].events,folder/"snapshot.json")
                    from dataclasses import asdict
                    write_json(folder/"raw_events.json",[asdict(e) for e in queries[0].events])
                    storage+=sum(p.stat().st_size for p in folder.iterdir())
                peaks.append(monitor.peak_rss)
            row={"version":version,"repetition":rep,"questions":len(cases),"episodes":total_episodes,
                "ingestion_seconds_per_episode":ingest/total_episodes,
                "retrieval_seconds_per_query":float(np.mean(retrieval)),
                "retrieval_p95_seconds":float(np.quantile(retrieval,.95)),
                "peak_worker_rss_bytes":max(peaks),"serialized_memory_bytes":storage,
                "scope":"native_controller_smoke; shared_packet_and_generation_excluded",
                "pair_score_cache":"disabled","remote_calls":0}
            rows.append(row)
    write_csv(out/"repetitions.csv",rows)
    from mindset.acl.pipeline import source_fingerprint
    report={"method":"mindset","implementation_version":"v2","source_fingerprint":source_fingerprint(),
            "status":"diagnostic_not_final_paper","questions":len(cases),"conversation_clusters":len(groups),
            "repetitions":repetitions,"warmup":1,"cache":"no_output_pair_cache; fresh memory each repetition",
            "versions":{}}
    for version in ("v1","v2"):
        report["versions"][version]={metric:mean_ci([r[metric] for r in rows if r["version"]==version])
            for metric in ("ingestion_seconds_per_episode","retrieval_seconds_per_query")}
    for metric in ("ingestion_seconds_per_episode","retrieval_seconds_per_query"):
        ratios=[]
        for rep in range(repetitions):
            pair={r["version"]:r for r in rows if r["repetition"]==rep}
            ratios.append(np.log(pair["v2"][metric]/pair["v1"][metric]))
        interval=mean_ci(ratios)
        report[metric+"_v2_over_v1"]={k:float(np.exp(interval[k])) for k in ("mean","ci95_low","ci95_high")}
    write_json(out/"profile.json",report)
    print("Paired CPU controller profile saved:",out,flush=True)
    return report
