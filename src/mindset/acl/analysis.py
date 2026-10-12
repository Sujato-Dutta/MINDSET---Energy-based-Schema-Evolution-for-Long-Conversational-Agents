from __future__ import annotations
import itertools
from collections import defaultdict
from pathlib import Path
import numpy as np
from mindset.acl.io import read_json, unique_rows, write_json
from mindset.acl.pipeline import config, ROOT, prepare, write_csv, distribution
from mindset.statistics.tests import holm_correction, exact_sign_flip_pvalue, exact_mcnemar
from mindset.utils.hashing import stable_hash

def paired_ids_for_metric(first,second,metric):
    ids=[i for i in first if i in second]
    if metric.startswith(("recall","precision","mrr","ndcg")):
        return [i for i in ids if first[i].get("has_gold_evidence",False)
                and second[i].get("has_gold_evidence",False)]
    return ids

def paired_stats(a,b,clusters=None,seed=0,resamples=10000):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    if a.shape!=b.shape or a.ndim!=1 or not len(a) or not np.all(np.isfinite(a-b)):
        raise ValueError("Nonempty aligned finite score arrays required")
    diff=a-b; rng=np.random.default_rng(seed)
    if clusters is not None:
        if len(clusters)!=len(a): raise ValueError("Cluster IDs are misaligned")
        ids=sorted(set(clusters))
        groups=[diff[np.asarray(clusters)==c] for c in ids]
        sums=np.asarray([g.sum() for g in groups]); counts=np.asarray([len(g) for g in groups])
        draw=rng.integers(0,len(groups),(resamples,len(groups)))
        sampled=sums[draw].sum(axis=1)/counts[draw].sum(axis=1)
        p=exact_sign_flip_pvalue(sums)
    else:
        draw=rng.integers(0,len(diff),(resamples,len(diff)))
        sampled=diff[draw].mean(axis=1)
        if set(a)|set(b)<={0.0,1.0}:
            p=exact_mcnemar(a.astype(bool),b.astype(bool))["p_value"]
        else:
            p=(1+sum(abs(float((rng.choice([-1,1],len(diff))*diff).mean()))>=
                 abs(float(diff.mean()))-1e-12 for _ in range(resamples)))/(resamples+1)
    low,high=np.quantile(sampled,[.025,.975])
    return {"count":len(a),"delta":float(diff.mean()),"ci_low":float(low),"ci_high":float(high),
            "p":float(p),"units":len(set(clusters)) if clusters is not None else len(a)}

def statistics(runs,output):
    from mindset.acl.pipeline import complete_judge_labels
    groups=defaultdict(list); cfg=config()
    for path in runs:
        p=Path(path); rc=read_json(p/"run_config.json")
        groups[(rc["prepared_config"]["dataset"],rc["model"])].append((p,rc))
    results=[]
    for (dataset,model),cells in groups.items():
        ours=[(p,c) for p,c in cells if c["prepared_config"]["method"]=="mindset"
              and c["prepared_config"]["controller"]=="energy"
              and c["prepared_config"]["protocol"].get("mindset",{}).get("ablation","full")=="full"]
        if len(ours)!=1: raise ValueError("Exactly one full MINDSET run required per panel")
        op,oc=ours[0]; a=unique_rows(op/"predictions.jsonl"); aj=unique_rows(op/"judgements.jsonl")
        expected=read_json(ROOT/cfg["benchmarks"][dataset]["manifest"])["records"]
        if set(a)!={r["id"] for r in expected}: raise ValueError("Incomplete main run")
        a_has_judge=complete_judge_labels(dataset,a,aj)
        versions={r.get("provider_model_version") for r in aj.values()}
        families=defaultdict(list)
        for bp,bc in cells:
            if bp==op: continue
            b=unique_rows(bp/"predictions.jsonl"); bj=unique_rows(bp/"judgements.jsonl")
            if set(a)!=set(b): raise ValueError("Unpaired/incomplete benchmark runs")
            b_has_judge=complete_judge_labels(dataset,b,bj)
            versions.update(r.get("provider_model_version") for r in bj.values())
            metrics=["token_f1","recall@8","precision@8","mrr@8","ndcg@8"]
            if a_has_judge and b_has_judge: metrics.insert(0,"judge_accuracy")
            if dataset=="locomo": metrics.insert(1,"native_f1")
            for metric in metrics:
                ids=list(a)
                if metric.startswith(("recall","precision","mrr","ndcg")):
                    ids=[i for i in ids if a[i]["retrieval"] is not None and b[i]["retrieval"] is not None]
                if not ids: continue
                views={"all":ids}
                if dataset=="longmemeval":
                    for category in cfg["statistics"].get("prespecified_longmemeval_subsets",[]):
                        views[category]=[i for i in ids if a[i]["category"]==category]
                def value(rows,judges,i):
                    if metric=="judge_accuracy": return int(judges[i]["label"])
                    if metric in {"token_f1","native_f1"}: return rows[i][metric]
                    return rows[i]["retrieval"][metric]
                family="controller" if bc["prepared_config"]["method"]=="mindset" else "baseline"
                for view,selected in views.items():
                    if not selected: continue
                    av=[value(a,aj,i) for i in selected]; bv=[value(b,bj,i) for i in selected]
                    result=paired_stats(av,bv,[a[i]["cluster"] for i in selected] if dataset=="locomo" else None,
                                        cfg["seed"],cfg["statistics"]["resamples"])
                    binary=set(av)|set(bv)<={0.,1.}
                    row={"dataset":dataset,"generator":model,"baseline":bc["prepared_config"]["method"],
                        "ablation":bc["prepared_config"]["protocol"].get("mindset",{}).get("ablation","full")
                            if bc["prepared_config"]["method"]=="mindset" else "not_applicable",
                        "controller":bc["prepared_config"]["controller"],"metric":metric,"view":view,
                        "hypothesis_family":family,**result,"id_set_sha256":stable_hash(sorted(selected)),
                        "alternative":"two-sided","test":"cluster_exact_sign_flip" if dataset=="locomo" else
                        ("exact_mcnemar" if binary else "paired_randomization")}
                    families[(metric,family)].append(row)
        if versions and (None in versions or len(versions)!=1):
            raise ValueError("Final inference requires one recorded judge revision across the comparison panel")
        for rows in families.values():
            for row,p in zip(rows,holm_correction([r["p"] for r in rows])):
                row["holm_p"]=p; results.append(row)
    write_csv(Path(output)/"statistics.csv",results)

def efficiency(dataset,methods,output,repetitions=3,limit=None):
    import subprocess,sys,time
    from mindset.acl.profiling import mean_ci
    if repetitions<3: raise ValueError("Paper efficiency protocol requires >=3 repetitions")
    cfg=config()
    if not methods or any(m not in cfg["methods"] for m in methods): raise ValueError("Unknown method panel")
    out=Path(output).resolve(); report=[]; hardware=None
    if out.exists() and any(out.iterdir()):
        raise ValueError("Efficiency measurements require a fresh output directory; do not time resumed artifacts")
    rng=np.random.default_rng(cfg["seed"])
    for rep in range(repetitions):
        for method in rng.permutation(methods).tolist():
            folder=out/dataset/method/f"rep-{rep}"
            command=[sys.executable,str(ROOT/"scripts/acl.py"),"prepare","--dataset",dataset,
                     "--method",method,"--output",str(folder),"--no-cache"]
            if limit: command.extend(["--limit",str(limit)])
            started=time.perf_counter()
            subprocess.run(command,cwd=ROOT,check=True)
            worker_wall=time.perf_counter()-started
            env=read_json(folder/"environment.json")
            signature={k:env.get(k) for k in ("hostname","machine","processor","logical_cpus","cpu_affinity","dependencies")}
            allocation=env.get("allocation",{})
            signature["threads"]={k:allocation.get(k) for k in ("SLURM_CPUS_PER_TASK","OMP_NUM_THREADS","MKL_NUM_THREADS")}
            if hardware is None: hardware=signature
            elif signature!=hardware: raise ValueError("Efficiency repetitions do not share hardware/software/thread settings")
            ing=list(unique_rows(folder/"ingestion.jsonl").values())
            queries=list(unique_rows(folder/"evidence.jsonl").values())
            row={"dataset":dataset,"method":method,"repetition":rep,"cache_policy":"fresh_process_no_score_cache",
                 "questions":len(queries),"episodes":sum(r["episodes"] for r in ing),
                 "worker_wall_seconds":worker_wall,"protocol_status":cfg.get("status")}
            for field,vals in {
                "ingestion_seconds_per_episode":[r["seconds_per_episode"] for r in ing],
                "retrieval_seconds_per_query":[r["retrieval_seconds"] for r in queries],
                "serialized_memory_bytes":[r["serialized_memory_bytes"] for r in ing],
                "serialization_seconds":[r["serialization_seconds"] for r in ing],
                "peak_process_rss_bytes":[r["peak_process_rss_bytes"] for r in ing],
                "peak_device_vram_bytes":[r["peak_device_vram_bytes"] for r in ing if r["peak_device_vram_bytes"] is not None]
            }.items():
                for stat,value in distribution(vals).items(): row[field+"_"+stat]=value
                row[field+"_max"]=max(vals) if vals else None
            row["weighted_ingestion_seconds_per_episode"]=sum(r["ingestion_seconds"] for r in ing)/max(1,row["episodes"])
            row["construction_local_llm_calls"]=sum(r["telemetry"].get("write_local_llm_calls",0) for r in ing)
            for direction in ("input","output"):
                row["construction_"+direction+"_tokens"]=sum(r["telemetry"].get("construction_"+direction+"_tokens",0) for r in ing)
            row["remote_provider_calls"]=0; row["status"]="smoke" if limit else "complete"
            from mindset.acl.latency import latency_summary
            row.update(latency_summary(queries))
            report.append(row)
    write_csv(out/"efficiency_repetitions.csv",report)
    aggregate=[]
    for method in methods:
        rows=[r for r in report if r["method"]==method]
        summary={"dataset":dataset,"method":method,"repetitions":len(rows),"status":"smoke" if limit else "complete",
                 "ci_method":"Student_t_on_independent_fresh_process_repetition_means"}
        for k in rows[0]:
            if k.endswith("_mean") or k=="weighted_ingestion_seconds_per_episode":
                vals=[r[k] for r in rows if r[k] is not None]
                interval=mean_ci(vals) if len(vals)>=3 else None
                summary[k]=interval["mean"] if interval else None
                for field in ("sample_std","ci95_low","ci95_high"):
                    summary[k+"_"+field]=interval[field] if interval else None
        for k in ("peak_process_rss_bytes_max","peak_device_vram_bytes_max"):
            vals=[r[k] for r in rows if r[k] is not None]; summary[k]=max(vals) if vals else None
        # Pool fresh repetition observations for descriptive latency percentiles;
        # keep each repetition's percentiles in the companion CSV as well.
        queries=[r for rep in range(repetitions) for r in unique_rows(
            out/dataset/method/f"rep-{rep}"/"evidence.jsonl").values()]
        from mindset.acl.latency import latency_summary
        summary.update(latency_summary(queries))
        summary["latency_scope"]="retrieval_plus_evidence; end_to_end_unavailable_without_actual_answers"
        aggregate.append(summary)
    write_csv(out/"efficiency.csv",aggregate)
    write_json(out/"hardware.json",hardware)

def sweep(dataset,output,limit=None):
    cfg=config(); grid=cfg["parameter_sweep"]
    for values in itertools.product(*grid.values()):
        setting=dict(zip(grid,values)); folder=Path(output)/stable_hash(setting)[:16]
        prepare(dataset,"mindset",folder,limit=limit,overrides=setting)
    rows=[]
    for folder in sorted(Path(output).iterdir()):
        if not (folder/"evidence.jsonl").is_file(): continue
        rc=read_json(folder/"run_config.json"); q=list(unique_rows(folder/"evidence.jsonl").values())
        rank=[r["retrieval"] for r in q if r["retrieval"] is not None]
        snapshots=list((folder/"memory").glob("*/snapshot.json"))
        transitions=[t for p in snapshots for t in read_json(p).get("transitions",[])]
        row={k:rc["protocol"]["mindset"][k] for k in grid}
        row.update({"retrieval_count":len(rank),"schemas":sum(len(read_json(p).get("schemas",[])) for p in snapshots),
                    "question_count":len(q),"status":"smoke" if rc["limit"] else "complete"})
        for k in rank[0] if rank else []: row[k]=float(np.mean([r[k] for r in rank]))
        for op in ["reinforce","supersede","split","new"]:
            row[op]=sum(t["operation"]==op for t in transitions)
        row["hysteresis_applied"]=sum(t["hysteresis_applied"] for t in transitions)
        rows.append(row)
    write_csv(Path(output)/"robustness.csv",rows)

def human_audit(runs,output,fraction=.1):
    if not .05<=fraction<=.10: raise ValueError("Audit fraction must be 5–10%")
    rng=np.random.default_rng(0); rows=[]; by_type=defaultdict(list)
    for path in runs:
        p=Path(path); pred=unique_rows(p/"predictions.jsonl"); judged=unique_rows(p/"judgements.jsonl")
        rc=read_json(p/"run_config.json")
        for id,row in pred.items():
            if id not in judged: raise ValueError("Judge run incomplete")
            by_type[(rc["prepared_config"]["dataset"],row["category"],
                     rc["prepared_config"]["method"],rc["model"])].append((p,row,judged[id]))
    for group,items in sorted(by_type.items()):
        for i in sorted(rng.choice(len(items),size=max(1,int(np.ceil(len(items)*fraction))),replace=False)):
            path,row,j=items[int(i)]
            rows.append({"audit_id":stable_hash((str(path),row["id"]))[:20],"dataset":group[0],
                "category":group[1],"question":row["question"],"reference":row["answer"],
                "prediction":row["prediction"],"human_label":"","question_id":row["id"]})
    write_csv(output,rows)
    # Blinded human sheet excludes method and judge labels; private mapping allows agreement computation.
    mapping={stable_hash((str(p),r["id"]))[:20]:{"run":str(p),"id":r["id"],"judge_label":int(j["label"])}
             for items in by_type.values() for p,r,j in items}
    write_json(str(output)+".mapping.json",{r["audit_id"]:mapping[r["audit_id"]] for r in rows})

def audit_agreement(path,output):
    import csv
    from sklearn.metrics import cohen_kappa_score
    with Path(path).open(encoding="utf-8") as f: rows=list(csv.DictReader(f))
    mapping=read_json(str(path)+".mapping.json")
    if {r["audit_id"] for r in rows}!=set(mapping): raise ValueError("Audit rows differ from frozen mapping")
    human=[]; judge=[]
    for r in rows:
        if r["human_label"] not in {"0","1"}: raise ValueError("Actual human labels (0/1) are required")
        human.append(int(r["human_label"])); judge.append(mapping[r["audit_id"]]["judge_label"])
    kappa=float(cohen_kappa_score(human,judge))
    write_json(output,{"count":len(rows),"agreement":float(np.mean(np.asarray(human)==judge)),
                       "cohen_kappa":kappa if np.isfinite(kappa) else None})

