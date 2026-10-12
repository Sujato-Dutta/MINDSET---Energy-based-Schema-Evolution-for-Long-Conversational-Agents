"""DGX-only search-depth diagnostic; LightMem implementation is unchanged."""
from pathlib import Path
from collections import defaultdict
import argparse
import os
import sys
if os.name=="nt" or not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("DGX Slurm-only diagnosis")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import json
import numpy as np
from mindset.acl.pipeline import config,cases_for,local_stack,rank_metrics,write_csv,source_fingerprint
from mindset.acl.io import read_json,write_json,digest_file,append_row
from mindset.acl.clients import load_tokenizer
from mindset.acl.evidence import EvidenceCache
from mindset.acl.runtime import CachedEncoder
from mindset.baselines.lightmem import LightMemACL2026Reproduction
from mindset.methods.rerank_cache import CachedReranker


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",required=True)
    args=parser.parse_args(); out=Path(args.output)
    if out.exists() and any(out.iterdir()): raise ValueError("Fresh output directory required")
    out.mkdir(parents=True,exist_ok=True); cfg=config()
    lock=read_json(ROOT/"configs/checkpoints.lock.json")
    baseline_sha=digest_file(ROOT/"src/mindset/baselines/lightmem.py")
    write_json(out/"audit.json",{"source_fingerprint":source_fingerprint(),"script_sha256":digest_file(__file__),
        "lightmem_source_sha256":baseline_sha,"scope":"50 fixed development questions; native source unmodified; search budget only",
        "requested_return_counts":[8,24,40],"nominal_ce_candidates":[16,48,80],"final_evidence_episodes":8,
        "qa_calls":0,"efficiency_claims":False})
    encoder,reranker=local_stack(cfg,cache=False); encoder=CachedEncoder(encoder,enabled=True)
    cached=CachedReranker(reranker,json.dumps(lock["reranker"],sort_keys=True),out/"pairs.sqlite")
    class Counting:
        def predict(self,pairs,**kwargs):
            self.last_count=len(pairs)
            return cached.predict(pairs,**kwargs)
    selector=Counting(); tokenizer=load_tokenizer(lock[cfg["evidence_tokenizer"]])
    ids={r["id"] for r in read_json(ROOT/"data/manifests/mindset_v2_development100.json")["records"] if r["dataset"]=="locomo"}
    groups=defaultdict(list)
    for case in cases_for("locomo",cfg):
        if case.id in ids: groups[case.cluster].append(case)
    rows=[]
    for cluster,cases in sorted(groups.items()):
        memory=LightMemACL2026Reproduction(encoder,selector); memory.ingest(cases[0].events)
        index=EvidenceCache(cases[0].events,encoder,tokenizer,cfg)
        for case in cases:
            for requested in (8,24,40):
                hits=memory.retrieve(case.question,requested); selected,_=index.packet(hits,case.question)
                row={"id":f"{case.id}/return{requested}","question_id":case.id,"cluster":cluster,
                    "requested_return_count":requested,"actual_ce_pairs":selector.last_count,
                    "selected_ids":[s["event_id"] for s in selected],"metrics":rank_metrics(selected,case,8)}
                append_row(out/"predictions.jsonl",row); rows.append(row)
            print(f"LightMem native-16 / cost-matched-48 / expanded-80: {case.id}",flush=True)
        encoder.cache.clear()
    summary=[]
    for requested in (8,24,40):
        subset=[r for r in rows if r["requested_return_count"]==requested]
        summary.append({"requested_return_count":requested,"questions":len(subset),
            "mean_ce_pairs":float(np.mean([r["actual_ce_pairs"] for r in subset])),
            **{k:float(np.mean([r["metrics"][k] for r in subset])) for k in ("recall@8","precision@8","mrr@8","ndcg@8")}})
    if digest_file(ROOT/"src/mindset/baselines/lightmem.py")!=baseline_sha: raise ValueError("Baseline changed")
    write_csv(out/"summary.csv",summary)
    write_json(out/"status.json",{"status":"complete_diagnostic","questions":50,"arms":3,"qa_calls":0,"lightmem_unchanged":True})

if __name__=="__main__": main()
