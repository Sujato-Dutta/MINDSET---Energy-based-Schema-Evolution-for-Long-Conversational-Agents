"""Audit the exact run IDs named in the legacy paper's corrected statistics."""
import os
import sys
import argparse
from pathlib import Path
from collections import Counter
if os.name=="nt" or not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("DGX Slurm only")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from mindset.acl.io import read_json,write_json,read_rows,digest_file


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",required=True)
    args=parser.parse_args(); out=Path(args.output)
    stats_path=ROOT/"diagnostics_inputs/statistical_tests_v1.json"
    statistics=read_json(stats_path); rows=[]
    for group,methods in statistics["run_manifests"].items():
        for label,run in methods.items():
            folder=ROOT/"diagnostics_inputs/legacy_runs"/run
            if not (folder/"metrics.json").exists():
                rows.append({"group":group,"label":label,"run":run,"status":"not_uploaded_not_imputed"}); continue
            metrics=read_json(folder/"metrics.json"); predictions=read_rows(folder/"predictions.jsonl")
            entry={"group":group,"label":label,"run":run,"status":"verified_selected_artifact",
                "questions":len(predictions),"native_f1":metrics.get("native_f1"),
                "ranking":metrics.get("answerable_retrieval"),"icl_exact_match":metrics.get("icl_exact_match"),
                "icl_count":metrics.get("icl_count"),"recsys_count":metrics.get("recsys_count"),
                "recsys_recall@5":metrics.get("recsys_recall@5"),
                "metrics_sha256":digest_file(folder/"metrics.json"),"predictions_sha256":digest_file(folder/"predictions.jsonl")}
            if group=="mab_main":
                per=metrics["per_example"]; icl=[r for r in per if str(r.get("source","")).startswith("icl_")]
                successes=sum(float(r["exact_match"]) for r in icl)
                if len(icl)!=metrics["icl_count"] or abs(successes/len(icl)-metrics["icl_exact_match"])>1e-12:
                    raise ValueError("Frozen native ICL metric mismatch")
                entry["icl_correct_count"]=successes
                entry["literal_insufficient_evidence_replies_by_source"]={c:{"questions":len(subset),
                    "no_information_available":sum("no information available" in str(r["prediction"]).casefold() for r in subset)}
                    for c,subset in ((c,[r for r in predictions if r.get("category")==c]) for c in sorted({r["category"] for r in predictions}))}
            rows.append(entry)
    write_json(out,{"statistics_sha256":digest_file(stats_path),"records":rows,
        "selection":"Exact run_manifests in corrected frozen statistics; alternate runs are not substituted"},immutable=True)
    for r in rows:
        if r["group"]=="mab_main": print(r,flush=True)

if __name__=="__main__": main()
