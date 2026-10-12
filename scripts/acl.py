"""Run from the repository root; pip requirements + PYTHONPATH, no pyproject needed."""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from mindset.acl.pipeline import freeze,prepare,answer,judge,summarize,config,ROOT as PROJECT_ROOT
from mindset.acl.analysis import statistics,efficiency,sweep,human_audit,audit_agreement
from mindset.acl.controllers import fit_controller
from mindset.acl.io import read_json,write_json,digest_file

def main():
    os.chdir(ROOT)
    p=argparse.ArgumentParser(description="ACL local memory benchmark pipeline")
    sub=p.add_subparsers(dest="command",required=True)
    f=sub.add_parser("freeze"); f.add_argument("--manifests-only",action="store_true")
    f=sub.add_parser("prepare"); f.add_argument("--dataset",choices=["locomo","longmemeval"],required=True)
    f.add_argument("--method",choices=config()["methods"],required=True); f.add_argument("--output",required=True)
    f.add_argument("--controller",choices=config()["controllers"],default="energy")
    f.add_argument("--learned-path"); f.add_argument("--limit",type=int)
    f.add_argument("--no-cache",action="store_true")
    f.add_argument("--ablation",choices=config().get("ablations",[]))
    f=sub.add_parser("answer"); f.add_argument("--prepared",required=True)
    f.add_argument("--model",required=True); f.add_argument("--output",required=True)
    f=sub.add_parser("judge"); f.add_argument("--run",required=True)
    for name in ["summarize","statistics"]:
        f=sub.add_parser(name); f.add_argument("--runs",nargs="+",required=True); f.add_argument("--output",required=True)
    f=sub.add_parser("efficiency"); f.add_argument("--dataset",choices=["locomo","longmemeval"],required=True)
    f.add_argument("--methods",nargs="+",default=config()["methods"]); f.add_argument("--output",required=True)
    f.add_argument("--repetitions",type=int,default=3); f.add_argument("--limit",type=int)
    f=sub.add_parser("validate-mindset"); f.add_argument("--output",required=True)
    f.add_argument("--locomo-per-cluster",type=int,default=2)
    f.add_argument("--lme-per-category",type=int,default=1)
    f.add_argument("--datasets",nargs="+",choices=["locomo","longmemeval"])
    f.add_argument("--categories",nargs="+")
    f=sub.add_parser("sweep"); f.add_argument("--dataset",choices=["locomo","longmemeval"],required=True)
    f.add_argument("--output",required=True); f.add_argument("--limit",type=int)
    f=sub.add_parser("profile-controller"); f.add_argument("--output",required=True)
    f.add_argument("--questions",type=int,default=20); f.add_argument("--repetitions",type=int,default=3)
    f=sub.add_parser("calibrate"); f.add_argument("--output",default="outputs/acl/mindset_v2_calibration")
    f.add_argument("--per-family",type=int,default=8)
    f.add_argument("--dataset-path",default="data/development/mindset_v2.json")
    f.add_argument("--corpus",help="Independently labelled external histories with train/validation/holdout splits")
    f=sub.add_parser("train-controller"); f.add_argument("--training",required=True); f.add_argument("--output",required=True)
    f=sub.add_parser("human-audit"); f.add_argument("--runs",nargs="+",required=True)
    f.add_argument("--output",required=True); f.add_argument("--fraction",type=float,default=.10)
    f=sub.add_parser("audit-agreement"); f.add_argument("--audit",required=True); f.add_argument("--output",required=True)
    f=sub.add_parser("plan"); f.add_argument("--output",default="experiments/mindset/plan_instance_ids.json")
    a=p.parse_args()
    if os.name=="nt" or not os.environ.get("SLURM_JOB_ID"):
        p.error("Experiment computation is DGX-only: run this command inside an allocated Slurm job")
    if a.command=="freeze": freeze(not a.manifests_only)
    elif a.command=="prepare":
        if a.ablation and (a.method!="mindset" or a.controller!="energy"):
            raise ValueError("Ablations require the MINDSET energy controller")
        prepare(a.dataset,a.method,a.output,a.controller,a.learned_path,a.limit,
                overrides={"ablation":a.ablation} if a.ablation else None,cache=not a.no_cache)
    elif a.command=="answer": answer(a.prepared,a.model,a.output)
    elif a.command=="judge": judge(a.run)
    elif a.command=="summarize": summarize(a.runs,a.output)
    elif a.command=="statistics": statistics(a.runs,a.output)
    elif a.command=="efficiency": efficiency(a.dataset,a.methods,a.output,a.repetitions,a.limit)
    elif a.command=="sweep": sweep(a.dataset,a.output,a.limit)
    elif a.command=="profile-controller":
        from mindset.acl.profiling import profile_controller
        profile_controller(a.output,a.questions,a.repetitions)
    elif a.command=="validate-mindset":
        from mindset.acl.validation import validate_mindset
        validate_mindset(a.output,a.locomo_per_cluster,a.lme_per_category,a.datasets,a.categories)
    elif a.command=="calibrate":
        from mindset.acl.calibration import calibrate
        calibrate(a.output,a.per_family,a.dataset_path,a.corpus)
    elif a.command=="train-controller":
        forbidden={r["id"] for s in config()["benchmarks"].values()
                   for r in read_json(ROOT/s["manifest"])["records"]}
        from mindset.acl.calibration import content_hash
        from mindset.acl.pipeline import cases_for
        histories={content_hash(c.events) for dataset in config()["benchmarks"] for c in cases_for(dataset,config())}
        clusters={c.cluster for dataset in config()["benchmarks"] for c in cases_for(dataset,config())}
        fit_controller(a.training,a.output,forbidden|clusters,histories)
    elif a.command=="human-audit": human_audit(a.runs,a.output,a.fraction)
    elif a.command=="audit-agreement": audit_agreement(a.audit,a.output)
    elif a.command=="plan":
        cfg=config()
        cells=[{"dataset":d,"model":model,"method":method,"controller":"energy"}
               for d in cfg["benchmarks"] for model,methods in cfg["generator_methods"].items() for method in methods]
        controls=[{"dataset":d,"model":cfg["primary_generator"],"method":"mindset","controller":c}
                  for d in cfg["benchmarks"] for c in cfg["controllers"] if c!="energy"]
        ablations=[{"dataset":d,"model":cfg["primary_generator"],"method":"mindset",
                    "controller":"energy","ablation":a}
                   for d in cfg["benchmarks"] for a in cfg.get("ablations",[])]
        write_json(a.output,{"protocol":cfg["protocol_version"],"generator_cells":cells,"controller_cells":controls,
            "implementation_version":cfg.get("implementation_version"),
            "config_sha256":digest_file(ROOT/"configs/acl.yaml"),
            "manifest_sha256":{d:digest_file(ROOT/s["manifest"]) for d,s in cfg["benchmarks"].items()},
            "calibration_sha256":digest_file(ROOT/cfg["calibration"]["artifact"]),
            "ablation_cells":ablations,
            "memory_builds":[{"dataset":d,"method":m,"controller":"energy"}
                for d in cfg["benchmarks"] for m in cfg["methods"]]+[
                    {k:v for k,v in cell.items() if k!="model"} for cell in controls+ablations],
            "status":"planned_no_results"},immutable=True)
        print(f"Plan: {len(cells)} generation cells + {len(controls)} controller comparisons + {len(ablations)} ablations")

if __name__=="__main__": main()

