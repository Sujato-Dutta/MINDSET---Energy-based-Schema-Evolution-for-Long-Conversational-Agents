"""Freeze or resume the bounded MINDSET development diagnostic."""
from pathlib import Path
import argparse
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
os.environ.setdefault("HF_HOME",str(ROOT/".cache/huggingface"))
os.environ["CUDA_VISIBLE_DEVICES"]=""
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key,"4")
from mindset.acl.iteration import freeze_iteration,run_iteration,VARIANTS,BASELINES

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("freeze","run"))
    parser.add_argument("--manifest",default=str(ROOT/"data/manifests/mindset_v2_development100.json"))
    parser.add_argument("--output",default=str(ROOT/"outputs/acl/mindset_v2_development100_r01"))
    parser.add_argument("--methods",nargs="+",choices=(*VARIANTS,*BASELINES))
    args=parser.parse_args()
    if os.name=="nt" or not os.environ.get("SLURM_JOB_ID"):
        parser.error("Experiment computation is DGX-only: submit hpc/dgx_iteration.slurm through Slurm")
    if args.command=="freeze":
        value=freeze_iteration(args.manifest); print(f"Frozen {value['count']} development questions; no model/API calls")
    else: run_iteration(args.manifest,args.output,args.methods)

if __name__=="__main__": main()
