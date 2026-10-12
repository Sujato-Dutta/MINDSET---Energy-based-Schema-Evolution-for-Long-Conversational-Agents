"""DGX-only isolated joint-energy prototype and development latency runner."""
from pathlib import Path
import os
import sys
import argparse
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("calibrate","quality","latency","latency-worker"))
    p.add_argument("--manifest",default=str(ROOT/"data/manifests/mindset_v2_development100.json"))
    p.add_argument("--calibration",default=str(ROOT/"experiments/mindset/joint_prototype/calibration.json"))
    p.add_argument("--output",required=True)
    p.add_argument("--methods",nargs="+",default=None)
    p.add_argument("--repetition",type=int,default=0)
    p.add_argument("--answers",action="store_true")
    args=p.parse_args()
    if os.name=="nt" or not os.environ.get("SLURM_JOB_ID"):
        p.error("All computation must run on DGX through Slurm; never on laptop or login node")
    sys.path.insert(0,str(ROOT/"src"))
    from mindset.acl.joint_calibration import calibrate_joint
    from mindset.acl.joint_iteration import quality,latency,latency_worker,ARMS
    if args.methods and not set(args.methods)<=set(ARMS): p.error("Unknown development method arm")
    if args.command=="calibrate": calibrate_joint(args.output)
    elif args.command=="quality": quality(args.manifest,args.calibration,args.output,args.methods or ARMS)
    elif args.command=="latency": latency(args.manifest,args.calibration,args.output,args.methods or ("joint","compact","lightmem"),answers=args.answers)
    else: latency_worker(args.manifest,args.calibration,args.output,args.methods or ("joint","compact","lightmem"),args.repetition,args.answers)

if __name__=="__main__": main()
