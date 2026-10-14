"""Freeze complete native KU IDs and prior exposure; never score-based sampling."""
from pathlib import Path
import argparse
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mindset.acl.knowledge_update import freeze

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--manifest',required=True); p.add_argument('--calibration',required=True)
    a=p.parse_args()
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    m=freeze(a.manifest,a.calibration)
    print({'count':m['count'],'development100_overlap':len(m['overlap_development100_ids']),
           'known_v2_scored_exposure':len(m['known_v2_scored_exposure']),
           'pristine_holdout_claim':m['pristine_holdout_claim'],'parameter_fit':False},flush=True)
