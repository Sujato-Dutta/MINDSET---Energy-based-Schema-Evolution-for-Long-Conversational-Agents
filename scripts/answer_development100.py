"""DGX-only bounded original development100 answer continuation."""
from pathlib import Path
import argparse
import os
import sys
ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default=str(ROOT/'data/manifests/mindset_v2_development100.json'))
    for name in ('calibration','prepared','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--inherit-successful-cache')
    p.add_argument('--regression-panel',action='store_true')
    p.add_argument('--reader-protocol',choices=('current','locomo_short_phrase'),default='current')
    p.add_argument('--native-captions',action='store_true')
    a=p.parse_args()
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    sys.path.insert(0,str(ROOT/'src'))
    from mindset.acl.development_answers import run
    run(a.manifest,a.calibration,a.prepared,a.output,a.inherit_successful_cache,a.regression_panel,a.reader_protocol,a.native_captions)
