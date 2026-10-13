"""Real-model smoke checks in the isolated DGX syntax environment."""
from pathlib import Path
import argparse
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'src'))
from parse_syntax_inputs import functional

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True); a = p.parse_args()
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    import spacy
    nlp = spacy.load('en_core_web_sm')
    assert spacy.__version__ == '3.8.7' and nlp.meta['version'] == '3.8.0'
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    functional(nlp, out)
    print('9 real-parser source checks + 2 query checks passed', flush=True)
