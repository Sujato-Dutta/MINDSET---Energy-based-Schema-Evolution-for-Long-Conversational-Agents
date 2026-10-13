"""Input-only scope diagnosis. Never read/use answer or retrieval gold scores."""
from pathlib import Path
from collections import Counter
import argparse
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mindset.acl.io import unique_rows, write_json, digest_file

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True); p.add_argument('--output', required=True); args = p.parse_args()
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    rows = [r for r in unique_rows(Path(args.input)).values() if r['arm'] == 'full']
    summaries, inputs = [], []
    for dataset in ('locomo', 'longmemeval'):
        selected = [r for r in rows if r['dataset'] == dataset]
        summaries.append({'dataset': dataset, 'questions': len(selected),
            'scope_modes': dict(Counter(r['certificate']['scope']['mode'] for r in selected)),
            'identity_bindings': dict(Counter(r['certificate'].get('identity_binding') for r in selected)),
            'with_owner': sum(bool(r['certificate']['scope']['owners']) for r in selected),
            'with_slot': sum(bool(r['certificate']['scope']['slots']) for r in selected),
            'with_owner_and_slot': sum(bool(r['certificate']['scope']['owners'] and r['certificate']['scope']['slots']) for r in selected)})
        for r in selected:
            inputs.append({'dataset': dataset, 'question_id': r['question_id'], 'question': r['question'],
                'question_date': r['question_date'], 'scope': r['certificate']['scope'],
                'binding': r['certificate'].get('identity_binding'), 'grounded_candidates': r['certificate']['grounded_candidates']})
    write_json(Path(args.output), {'source_sha256': digest_file(args.input), 'scope': 'input-only diagnostic; no answers/gold/score-driven fitting',
                                 'summary': summaries, 'questions': inputs})
