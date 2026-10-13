"""Export development input text ONLY, without gold, answers or categories."""
from pathlib import Path
import argparse
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mindset.acl.joint_iteration import protocol, groups_for
from mindset.acl.io import write_json, append_row, digest_file
from mindset.utils.hashing import stable_hash

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True)
    p.add_argument('--calibration', required=True)
    p.add_argument('--manifest', default=str(ROOT / 'data/manifests/mindset_v2_development100.json'))
    a = p.parse_args()
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    cfg, cohort, _ = protocol(a.manifest, a.calibration); out = Path(a.output)
    if out.exists() and any(out.iterdir()): raise ValueError('Use fresh input export')
    out.mkdir(parents=True)
    sources = queries = 0
    for dataset, cluster, cases in groups_for(cohort, cfg):
        for event in cases[0].events:
            row = {'id': f'{dataset}/{cluster}/{event.event_id}', 'kind': 'episode',
                   'dataset': dataset, 'cluster': cluster, 'source_id': event.event_id,
                   'speaker': event.speaker, 'timestamp': event.timestamp, 'text': event.text}
            row['input_sha256'] = stable_hash(row)
            append_row(out / 'inputs.jsonl', row); sources += 1
        for case in cases:
            row = {'id': f'{dataset}/{cluster}/query/{case.id}', 'kind': 'query', 'dataset': dataset,
                   'cluster': cluster, 'question_id': case.id, 'question_date': case.question_date, 'text': case.question}
            row['input_sha256'] = stable_hash(row)
            append_row(out / 'inputs.jsonl', row); queries += 1
    write_json(out / 'audit.json', {'manifest_sha256': digest_file(a.manifest),
        'worker_sha256': digest_file(__file__), 'input_sha256': digest_file(out / 'inputs.jsonl'),
        'episodes': sources, 'questions': queries, 'fields_excluded': ['answer', 'gold', 'category', 'metrics'],
        'scope': 'source/question syntax projection only; no fitting'})
