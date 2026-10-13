"""Standalone syntax environment: source-grounded, resumable projections."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from time import perf_counter
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mindset.acl.io import write_json, digest_file, append_row, unique_rows, read_json
# Load the pure projection module without importing memory/__init__, whose
# broader controller dependencies deliberately are NOT in the syntax venv.
spec = importlib.util.spec_from_file_location('mindset_syntax_projection', ROOT / 'src/mindset/memory/syntax_frames.py')
projection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projection)
project_frames, project_query = projection.project_frames, projection.project_query


def functional(nlp, out):
    # Independently constructed syntax/invariance checks, not benchmark labels.
    checks = [
        ('My budget is 120 dollars.', 'syntax_property_budget'),
        ('My dentist is Alex.', 'syntax_property_dentist'),
        ('I drink three cups of coffee in the morning.', 'syntax_quantity_cup_coffee'),
        ('I changed my budget to 80 dollars.', 'syntax_property_budget'),
        ('I do not drink three cups of coffee.', None),
        ('Would I drink three cups of coffee?', None),
        ('"My budget is 120 dollars."', None),
        ('My budget was 120 dollars.', None),
        ('My budget is 120 dollars. My budget is 180 dollars.', None),
    ]
    rows = []
    for text, slot in checks:
        frames = project_frames(nlp(text), text)
        if slot is None: assert not frames, (text, frames)
        else: assert slot in {f['slot'] for f in frames}, (text, frames)
        for f in frames: assert text[slice(*f['value_span'])] == f['value']
        rows.append({'text': text, 'expected_slot': slot, 'frames': frames, 'passed': True})
    q = project_query(nlp('What is my current budget?'))
    assert q['possessive_self'] and 'budget' in q['terms']
    q = project_query(nlp("Where does my friend's dentist work?"))
    assert not q['possessive_self'] and not q['direct_self']
    write_json(out / 'functional_tests.json', {'checks': rows, 'query_checks': 2,
        'scope': 'constructed functionality; NOT natural state/answer accuracy'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--input', required=True)
    p.add_argument('--output', required=True); args = p.parse_args()
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    import spacy
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / 'worker.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        nlp = spacy.load('en_core_web_sm')
        if spacy.__version__ != '3.8.7' or nlp.meta['version'] != '3.8.0': raise ValueError('Parser revision mismatch')
        packages = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True).stdout.splitlines()
        fingerprint = hashlib.sha256(nlp.to_bytes()).hexdigest()
        write_json(out / 'audit.json', {'input_sha256': digest_file(args.input),
            'parser': 'en_core_web_sm', 'parser_version': nlp.meta['version'], 'spacy': spacy.__version__,
            'parser_bytes_sha256': fingerprint, 'projection_sha256': digest_file(ROOT / 'src/mindset/memory/syntax_frames.py'),
            'worker_sha256': digest_file(__file__), 'python': sys.version, 'dependencies': packages,
            'hostname': platform.node(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
            'scope': 'no benchmark labels/answers or training; fixed projection assumptions', 'qa_calls': 0}, immutable=True)
        functional(nlp, out)
        inputs = unique_rows(Path(args.input)); done = unique_rows(out / 'frames.jsonl')
        if set(done) - set(inputs): raise ValueError('Unexpected resumed projection IDs')
        for key, row in done.items():
            if row['input_sha256'] != inputs[key]['input_sha256']: raise ValueError('Changed input on resume')
        pending = [r for k, r in inputs.items() if k not in done]
        started = perf_counter()
        for doc, row in nlp.pipe(((r['text'], r) for r in pending), as_tuples=True, batch_size=64, n_process=1):
            t = perf_counter()
            projected = {'frames': project_frames(doc, row['text'])} if row['kind'] == 'episode' else {'query': project_query(doc)}
            result = {**row, **projected, 'projection_seconds': perf_counter() - t}
            append_row(out / 'frames.jsonl', result); done[row['id']] = result
            if len(done) % 1000 == 0: print(f'{len(done)}/{len(inputs)} syntax inputs', flush=True)
        write_json(out / 'status.json', {'status': 'complete_projection', 'rows': len(done),
            'expected_rows': len(inputs), 'fresh_worker_seconds': perf_counter() - started,
            'resume_skipped_rows': len(inputs) - len(pending),
            'timing_scope': 'parser+projection+durable export; NOT inference-only latency', 'qa_calls': 0})
