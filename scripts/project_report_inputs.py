"""Pinned syntax environment: input-only extractive report projection."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import hashlib
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('quantity_report_projection', ROOT / 'src/mindset/memory/report_frames.py')
projection = importlib.util.module_from_spec(spec); spec.loader.exec_module(projection)


def main(args):
    import spacy
    nlp = spacy.load('en_core_web_sm')
    if spacy.__version__ != '3.8.7' or nlp.meta['version'] != '3.8.0': raise ValueError('Parser revision mismatch')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        with (out / name).open('w', encoding='utf8') as f: json.dump(value, f, indent=2, ensure_ascii=False)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    tests = []
    for text, expected in [('I led five workers.', '5'), ('I currently lead nine workers.', '9'),
                           ('Morgan manages three contractors.', '3'), ('I lead twenty-one workers.', '21'),
                           ('I could lead three workers.', None), ('I do not lead three workers.', None),
                           ('Would I lead three workers?', None), ('"I lead three workers."', None),
                           ('I lead about three workers.', None)]:
        frames = projection.project_reports(nlp(text), text, 'user')
        if expected is None: assert not frames, (text, frames)
        else: assert expected in [f['normalized_value'] for f in frames], (text, frames)
        tests.append({'text': text, 'expected': expected, 'frames': frames, 'passed': True})
    assert projection.project_report_query(nlp('How many workers did I supervise initially? How many do I supervise now?'),
                                         'How many workers did I supervise initially? How many do I supervise now?')['mode'] == 'report_compare'
    save('functional_tests.json', tests)
    save('audit.json', {'input_sha256': sha(args.input), 'worker_sha256': sha(__file__),
        'projection_sha256': sha(ROOT / 'src/mindset/memory/report_frames.py'), 'parser': nlp.meta,
        'parser_bytes_sha256': hashlib.sha256(nlp.to_bytes()).hexdigest(), 'spacy': spacy.__version__,
        'scope': 'source/query inputs only; no answers/gold or current-state truth certification', 'qa_calls': 0})
    with Path(args.input).open(encoding='utf8') as f: inputs = [json.loads(line) for line in f if line.strip()]
    count = facts = 0
    with (out / 'reports.jsonl').open('w', encoding='utf8') as stream:
        for doc, row in nlp.pipe(((r['text'], r) for r in inputs), as_tuples=True, batch_size=64, n_process=1):
            extra = {'reports': projection.project_reports(doc, row['text'], row['speaker'])} if row['kind'] == 'episode' else {
                'report_query': projection.project_report_query(doc, row['text'])}
            if 'reports' in extra: facts += len(extra['reports'])
            stream.write(json.dumps({**row, **extra}, ensure_ascii=False, sort_keys=True) + '\n'); stream.flush()
            count += 1
            if count % 1000 == 0: print(f'{count}/{len(inputs)} report inputs; {facts} numeric reports', flush=True)
    save('status.json', {'status': 'complete_projection', 'rows': count, 'expected_rows': len(inputs),
                         'quantity_reports': facts, 'qa_calls': 0}); print(f'Complete: {count} inputs; {facts} reports', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--input', required=True); p.add_argument('--output', required=True); a = p.parse_args()
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    main(a)
