"""Input-only grounding audit; no answers, gold IDs or retrieval scores used."""
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import os
import sys
import re
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mindset.acl.io import unique_rows, write_json, digest_file
from mindset.methods.factorized_state import atomic_features


def run(projections, output):
    sources = defaultdict(list); queries = []
    for row in unique_rows(Path(projections)).values():
        if row['kind'] == 'query': queries.append(row)
        else: sources[(row['dataset'], row['cluster'])].append(row)
    inventories = {}; reports = []
    for key, rows in sources.items():
        counts = Counter(); labels = {}; examples = {}; syntax = Counter()
        for r in rows:
            for f in atomic_features(r['speaker'], r['text']):
                for slot, value in f.attributes:
                    label = slot.removeprefix('relationship_').removeprefix('favorite_').replace('_', ' ')
                    counts[slot] += 1; labels[slot] = label
                    examples.setdefault(slot, []).append({'source_id': r['source_id'], 'speaker': r['speaker'], 'text': r['text']})
            for f in r['frames']: syntax[f['slot']] += 1
        inventories[key] = {'speakers': sorted({r['speaker'] for r in rows}), 'slots': dict(counts), 'syntax_slots': dict(syntax)}
        for q in (q for q in queries if (q['dataset'], q['cluster']) == key):
            terms = set(q['query']['terms']); matches = [s for s, label in labels.items() if set(label.split()) <= terms]
            projected = [f for r in rows for f in r['frames'] if set(f['match_terms']) <= terms]
            reports.append({'dataset': q['dataset'], 'cluster': q['cluster'], 'question_id': q['question_id'],
                'question': q['text'], 'question_date': q['question_date'], 'projection': q['query'],
                'speakers': inventories[key]['speakers'], 'bounded_slots': dict(counts),
                'literal_label_matches': matches, 'syntax_literal_matches': sorted({f['slot'] for f in projected}),
                'source_examples': {s: examples[s][:2] for s in matches},
                'self_reference_observable': bool(re.search(r'\b(?:I|my|me|mine)\b', q['text'], re.I))})
    summaries = []
    for d in ('locomo', 'longmemeval'):
        panel = [r for r in reports if r['dataset'] == d]
        summaries.append({'dataset': d, 'questions': len(panel),
            'with_self_reference': sum(r['self_reference_observable'] for r in panel),
            'with_literal_bounded_slot_match': sum(bool(r['literal_label_matches']) for r in panel),
            'with_literal_syntax_match': sum(bool(r['syntax_literal_matches']) for r in panel)})
    write_json(output, {'scope': 'input-only; NO answer/gold/metric-driven feature fitting',
        'source_sha256': digest_file(projections), 'worker_sha256': digest_file(__file__),
        'summary': summaries, 'questions': reports})
    print(summaries, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--projections', required=True); p.add_argument('--output', required=True)
    a = p.parse_args()
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    run(a.projections, a.output)
