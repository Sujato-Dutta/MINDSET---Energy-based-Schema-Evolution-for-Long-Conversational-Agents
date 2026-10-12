"""Independent, generated functional probes; not natural OOD validation/tuning."""
from pathlib import Path
import argparse
import os
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from test_state_readout import build, ARMS, identity
from mindset.methods.base import ConversationEvent
from mindset.acl.joint_iteration import protocol
from mindset.acl.pipeline import local_stack, record_environment, write_csv
from mindset.acl.io import write_json, append_row
from mindset.acl.runtime import CachedEncoder


def cases():
    result = []
    for owner, first, second in (
        ('Avery', 'Oslo', 'Lisbon'), ('Blair', 'Kyoto', 'Dakar'),
        ('Casey', 'Helsinki', 'Lima'), ('Drew', 'Athens', 'Seoul'),
        ('Emery', 'Nairobi', 'Prague'), ('Finley', 'Zurich', 'Accra'),
    ):
        scenarios = (
            ('current', [f'I live in {first}.', f'I moved to {second}.'],
             f'Where does {owner} live now?', ['e2']),
            ('historical', [f'I live in {first}.', f'I moved to {second}.'],
             f'Where did {owner} live on 2025-01-01?', ['e1']),
            ('context', ['I prefer tea at home.', 'I prefer coffee at work.'],
             f'What does {owner} prefer at work?', ['e2']),
            ('noise', [f'I live in {first}.', f'I live in {second}.'],
             f'Where does {owner} live now?', ['e1']),
            ('confirmation', [f'I live in {first}.', f'I live in {second}.', f'I live in {second}.'],
             f'Where does {owner} live now?', ['e3']),
            ('event', [f'I live in {first}.', f'I moved to {second}.'],
             f'When did {owner} move?', ['e1', 'e2']),
            ('unknown_owner', [f'I live in {first}.', f'I moved to {second}.'],
             'Where does Morgan live now?', ['e1', 'e2']),
            ('multi_period', [f'I live in {first}.', f'I moved to {second}.'],
             f'Where did {owner} live in 2024 and 2025?', ['e1', 'e2']),
        )
        for kind, texts, question, required in scenarios:
            result.append({'id': f'{owner}/{kind}', 'kind': kind, 'speaker': owner,
                           'texts': texts, 'question': question, 'required': required})
    return result


def run(calibration, output):
    cfg, _, artifact = protocol(ROOT / 'data/manifests/mindset_v2_development100.json', calibration)
    out = Path(output)
    if out.exists() and any(out.iterdir()): raise ValueError('Fresh controlled-probe directory required')
    out.mkdir(parents=True); generated = cases()
    write_json(out / 'cases.json', generated, immutable=True)
    write_json(out / 'audit.json', {**identity(ROOT / 'data/manifests/mindset_v2_development100.json', calibration,
                                             cfg, artifact, 'generated_functional_checks_not_ood_benchmark'),
        'conditions': '48 explicit English template probes, overlapping extraction grammar; no fitting',
        'scoring_modes': ['actual_pinned_ce', 'adversarial_older_source_first'],
        'adversarial_scope': 'diagnostic deterministic scores, not a learned model or natural QA metric',
        'candidate_budget': 'all 2-3 sources, <=48; no padding or invented evidence'})
    record_environment(out)
    encoder, reranker = local_stack(cfg, cache=False); encoder = CachedEncoder(encoder, enabled=True)
    rows = []
    for case in generated:
        events = [ConversationEvent(f'e{i}', case['id'], f's{i}', f'2025-01-{i:02d}', case['speaker'], text)
                  for i, text in enumerate(case['texts'], 1)]
        memories = {a: build(a, events, cfg, artifact['selected_parameters'], encoder, reranker) for a in ARMS}
        base = memories['full']; pool = list(range(len(events)))
        actual = np.asarray(reranker.predict([(case['question'], text) for text in base._documents],
                                            batch_size=8, show_progress_bar=False), dtype=float)
        for mode, scores in (('actual_pinned_ce', actual),
                             ('adversarial_older_source_first', np.arange(len(events), 0, -1, dtype=float))):
            for arm, m in memories.items():
                selected = [h.memory_id for h in m.select_scored(case['question'], pool, scores, 8)]
                required = set(case['required'])
                row = {'id': f"{case['id']}/{mode}/{arm}", 'case_id': case['id'], 'kind': case['kind'],
                    'arm': arm, 'score_mode': mode, 'required': case['required'], 'selected': selected,
                    'required_source_recall': len(required & set(selected)) / len(required),
                    'exact_scoped_packet': set(selected) == required,
                    'top1_expected_state': selected[0] in required if len(required) == 1 else None,
                    'certificate': m.last_readout_certificate}
                rows.append(row); append_row(out / 'results.jsonl', row)
        encoder.cache.clear()
    report = []
    for mode in ('actual_pinned_ce', 'adversarial_older_source_first'):
        for kind in sorted({c['kind'] for c in generated}):
            for arm in ARMS:
                subset = [r for r in rows if r['score_mode'] == mode and r['kind'] == kind and r['arm'] == arm]
                report.append({'score_mode': mode, 'kind': kind, 'arm': arm, 'cases': len(subset),
                    'exact_scoped_packet_rate': float(np.mean([r['exact_scoped_packet'] for r in subset])),
                    'required_source_recall': float(np.mean([r['required_source_recall'] for r in subset])),
                    'top1_expected_state': float(np.mean([r['top1_expected_state'] for r in subset]))
                        if subset[0]['top1_expected_state'] is not None else None})
    write_csv(out / 'summary.csv', report)
    write_json(out / 'status.json', {'status': 'complete_functional_probes', 'cells': len(rows),
        'expected_cells': 480, 'qa_calls': 0, 'judge_calls': 0, 'natural_state_accuracy': None,
        'scope': 'generated functional probes only; not benchmark QA or natural OOD validation'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    run(args.calibration, args.output)
