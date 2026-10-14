"""Integrity audit and variable-history threshold diagnostic; DGX Slurm only."""
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import os
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from probe_mdl_energy import ARMS, TRACKS, MODES
from probe_state_causality import requests
from mindset.acl.io import read_json, unique_rows, digest_file, write_json
from mindset.acl.pipeline import write_csv
from mindset.methods.mdl_state import ObservedMDLEnergyMemory
from mindset.methods.compact import CompactConfig
from mindset.methods.base import ConversationEvent


class FixedVector:
    """A controlled constant vector, not an encoder-quality experiment."""
    def encode(self, texts, **kwargs): return np.ones((len(texts), 8), dtype=np.float32) / np.sqrt(8)


def thresholds(out):
    table = []
    for initial in (1, 2, 4, 8, 16, 32, 64, 128):
        for trajectory in ('repeated_report', 'two_report_burst_then_return'):
            values = ['Cedar'] * initial + (['Willow'] * 5 if trajectory == 'repeated_report' else ['Willow'] * 2 + ['Cedar'] * 3)
            events = [ConversationEvent(f'e{i:04d}', 'threshold-only', f's{i:04d}',
                (datetime(2031, 1, 1) + timedelta(days=i)).strftime('%Y-%m-%d'), 'Sam', f'I live in {v} at home.') for i, v in enumerate(values)]
            observations = {e.event_id: {'speaker': 'Sam', 'text': e.text, 'slot': 'location',
                'value': v, 'contexts': ['at home'], 'explicit_change': False} for e, v in zip(events, values)}
            for policy in ('energy', 'two_reports'):
                m = ObservedMDLEnergyMemory(CompactConfig(complexity_penalty=.1, switch_margin=.05),
                    observations=observations, scopes={}, encoder=FixedVector(), mdl_policy=policy)
                for i, e in enumerate(events):
                    m.ingest([e]); m.validate_invariants()
                    if i < initial: continue
                    record = m.operation_log.all()[-1]; cert = m.observation_audit()[-1]
                    table.append({'initial_support': initial, 'trajectory': trajectory, 'policy': policy,
                        'post_initial_report': i - initial + 1, 'asserted_value': values[i], 'operation': record.operation,
                        'active_mode': next(p['mode'] for sid, p in m._mdl_posteriors.items() if m.schemas[sid].status == 'active'),
                        'selected_energy': cert['selected_adjusted_energy'], 'optimality_gap': cert['optimality_gap'],
                        'cut_session': cert['chosen_cut_session'], 'active_schemas': sum(s.status == 'active' for s in m.schemas.values()),
                        'metric_scope': 'observation-only threshold behavior, no truth labels, no QA/latency claim'})
    write_csv(out / 'variable_history_thresholds.csv', table)
    for initial in (1, 2, 4, 8, 16, 32, 64, 128):
        for trajectory in ('repeated_report', 'two_report_burst_then_return'):
            signatures = {p: [(r['operation'], r['active_mode']) for r in table if r['initial_support'] == initial
                and r['trajectory'] == trajectory and r['policy'] == p] for p in ('energy', 'two_reports')}
            print(f'initial={initial} {trajectory}: energy={signatures["energy"]}; two_reports={signatures["two_reports"]}', flush=True)


def check(path):
    out = Path(path); audit = read_json(out / 'audit.json'); histories = read_json(out / 'worlds.json')
    assert read_json(out / 'status.json')['status'] == 'complete_controlled_probe'
    assert audit['controller_sha256'] == digest_file(ROOT / 'src/mindset/methods/mdl_state.py')
    assert audit['worker_sha256'] == digest_file(ROOT / 'scripts/probe_mdl_energy.py')
    assert audit['worlds_sha256'] == digest_file(out / 'worlds.json')
    rows = unique_rows(out / 'queries.jsonl'); ops = unique_rows(out / 'operations.jsonl')
    n = sum(len(requests(h, i)) for h in histories for i, e in enumerate(h['events']) if e['speaker'] == 'Sam')
    assert len(rows) == n * len(ARMS) * len(TRACKS) * len(MODES)
    assert len(ops) == sum(len(h['events']) for h in histories) * len(ARMS) * len(TRACKS)
    keyed = {(r['history'], r['prefix'], r['query_kind'], r['track'], r['arm'], r['score_mode']): r for r in rows.values()}
    checked = 0
    for r in rows.values():
        assert len(r['selected']) <= 8 and len(set(r['selected'])) == len(r['selected'])
        assert len(r['candidate_ids']) <= 48 and set(r['selected']) <= set(r['candidate_ids'])
        baseline = keyed[(r['history'], r['prefix'], r['query_kind'], TRACKS[0], 'energy', r['score_mode'])]
        assert r['candidate_ids'] == baseline['candidate_ids']
    for r in ops.values():
        if r['arm'] in ('mdl', 'mdl_no_boundary', 'mdl_no_likelihood', 'mdl_read_off'):
            assert (r['energy_gap'] or 0) < 1e-9
    for a in (h for h in histories if h['family'] == 'confirmation'):
        b = next(h for h in histories if h['family'] == 'correlated_noise'
                 and h['shared_words'] == a['shared_words'] and h['distractors'] == a['distractors'])
        assert a['events'] == b['events']
        for i, e in enumerate(a['events']):
            if e['speaker'] != 'Sam': continue
            for req in requests(a, i):
                for track in TRACKS:
                    for arm in ARMS:
                        for mode in MODES:
                            x = keyed[(a['id'], i, req['kind'], track, arm, mode)]
                            y = keyed[(b['id'], i, req['kind'], track, arm, mode)]
                            assert x['selected'] == y['selected']
                            checked += 1
    result = {'valid': True, 'histories': len(histories), 'query_cells': len(rows), 'operation_cells': len(ops),
              'identical_observation_different_truth_pairs': checked, 'qa_calls': 0, 'judge_calls': 0,
              'checker_sha256': digest_file(__file__), 'not_a_natural_QA_or_novelty_proof': True}
    write_json(out / 'verification.json', result, immutable=True); print(result, flush=True)
    thresholds(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True); a = p.parse_args()
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    check(a.output)
