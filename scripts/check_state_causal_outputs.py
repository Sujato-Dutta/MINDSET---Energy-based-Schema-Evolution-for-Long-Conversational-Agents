"""Verify a complete constructed causal run; execute on DGX Slurm only."""
import argparse
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mindset.acl.io import read_json, unique_rows, digest_file, write_json
from probe_state_causality import requests, ARMS, TRACKS, MODES


def check(path):
    out = Path(path)
    histories = read_json(out / 'worlds.json')
    audit = read_json(out / 'audit.json'); status = read_json(out / 'status.json')
    assert audit['worlds_sha256'] == digest_file(out / 'worlds.json')
    assert audit['controller_sha256'] == digest_file(ROOT / 'src/mindset/methods/state_causal_probe.py')
    assert audit['worker_sha256'] == digest_file(ROOT / 'scripts/probe_state_causality.py')
    assert status['status'] == 'complete_controlled_probe' and status['qa_calls'] == status['judge_calls'] == 0
    rows = unique_rows(out / 'queries.jsonl'); operations = unique_rows(out / 'operations.jsonl')
    expected = sum(len(requests(h, i)) for h in histories for i, e in enumerate(h['events']) if e['speaker'] == 'Sam')
    assert len(rows) == expected * len(ARMS) * len(TRACKS) * len(MODES) == status['query_cells']
    assert len(operations) == sum(len(h['events']) for h in histories) * len(ARMS) * len(TRACKS)
    keyed = {(r['history'], r['prefix'], r['query_kind'], r['track'], r['arm'], r['score_mode']): r for r in rows.values()}
    pairs = 0; energy_decisions = 0
    for r in rows.values():
        assert len(r['selected']) <= 8 and len(set(r['selected'])) == len(r['selected'])
        assert len(r['candidate_ids']) <= 48 and set(r['selected']) <= set(r['candidate_ids'])
        reference = keyed[(r['history'], r['prefix'], r['query_kind'], TRACKS[0], 'energy', r['score_mode'])]
        assert r['candidate_ids'] == reference['candidate_ids']
    for r in operations.values():
        if r['arm'] in ('energy', 'read_off', 'replay_energy', 'replay_no_debt'):
            assert not r['energy_gap'] or r['energy_gap'] < 1e-9
            energy_decisions += 1
    for first in (h for h in histories if h['family'] == 'confirmation'):
        second = next(h for h in histories if h['family'] == 'correlated_noise'
                      and h['shared_words'] == first['shared_words'] and h['distractors'] == first['distractors'])
        assert first['events'] == second['events']
        assert first['truth_timeline'][-1] != second['truth_timeline'][-1]
        for i, e in enumerate(first['events']):
            if e['speaker'] != 'Sam': continue
            for req in requests(first, i):
                for track in TRACKS:
                    for arm in ARMS:
                        for mode in MODES:
                            a = keyed[(first['id'], i, req['kind'], track, arm, mode)]
                            b = keyed[(second['id'], i, req['kind'], track, arm, mode)]
                            assert a['question'] == b['question'] and a['candidate_ids'] == b['candidate_ids']
                            assert a['selected'] == b['selected'] and a['source_packet_sha256'] == b['source_packet_sha256']
                            pairs += 1
    result = {'valid': True, 'histories': len(histories), 'query_cells': len(rows),
              'operation_cells': len(operations), 'minimum_energy_decisions_checked': energy_decisions,
              'identical_observation_different_truth_pairs_checked': pairs,
              'constructed_histories_not_iid_statistical_samples': len(histories),
              'qa_calls': 0, 'judge_calls': 0, 'checker_sha256': digest_file(__file__)}
    write_json(out / 'verification.json', result, immutable=True)
    print(result, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True)
    args = p.parse_args()
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    check(args.output)
