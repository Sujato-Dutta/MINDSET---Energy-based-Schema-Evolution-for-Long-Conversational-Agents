"""Independent artifact replay and mechanism audit for the fixed readout pilot."""
import json
import os
from pathlib import Path
import numpy as np
from mindset.acl.pipeline import ROOT
from mindset.acl.io import read_json, write_json, unique_rows, digest_file
from mindset.memory.invariant_readout import registry, readout, CONTRACT_HASH
from mindset.acl.clients import messages, ANSWER_PROMPT_HASH
from mindset.acl.locomo_protocol_diagnostic import short_phrase_messages, READER_HASH
from mindset.evaluation.metrics import token_f1
from mindset.evaluation.locomo import locomo_native_score
from mindset.datasets.locomo import CATEGORY_NAMES


def run():
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('DGX Slurm only')
    out = ROOT / 'outputs/acl' / os.environ.get('MINDSET_INVARIANT_OUTPUT', 'invariant_readout_r02')
    status = read_json(out / 'status.json'); audit = read_json(out / 'audit.json')
    if status['status'] != 'complete_development_answers' or audit['contract_hash'] != CONTRACT_HASH:
        raise ValueError('Incomplete or changed pilot')
    if audit['module_sha256'] != digest_file(ROOT / 'src/mindset/memory/invariant_readout.py'):
        raise ValueError('Method changed since run')
    predictions = unique_rows(out / 'predictions.jsonl')
    measurements = unique_rows(out / 'measurements.jsonl')
    if len(predictions) != 400 or len(measurements) != 100:
        raise ValueError('Wrong sample/cell count')
    # All readout programs must replay without reference answers.
    for key, m in measurements.items():
        row = predictions[f'{m["dataset"]}/invariant/{m["question_id"]}']
        reg = registry(row['selected'])
        if reg != m['registry']:
            raise ValueError('Source registry differs')
        if m['execution']['status'] != 'invalid_program':
            replay = readout(m['payload'], reg)
            if replay != m['execution']:
                raise ValueError('Program replay mismatch')
        arms = dict(raw_reference=None, same_response_direct=m['payload'].get('answer', '') if isinstance(m['payload'], dict) else '',
                    first_binding=m['execution']['first_binding_answer'], invariant=m['execution']['answer'])
        for arm, expected in arms.items():
            r = predictions[f'{m["dataset"]}/{arm}/{m["question_id"]}']
            if expected is not None and expected != r['prediction']:
                raise ValueError('Prediction/control mismatch')
            if r['selected'] != row['selected'] or r['evidence'] != row['evidence'] or r['retrieval'] != row['retrieval']:
                raise ValueError('Matched evidence/retrieval control violated')
            if abs(token_f1(r['prediction'], r['answer']) - r['token_f1']) > 1e-12:
                raise ValueError('Secondary score mismatch')
            if r['dataset'] == 'locomo':
                category = next(int(k) for k, v in CATEGORY_NAMES.items() if v == r['category'])
                if abs(locomo_native_score(r['prediction'], r['answer'], category) - r['native_f1']) > 1e-12:
                    raise ValueError('Native score mismatch')
    paired = read_json(out / 'analysis.json')['paired']
    gate_changes = [m['id'] for m in measurements.values()
                    if m['execution']['answer'] != m['execution']['first_binding_answer']]
    nontrivial = [m['id'] for m in measurements.values() if m['execution']['worlds'] > 1]
    jointly_better = all(paired[d]['invariant minus same_response_direct']['mean_difference'] > 0
                        for d in ('locomo', 'longmemeval'))
    gate_benefit = any(paired[d]['invariant minus first_binding']['mean_difference'] > 0
                      for d in ('locomo', 'longmemeval'))
    outcome = dict(
        status='verified_terminal_development',
        decision='requires_independent_confirmation_not_final_method' if jointly_better and gate_benefit else 'do_not_promote_as_uncertainty_method',
        replayed_programs=len(measurements), recomputed_prediction_scores=len(predictions),
        paired=paired, nontrivial_binding_sets=nontrivial, gate_changed_answers=gate_changes,
        gating_specific_benefit_observed=gate_benefit,
        exact_execution_not_semantic_truth=True,
        novelty_certified=False, full_runs_authorized=False,
        frozen_artifacts={p: digest_file(out / p) for p in ('audit.json', 'status.json', 'measurements.jsonl',
            'predictions.jsonl', 'comparison.csv', 'analysis.json')},
        dependencies={p: digest_file(ROOT / p) for p in ('src/mindset/memory/witness_closure.py',
            'src/mindset/memory/report_frames.py', 'src/mindset/memory/invariant_readout.py',
            'src/mindset/evaluation/locomo.py', 'src/mindset/evaluation/metrics.py')},
        scope='exposed development100; cannot establish independent generalization or A* acceptance',
        job_id=os.environ['SLURM_JOB_ID'], worker_sha256=digest_file(__file__))
    write_json(out / 'method_decision.json', outcome, immutable=True)
    print(json.dumps(outcome), flush=True)


if __name__ == '__main__':
    run()
