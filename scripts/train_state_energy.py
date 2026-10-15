"""DGX-only bounded learned methodology preflight, no hosted/answer calls."""
import json
import os
import subprocess
from pathlib import Path
from mindset.acl.pipeline import ROOT, record_environment
from mindset.acl.io import read_json, write_json, digest_file
from mindset.acl.state_energy_training import corpus, train, evaluate
from mindset.memory.grounded_state_energy import fit_risk


def run():
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('DGX Slurm only')
    name = os.environ.get('MINDSET_STATE_ENERGY_OUTPUT', 'grounded_state_energy_r02')
    if not name.startswith('grounded_state_energy_r') or '/' in name or '\\' in name:
        raise ValueError('Isolated experiment name required')
    out = ROOT / 'outputs/acl' / name; out.mkdir(parents=True, exist_ok=True)
    sources = ('src/mindset/memory/grounded_state_energy.py',
               'src/mindset/acl/state_energy_training.py', 'scripts/parse_state_bindings.py',
               'scripts/train_state_energy.py')
    import fcntl
    with (out / 'worker.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(out / 'audit.json', dict(sources={p: digest_file(ROOT / p) for p in sources},
            manifest_sha256=digest_file(ROOT / 'data/manifests/mindset_v2_development100.json'),
            scope='independent synthetic methodology preflight; no natural answer claims',
            benchmark_reference_use=False, ridge=1., no_hyperparameter_search=True,
            hosted_calls=0, full_runs_authorized=False), immutable=True)
        record_environment(out / 'continuations' / os.environ['SLURM_JOB_ID'])
        data = corpus(); write_json(out / 'synthetic.json', data, immutable=True)
        inputs = [s for h in data['histories'] for s in h['sources']]
        if not (out / 'syntax.json').exists():
            p = subprocess.run([str(ROOT / '.venv-syntax/bin/python'), str(ROOT / 'scripts/parse_state_bindings.py')],
                               input=json.dumps(inputs), text=True, capture_output=True)
            if p.returncode:
                raise RuntimeError(p.stderr)
            write_json(out / 'syntax.json', json.loads(p.stdout), immutable=True)
        parsed = read_json(out / 'syntax.json')['sources']
        model, records = train(data, parsed)
        local_model = fit_risk(records, local=True)
        write_json(out / 'training_records.json', records, immutable=True)
        write_json(out / 'model.json', {**model, 'synthetic_sha256': digest_file(out / 'synthetic.json'),
            'parser_sha256': read_json(out / 'syntax.json')['parser_sha256'],
            'training_records_sha256': digest_file(out / 'training_records.json')}, immutable=True)
        write_json(out / 'local_model.json', local_model, immutable=True)
        reports = [evaluate(data, parsed, local_model if mode == 'local' else model, split, mode) for split in ('validation', 'holdout')
                   for mode in ('joint', 'rules', 'local', 'hard_grounding')]
        write_json(out / 'synthetic_results.json', reports, immutable=True)
        holdout = {r['mode']: r for r in reports if r['split'] == 'holdout'}
        beats_rules = holdout['joint']['mean_read_error'] < holdout['rules']['mean_read_error'] - 1e-12
        beats_local = holdout['joint']['mean_read_error'] < holdout['local']['mean_read_error'] - 1e-12
        write_json(out / 'status.json', dict(status='complete_synthetic_preflight',
            joint_beats_same_grounder_rules=beats_rules, joint_beats_local_energy=beats_local,
            natural_qa_validated=False, novelty_validated=False,
            decision='eligible_for_natural_pilot' if beats_rules and beats_local else 'do_not_promote_from_synthetic_results',
            qa_calls=0, judge_calls=0, full_runs_authorized=False))
        print(json.dumps(read_json(out / 'status.json')), flush=True)
        print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    run()
