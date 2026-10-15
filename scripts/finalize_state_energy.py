"""Seal the actual synthetic candidate decision; Slurm only, no model calls."""
import os
from mindset.acl.pipeline import ROOT
from mindset.acl.io import read_json, write_json, digest_file


if __name__ == '__main__':
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('DGX Slurm only')
    out = ROOT / 'outputs/acl/grounded_state_energy_r02'
    status = read_json(out / 'status.json')
    if status['status'] != 'complete_synthetic_preflight':
        raise ValueError('Preflight not complete')
    reports = read_json(out / 'synthetic_results.json')
    holdout = {r['mode']: r['mean_read_error'] for r in reports if r['split'] == 'holdout'}
    if set(holdout) != {'joint', 'rules', 'local', 'hard_grounding'}:
        raise ValueError('Missing comparator')
    decision = dict(
        decision='reject_as_paper_central_method',
        reason='Joint coupling does not beat separately trained local energy or same-grounder rules on the independent synthetic check.',
        heldout_read_error=holdout,
        lower_is_better=True,
        scope='short synthetic state reads, NOT LoCoMo F1, LME judge accuracy, natural QA, or novelty proof',
        sources={p: digest_file(out / p) for p in ('audit.json', 'status.json', 'synthetic_results.json',
            'synthetic.json', 'syntax.json', 'model.json', 'local_model.json', 'training_records.json', 'tests.xml')},
        joint_improves_over_hard_grounding_only=holdout['joint'] < holdout['hard_grounding'],
        energy_specific_advantage_established=False,
        natural_qa_ran=False, hosted_calls=0, full_runs_authorized=False,
        worker_sha256=digest_file(__file__), job_id=os.environ['SLURM_JOB_ID'])
    write_json(out / 'method_decision.json', decision, immutable=True)
    print(decision, flush=True)
