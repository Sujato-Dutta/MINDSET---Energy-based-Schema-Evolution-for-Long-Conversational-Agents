"""Independent source-state readiness audit; run only in a DGX CPU allocation.

No model, QA or action-accuracy claim. Literal frame extraction and correct
binary compatibility need not encode enough information for state transitions.
"""
import os
import re

from mindset.acl.pipeline import ROOT, config, record_environment
from mindset.acl.io import read_json, write_json, digest_file
from probe_scope_instrument_controls import verified_parent
from stress_source_semantics import fixtures
from source_scope_confirmation_fixtures import confirmation_fixtures

NEGATION = re.compile(r"\b(?:not|never|no)\b|n't\b", re.IGNORECASE)
CORE = ('holder_quote', 'predicate_quote', 'value_quote')


def readiness(rows):
    accepted = [r for r in rows if r['validation_error'] is None]
    missing = []; absent_negation = []; aliases = []
    compatible_families = set()
    for row in accepted:
        frames = {f['source_id']: f for f in row['prediction']['frames']}
        if row['prediction']['relationship'] == 'compatible':
            compatible_families.add(row['family'])
        for sid, text in (('A', row['left']), ('B', row['right'])):
            f = frames[sid]
            absent = [k for k in CORE if not f[k]]
            if absent:
                missing.append({'id': row['id'], 'source_id': sid, 'missing': absent})
            if NEGATION.search(text) and not NEGATION.search(' '.join(f[k] for k in CORE)):
                absent_negation.append({'id': row['id'], 'source_id': sid, 'text': text,
                    'core_quotes': {k: f[k] for k in CORE},
                    'warning': 'literal English negation absent from core quotes; not a semantic truth test'})
        if row['expected_contradiction'] and all(frames['A'][k] == frames['B'][k] for k in CORE):
            aliases.append({'id': row['id'], 'core_quotes': {k: frames['A'][k] for k in CORE},
                'remaining_labels': [{k: frames[s][k] for k in ('canonical_relation', 'modality', 'value_mode')}
                    for s in ('A', 'B')],
                'warning': 'core quote tuple alone aliases a conflict control; complete raw sources remain distinct'})
    return {'cases': len(rows), 'accepted_cases': len(accepted),
        'rejected_cases': len(rows) - len(accepted),
        'accepted_frames_with_missing_core_fields': len(missing),
        'literal_negation_absent_from_core_quote_count': len(absent_negation),
        'conflict_control_core_quote_alias_count': len(aliases),
        'accepted_compatible_fixture_families': sorted(compatible_families),
        'missing_core_frames': missing, 'negation_diagnostics': absent_negation,
        'conflict_core_quote_aliases': aliases,
        'scope': 'representation readiness only; not semantic label accuracy, natural benchmark coverage or novel controller evidence',
        'raw_source_lossless': True, 'no_automatic_world_state_updates': True,
        'full_runs_authorized': False}


def run():
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('DGX Slurm only')
    cfg = config(); checkpoint = read_json(ROOT / 'configs/checkpoints.lock.json')[cfg['primary_generator']]
    cases = [{**c, 'split': 'development'} for c in fixtures()] + [
        {**c, 'split': 'lexical_confirmation'} for c in confirmation_fixtures()]
    parent, hashes = verified_parent(cases, checkpoint, cfg)
    out = ROOT / 'outputs/acl' / f'scope_state_readiness_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True, exist_ok=True); record_environment(out)
    report = readiness(list(parent.values()))
    report['parent_hashes'] = hashes
    report['worker_sha256'] = digest_file(__file__)
    write_json(out / 'analysis.json', report, immutable=True)
    print('READINESS', {k: v for k, v in report.items() if k not in (
        'missing_core_frames', 'negation_diagnostics', 'conflict_core_quote_aliases')}, flush=True)


if __name__ == '__main__':
    run()
