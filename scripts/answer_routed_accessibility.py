"""Only generate answers when the energy-on/off contrast reaches the reader."""
import argparse
import os
from mindset.acl.matched_source_answers import run_panel
from mindset.acl.pipeline import ROOT
from mindset.acl.io import read_json, unique_rows, write_json, digest_file

ARMS = ('access_full', 'access_no_history', 'access_no_current', 'access_off', 'flat_expanded')
CONTRASTS = (('access_full', 'access_off'), ('access_full', 'access_no_history'),
             ('access_full', 'access_no_current'), ('access_full', 'flat_expanded'))


def gate(prepared):
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('DGX Slurm only')
    status = read_json(prepared / 'status.json')
    if status['status'] != 'complete_development' or status['completed_cells'] != 500:
        raise ValueError('Retrieval panel is not complete')
    rows = unique_rows(prepared / 'retrieval.jsonl')
    changed = {}
    for dataset in ('locomo', 'longmemeval'):
        by = {a: {r['question_id']: r for r in rows.values() if r['dataset'] == dataset and r['method'] == a}
              for a in ARMS}
        if any(len(v) != 50 or set(v) != set(by['access_full']) for v in by.values()):
            raise ValueError('Unpaired original100')
        changed[dataset] = [q for q in sorted(by['access_full'])
            if by['access_full'][q]['evidence'] != by['access_off'][q]['evidence']]
    proceed = any(changed.values())
    report = dict(allow_answer_generation=proceed, changed_on_off_questions=changed,
        retrieval_sha256=digest_file(prepared / 'retrieval.jsonl'),
        reason='Energy changes at least one reader input; measure QA honestly' if proceed else
               'Energy changes no reader input; answer improvement impossible under the identical reader',
        significance_or_positive_gain_claim=False, full_runs_authorized=False)
    write_json(prepared / 'answer_gate.json', report, immutable=True)
    print(report, flush=True)
    return proceed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate-only', action='store_true')
    args = parser.parse_args()
    prepared = ROOT / 'outputs/acl/routed_accessibility_r01'
    if not gate(prepared):
        raise SystemExit(10 if args.gate_only else 0)
    if not args.gate_only:
        run_panel(prepared, prepared / 'answers', ARMS, CONTRASTS,
            'unmasked root-routing pilot; exposed original100; no novel-method or significance claim',
            ROOT / 'outputs/acl/accessibility_r01/answers')
