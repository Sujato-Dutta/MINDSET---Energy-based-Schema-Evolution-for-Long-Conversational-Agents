"""Seal the integrated reconstruction comparison; no new inference."""
import os
from mindset.acl.pipeline import ROOT
from mindset.acl.io import read_json, write_json, digest_file


def run():
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('DGX Slurm only')
    folder = ROOT / 'outputs/acl/witness_closure_r01'
    status = read_json(folder / 'answers/status.json')
    if status['status'] != 'complete_development_answers' or status['completed_cells'] != 500:
        raise ValueError('Matched answers not complete')
    data = read_json(folder / 'answers/analysis.json')
    comparisons = {}
    for dataset in ('locomo', 'longmemeval'):
        comparisons[dataset] = {contrast: data['paired'][dataset]['witness_closed minus ' + contrast]
            for contrast in ('raw', 'calendar_only', 'closure_only', 'window_calendar')}
        if any(r['wins'] + r['ties'] + r['losses'] != 50 for r in comparisons[dataset].values()):
            raise ValueError('Unpaired results')
    supported = all(comparisons[d][control]['mean_difference'] > 0
                    for d in comparisons for control in ('raw', 'calendar_only', 'window_calendar'))
    result = dict(candidate='budgeted witness-context closure plus calendar constraints',
        decision='eligible_for_further_confirmation' if supported else 'reject_as_main_method',
        novel_methodology_locked=False, full_runs_authorized=False,
        comparisons=comparisons, observed_means=data['summary'],
        native_longmemeval_judging='deferred; secondary F1 is not native accuracy',
        usage=status, artifact_sha256={str(p.relative_to(folder)): digest_file(p) for p in
            (folder / 'audit.json', folder / 'retrieval.jsonl', folder / 'answers/audit.json',
             folder / 'answers/predictions.jsonl', folder / 'answers/analysis.json')},
        limitations=['Fixed development cohort repeatedly exposed, not a confirmation set',
            'Declared context dependencies are heuristic, not semantic proofs',
            'ISO temporal serialization may be penalized by native lexical F1; no metric rewriting applied',
            'Calendar-only and neighbouring-window controls were included rather than credited as novelty'],
        next_design_requirement='Do not require extra context solely from pronouns/reply markers. '
            'Preserve the strong raw packet and target answer-level date/set/arithmetic consistency with source witnesses. '
            'Do not call ordinary calendar parsing or program execution alone novel.')
    write_json(folder / 'method_decision.json', result, immutable=True)
    print(result['decision'], flush=True)


if __name__ == '__main__':
    run()
