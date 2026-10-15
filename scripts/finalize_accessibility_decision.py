"""Seal the matched routing-energy result, without generating new answers."""
import os
from mindset.acl.pipeline import ROOT
from mindset.acl.io import read_json, write_json, digest_file


def run():
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('DGX Slurm only')
    folder = ROOT / 'outputs/acl/routed_accessibility_r01'
    status = read_json(folder / 'status.json')
    analysis = read_json(folder / 'answers/analysis.json')
    usage = read_json(folder / 'answers/status.json')
    if status['status'] != 'complete_development' or status['completed_cells'] != 500:
        raise ValueError('Retrieval is incomplete')
    if usage['status'] != 'complete_development_answers' or usage['completed_cells'] != 500:
        raise ValueError('Answer panel is incomplete')
    expected_metrics = {'locomo': 'native_f1', 'longmemeval': 'token_f1'}
    evidence = {}
    for dataset, metric in expected_metrics.items():
        comparison = analysis['paired'][dataset]['access_full minus access_off']
        if comparison['metric'] != metric:
            raise ValueError('Metric contract changed')
        if comparison['wins'] + comparison['ties'] + comparison['losses'] != 50:
            raise ValueError('Unpaired or incomplete evaluation')
        evidence[dataset] = dict(metric=metric, energy_on_off=comparison,
            current_term_ablation=analysis['paired'][dataset]['access_full minus access_no_current'])
    supported = all(evidence[d]['energy_on_off']['mean_difference'] > 0 for d in evidence)
    decision = dict(candidate='source-accessibility energy with unmasked root RRF',
        decision='eligible_for_further_confirmation' if supported else 'reject_as_paper_central_claim',
        scope='Exact exposed development100; not a universal impossibility result',
        evidence=evidence, native_longmemeval_accuracy=None,
        native_longmemeval_judging='deferred_by_user; secondary token F1 does not substitute',
        energy_specific_answer_gain_supported=supported,
        novel_methodology_locked=False, full_runs_authorized=False,
        fresh_local_answer_calls=usage['fresh_recorded_calls'],
        fresh_local_input_tokens=usage['fresh_recorded_input_tokens'],
        fresh_local_output_tokens=usage['fresh_recorded_output_tokens'],
        artifact_sha256={str(p.relative_to(folder)): digest_file(p) for p in
            (folder / 'audit.json', folder / 'retrieval.jsonl', folder / 'answers/audit.json',
             folder / 'answers/predictions.jsonl', folder / 'answers/analysis.json')},
        next_method_requirement='A new mechanism must affect answer-relevant source reconstruction, not just centroid routing. '
            'Require a matched energy-off comparison before promotion; no more routing-weight or probe-count sweeps.',
        prohibited_claims=['SOTA', 'guaranteed A* acceptance', 'energy improves answer F1',
                           'all components useful', 'native LongMemEval accuracy measured'])
    write_json(folder / 'method_decision.json', decision, immutable=True)
    print({k: decision[k] for k in ('decision', 'energy_specific_answer_gain_supported',
                                  'novel_methodology_locked', 'fresh_local_answer_calls')}, flush=True)


if __name__ == '__main__':
    run()
