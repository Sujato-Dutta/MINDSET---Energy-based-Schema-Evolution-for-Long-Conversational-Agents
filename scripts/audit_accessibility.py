"""Locate where the integrated write intervention ceases to affect readout."""
import os
from collections import Counter
from mindset.acl.pipeline import ROOT
from mindset.acl.io import read_json, unique_rows, write_json, digest_file


def run():
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('DGX Slurm only')
    out = ROOT / 'outputs/acl/accessibility_r01'
    if read_json(out / 'status.json')['status'] != 'complete_development':
        raise ValueError('Panel not terminal')
    rows = unique_rows(out / 'retrieval.jsonl')
    answers = read_json(out / 'answers/analysis.json')
    report = {}
    for dataset in ('locomo', 'longmemeval'):
        by = {a: {r['question_id']: r for r in rows.values() if r['dataset'] == dataset and r['method'] == a}
              for a in ('access_full', 'access_no_history', 'access_no_current', 'access_off')}
        baseline = by['access_off']; summary = {}
        for arm in ('access_full', 'access_no_history', 'access_no_current'):
            current = by[arm]
            if set(current) != set(baseline) or len(current) != 50:
                raise ValueError('Not the paired original100')
            transitions = Counter(); schemas = []; build = []; sample_counts = []
            for path in sorted((out / 'structure').glob(f'{dataset}-*-{arm}.json')):
                blob = read_json(path); telemetry = blob['telemetry']
                transitions['histories'] += 1
                transitions['episodes'] += telemetry['episode_count']
                transitions['local_write_interventions'] += telemetry['route_interventions']
                schemas.append(telemetry['schema_count']); build.append(blob['build_wall_seconds'])
                records = blob['decisions']
                off_path = path.with_name(path.name.replace('-' + arm + '.json', '-access_off.json'))
                off = read_json(off_path)['decisions']
                for eid in set(records) & set(off):
                    a = records[eid]['candidates'][records[eid]['selected']]
                    b = off[eid]['candidates'][off[eid]['selected']]
                    transitions['different_mutation_from_no_access'] += a['mutation'] != b['mutation']
                    transitions['different_target_from_no_access'] += a['schema_id'] != b['schema_id']
                sample_counts.extend(r['probe_count'] for r in records.values())
            summary[arm] = dict(**transitions,
                total_schemas=sum(schemas), quality_build_wall_seconds=sum(build),
                timing_scope='cached quality build, not paper efficiency',
                changed_seed_lists=sum(current[q]['seed_ids'] != baseline[q]['seed_ids'] for q in baseline),
                changed_candidate_lists=sum(current[q]['candidate_ids'] != baseline[q]['candidate_ids'] for q in baseline),
                changed_evidence_packets=sum(current[q]['evidence'] != baseline[q]['evidence'] for q in baseline),
                max_sample_size=max(sample_counts, default=0),
                matched_answer_delta=answers['paired'][dataset][arm.replace('access_no_history','access_full').replace('access_no_current','access_full') + ' minus access_off']['mean_difference']
                    if arm == 'access_full' else None)
        report[dataset] = summary
    write_json(out / 'causal_path_audit.json', dict(results=report,
        retrieval_sha256=digest_file(out / 'retrieval.jsonl'),
        answers_sha256=digest_file(out / 'answers/predictions.jsonl'),
        claim='write activity is not answer utility; inspect each causal stage',
        full_runs_authorized=False))
    print(report, flush=True)


if __name__ == '__main__':
    run()
