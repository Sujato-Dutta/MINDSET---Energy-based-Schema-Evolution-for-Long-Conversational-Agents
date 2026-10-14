"""Frozen original100 source-relationship coverage; no QA or world-state updates."""
import json
import os
from pathlib import Path
from collections import Counter

from mindset.acl.pipeline import ROOT, config, server_client, request_json, record_environment
from mindset.acl.joint_iteration import protocol, groups_for
from mindset.acl.io import read_json, write_json, unique_rows, append_row, digest_file
from mindset.acl.clients import prompt_tokens
from mindset.acl.source_relation_validation import verify_prepared_pair
from mindset.memory.source_relation_contract import SCHEMA, CONTRACT_HASH, relation_messages, validate_relation
from mindset.utils.hashing import stable_hash


def run():
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('DGX Slurm only')
    manifest = ROOT / 'data/manifests/mindset_v2_development100.json'
    calibration = ROOT / 'outputs/acl/mindset_joint_v2_27058/calibration/calibration.json'
    cfg, cohort, _ = protocol(manifest, calibration)
    checkpoint = read_json(ROOT / 'configs/checkpoints.lock.json')[cfg['primary_generator']]
    parent = ROOT / 'outputs/acl/natural_source_relations_r01'
    if read_json(parent / 'status.json')['status'] != 'complete_inputs':
        raise ValueError('Natural source inputs not complete')
    prepared_audit = read_json(parent / 'audit.json')
    if any(prepared_audit.get(k) != v for k, v in {
        'manifest_sha256': digest_file(manifest), 'calibration_sha256': digest_file(calibration),
        'source_module_sha256': digest_file(ROOT / 'src/mindset/methods/semantic_witness.py'),
        'candidate_module_sha256': digest_file(ROOT / 'src/mindset/acl/source_relation_candidates.py'),
        'worker_sha256': digest_file(ROOT / 'scripts/prepare_natural_source_relations.py')}.items()):
        raise ValueError('Natural preparation identity changed')
    synthetic = ROOT / 'outputs/acl/source_relation_instrument_r01'
    sa = read_json(synthetic / 'audit.json')
    if read_json(synthetic / 'status.json')['status'] != 'complete_synthetic' or any(sa.get(k) != v for k, v in {
        'contract_hash': CONTRACT_HASH, 'checkpoint': checkpoint, 'seed': cfg['seed'],
        'contract_module_sha256': digest_file(ROOT / 'src/mindset/memory/source_relation_contract.py'),
        'client_module_sha256': digest_file(ROOT / 'src/mindset/acl/clients.py'),
        'model_config_sha256': digest_file(ROOT / 'configs/models.yaml')}.items()):
        raise ValueError('Previously tested relation identity changed')
    # Passing synthetic checks is NOT a semantic truth certificate: errors are
    # retained in that report. This worker measures coverage, not state mutation.
    inputs = unique_rows(parent / 'queries.jsonl')
    expected_queries = {f'{r["dataset"]}/{r["id"]}' for r in cohort['records']}
    if set(inputs) != expected_queries:
        raise ValueError('Original100 input panel changed')
    unique = {}
    for dataset, cluster, cases in groups_for(cohort, cfg):
        for case in cases:
            row = inputs[f'{dataset}/{case.id}']
            if row['cluster'] != cluster or len(row['measurement_pairs']) > 4:
                raise ValueError('Changed source query/budget')
            for pair in row['measurement_pairs']:
                verify_prepared_pair(pair, case.events, question_date=case.question_date)
                if pair['pair_key'] in unique and unique[pair['pair_key']] != pair:
                    raise ValueError('Pair identity collision')
                unique[pair['pair_key']] = pair
    if sorted(unique.values(), key=lambda p: p['pair_key']) != read_json(parent / 'unique_pairs.json'):
        raise ValueError('Prepared unique source pairs changed')
    if len(unique) > 400:
        raise ValueError('Bounded instrument budget exceeded')
    out = parent / 'measurements'; out.mkdir(parents=True, exist_ok=True)
    identity = {'manifest_sha256': digest_file(manifest), 'calibration_sha256': digest_file(calibration),
        'source_audit_sha256': digest_file(parent / 'audit.json'),
        'source_queries_sha256': digest_file(parent / 'queries.jsonl'),
        'source_pairs_sha256': digest_file(parent / 'unique_pairs.json'),
        'synthetic_audit_sha256': digest_file(synthetic / 'audit.json'),
        'synthetic_analysis_sha256': digest_file(synthetic / 'analysis.json'),
        'checkpoint': checkpoint, 'seed': cfg['seed'], 'contract_hash': CONTRACT_HASH,
        'worker_sha256': digest_file(__file__), 'max_output_tokens': 96,
        'source_validation_sha256': digest_file(ROOT / 'src/mindset/acl/source_relation_validation.py'),
        'no_question_gold_answer_category_in_model_inputs': True,
        'no_energy_selection_or_world_updates': True,
        'scope': 'original100 provisional source-relationship coverage; no natural semantic-accuracy or QA claim',
        'full_runs_authorized': False}
    import fcntl
    with (out / 'worker.lock').open('a+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(out / 'audit.json', identity, immutable=True)
        record_environment(out / 'continuations' / os.environ['SLURM_JOB_ID'])
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint['checkpoint'], revision=checkpoint['revision'],
            cache_dir=str(Path.home() / 'MINDSET_v2/.cache/gpu_prototype/huggingface/hub'), local_files_only=True)
        url = os.environ.get('MINDSET_LOCAL_BASE_URL', '')
        if url.split('://')[-1].split(':')[0] not in ('localhost', '127.0.0.1'):
            raise ValueError('Local-only model required')
        info = request_json(url.rstrip('/') + '/models')
        if {r['id'] for r in info['data']} != {checkpoint['checkpoint']}:
            raise ValueError('Wrong served model')
        client = server_client(cfg['primary_generator'], cfg, cache_path=out / 'completion_cache.sqlite')
        client.allow_incomplete = True
        client.options = {**client.options, 'response_format': {'type': 'json_schema',
            'json_schema': {'name': 'source_relation', 'strict': True, 'schema': SCHEMA}}}
        done = unique_rows(out / 'pairs.jsonl')
        if set(done) - set(unique):
            raise ValueError('Unexpected resumed source pair')
        for key in sorted(unique):
            if key in done:
                continue
            pair = unique[key]; prompt = relation_messages(pair['reports'])
            count = prompt_tokens(tokenizer, prompt, thinking=False)
            c = client.complete(prompt, max_tokens=96, seed=cfg['seed'])
            if c.input_tokens != count:
                raise ValueError('Tokenization mismatch')
            error = None; payload = None; certificate = None
            try:
                if c.incomplete:
                    raise ValueError('Capped or empty response')
                payload = json.loads(c.text); certificate = validate_relation(payload, pair['reports'])
            except (ValueError, KeyError, TypeError) as exc:
                error = str(exc)
            row = {**pair, 'id': key, 'prediction': payload, 'certificate': certificate,
                'raw_response': c.text, 'validation_error': error,
                'input_tokens': c.input_tokens, 'output_tokens': c.output_tokens,
                'request_latency_seconds': c.latency_seconds, 'cache_hit': c.cache_hit,
                'incomplete': c.incomplete, 'request_id': c.request_id,
                'prompt_sha256': stable_hash(prompt), 'source_job_id': os.environ['SLURM_JOB_ID']}
            append_row(out / 'pairs.jsonl', row); done[key] = row
            write_json(out / 'status.json', {'status': 'complete_source_measurements' if set(done) == set(unique) else 'partial_source_measurements',
                'completed_unique_pairs': len(done), 'expected_unique_pairs': len(unique),
                'qa_calls': 0, 'hosted_API_calls': 0, 'world_updates': 0, 'full_runs_authorized': False})
            print(f'NATURAL_RELATION {len(done)}/{len(unique)} valid={error is None}', flush=True)
        if set(done) != set(unique):
            raise ValueError('Incomplete natural source measurement panel')
        kinds = Counter(r['prediction']['relationship'] for r in done.values() if r['validation_error'] is None)
        per_query = []
        for rid, row in inputs.items():
            labels = [done[p['pair_key']] for p in row['measurement_pairs']]
            per_query.append({'id': rid, 'dataset': row['dataset'], 'cluster': row['cluster'],
                'pairs': len(labels), 'invalid': sum(r['validation_error'] is not None for r in labels),
                'provisional_types': [r['prediction']['relationship'] if r['validation_error'] is None else 'invalid' for r in labels],
                'provisional_temporal_pairs': sum(r['validation_error'] is None and
                    r['prediction']['relationship'] == 'temporal_change' for r in labels)})
        report = {'queries': 100, 'unique_pairs': len(done), 'provisional_type_counts': dict(kinds),
            'invalid_or_incomplete': sum(r['validation_error'] is not None for r in done.values()),
            'no_pair_queries': sum(not r['measurement_pairs'] for r in inputs.values()), 'per_query': per_query,
            'fresh_recorded_calls': sum(not r['cache_hit'] for r in done.values()),
            'fresh_input_tokens': sum(r['input_tokens'] for r in done.values() if not r['cache_hit']),
            'fresh_output_tokens': sum(r['output_tokens'] for r in done.values() if not r['cache_hit']),
            'scope': 'provisional coverage, NOT verified natural source relationships, useful energy transitions or QA gains',
            'full_runs_authorized': False}
        write_json(out / 'analysis.json', report)
        print('COVERAGE', {k: v for k, v in report.items() if k != 'per_query'}, flush=True)


if __name__ == '__main__':
    run()
