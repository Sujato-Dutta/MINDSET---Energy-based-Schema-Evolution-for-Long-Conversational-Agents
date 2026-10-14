"""DGX-only finite source-relation diagnostic. No memory or QA deployment."""
import json
import os
from pathlib import Path

from mindset.acl.pipeline import ROOT, config, server_client, request_json, record_environment
from mindset.acl.io import read_json, write_json, append_row, unique_rows, digest_file
from mindset.acl.clients import prompt_tokens
from mindset.memory.source_relation_contract import SCHEMA, CONTRACT_HASH, relation_messages, validate_relation
from mindset.utils.hashing import stable_hash
from source_relation_fixtures import relation_fixtures


def summarize(rows):
    groups = {}
    for split in ('development', 'lexical_confirmation'):
        subset = [r for r in rows if r['split'] == split]
        valid = [r for r in subset if r['validation_error'] is None]
        joint_correct = [r for r in valid if r['prediction'] == {
            'relationship': r['expected_relationship'], 'direction': r['expected_direction']}]
        confusion = {}
        for row in subset:
            predicted = 'invalid' if row['validation_error'] else row['prediction']['relationship']
            key = f'{row["expected_relationship"]}->{predicted}'
            confusion[key] = confusion.get(key, 0) + 1
        groups[split] = {'cases': len(subset), 'valid_outputs': len(valid),
            'invalid_or_incomplete': len(subset) - len(valid), 'joint_type_direction_correct': len(joint_correct),
            'relationship_correct': sum(r['prediction']['relationship'] == r['expected_relationship'] for r in valid),
            'confusion': confusion,
            'retrospective_temporal_correct': sum(r['expected_direction'] == 'b_to_a' for r in joint_correct),
            'retrospective_temporal_cases': sum(r['expected_direction'] == 'b_to_a' for r in subset),
            'logical_input_tokens': sum(r['input_tokens'] for r in subset),
            'logical_output_tokens': sum(r['output_tokens'] for r in subset)}
    return {'splits': groups, 'cases': len(rows), 'rejections_not_dropped': True,
            'scope': 'synthetic source-relation feasibility, not semantic truth, native QA, domain OOD or novel energy controller',
            'full_runs_authorized': False}


def run():
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('DGX Slurm only')
    cfg = config(); checkpoint = read_json(ROOT / 'configs/checkpoints.lock.json')[cfg['primary_generator']]
    cases = relation_fixtures()
    out = ROOT / 'outputs/acl/source_relation_instrument_r01'
    out.mkdir(parents=True, exist_ok=True)
    audit = {'fixture_sha256': stable_hash(cases), 'contract_hash': CONTRACT_HASH,
        'checkpoint': checkpoint, 'seed': cfg['seed'], 'worker_sha256': digest_file(__file__),
        'contract_module_sha256': digest_file(ROOT / 'src/mindset/memory/source_relation_contract.py'),
        'fixture_module_sha256': digest_file(ROOT / 'scripts/source_relation_fixtures.py'),
        'model_config_sha256': digest_file(ROOT / 'configs/models.yaml'),
        'client_module_sha256': digest_file(ROOT / 'src/mindset/acl/clients.py'),
        'max_output_tokens': 96, 'no_benchmark_inputs_or_parameter_fitting': True,
        'no_generated_atoms_or_automatic_state_updates': True, 'full_runs_authorized': False}
    import fcntl
    with (out / 'worker.lock').open('a+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(out / 'audit.json', audit, immutable=True)
        write_json(out / 'fixtures.json', cases, immutable=True)
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
        done = unique_rows(out / 'measurements.jsonl'); expected_ids = {r['id'] for r in cases}
        if set(done) - expected_ids:
            raise ValueError('Unexpected diagnostic case')
        for case in cases:
            if case['id'] in done:
                continue
            # Only source reports and output schema are passed to the model.
            prompt = relation_messages(case['reports'])
            count = prompt_tokens(tokenizer, prompt, thinking=False)
            c = client.complete(prompt, max_tokens=96, seed=cfg['seed'])
            if c.input_tokens != count:
                raise ValueError('Tokenization mismatch')
            error = None; payload = None; certificate = None
            try:
                if c.incomplete:
                    raise ValueError('Capped or empty response')
                payload = json.loads(c.text); certificate = validate_relation(payload, case['reports'])
            except (ValueError, KeyError, TypeError) as exc:
                error = str(exc)
            row = {**case, 'prediction': payload, 'raw_response': c.text,
                'validation_error': error, 'certificate': certificate,
                'input_tokens': c.input_tokens, 'output_tokens': c.output_tokens,
                'cache_hit': c.cache_hit, 'incomplete': c.incomplete, 'request_id': c.request_id,
                'prompt_sha256': stable_hash(prompt), 'source_job_id': os.environ['SLURM_JOB_ID']}
            append_row(out / 'measurements.jsonl', row); done[case['id']] = row
            write_json(out / 'status.json', {'status': 'complete_synthetic' if set(done) == expected_ids else 'partial_synthetic',
                'completed_cases': len(done), 'expected_cases': len(expected_ids), 'benchmark_questions': 0,
                'hosted_API_calls': 0, 'full_runs_authorized': False})
            print(f'RELATION {len(done)}/{len(expected_ids)} {case["id"]} valid={error is None}', flush=True)
        if set(done) != expected_ids:
            raise ValueError('Incomplete source relation panel')
        report = summarize(list(done.values())); write_json(out / 'analysis.json', report)
        print('RESULT', report, flush=True)


if __name__ == '__main__':
    run()
