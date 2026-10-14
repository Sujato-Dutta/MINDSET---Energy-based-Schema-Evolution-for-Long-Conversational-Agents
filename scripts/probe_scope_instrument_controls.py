"""DGX-only pair-label vs grounded-frame controls, no benchmark inputs."""
import json
import os
from pathlib import Path

from mindset.acl.pipeline import ROOT, server_client, request_json, record_environment, config
from mindset.acl.io import read_json, write_json, append_row, unique_rows, digest_file
from mindset.acl.clients import prompt_tokens
from mindset.memory.source_scope_contract import CONTRACT_HASH, scope_messages, validate_measurement
from mindset.memory.scope_instrument_controls import (
    ARMS, CONTROL_HASH, output_schema, control_messages, validate_control, summarize_controls)
from mindset.utils.hashing import stable_hash
from stress_source_semantics import fixtures
from source_scope_confirmation_fixtures import confirmation_fixtures


def verified_parent(cases, checkpoint, cfg):
    parent = ROOT / 'outputs/acl/source_scope_contract_r01'
    audit = read_json(parent / 'audit.json')
    expected = {'fixture_sha256': stable_hash(cases), 'contract_hash': CONTRACT_HASH,
        'checkpoint': checkpoint, 'seed': cfg['seed'],
        'contract_module_sha256': digest_file(ROOT / 'src/mindset/memory/source_scope_contract.py'),
        'worker_sha256': digest_file(ROOT / 'scripts/probe_source_scope_contract.py'),
        'model_config_sha256': digest_file(ROOT / 'configs/models.yaml'),
        'client_module_sha256': digest_file(ROOT / 'src/mindset/acl/clients.py')}
    if any(audit.get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen parent identity mismatch')
    rows = unique_rows(parent / 'measurements.jsonl')
    if set(rows) != {c['id'] for c in cases}:
        raise ValueError('Parent synthetic panel incomplete or unexpected')
    for case in cases:
        row = rows[case['id']]
        if any(row.get(k) != v for k, v in case.items()):
            raise ValueError('Parent source/expectation changed')
        if row['prompt_sha256'] != stable_hash(scope_messages(case['left'], case['right'])):
            raise ValueError('Parent source prompt changed')
        error = None
        try:
            if row['incomplete']:
                raise ValueError('Capped or empty response')
            payload = json.loads(row['raw_response'])
            validate_measurement(payload, case['left'], case['right'])
            if payload != row['prediction']:
                raise ValueError('Parent parsed response mismatch')
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
        if error != row['validation_error']:
            raise ValueError('Parent quote revalidation mismatch')
    return rows, {name: digest_file(parent / name) for name in ('audit.json', 'measurements.jsonl')}


def run():
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('DGX Slurm only')
    cfg = config()
    checkpoint = read_json(ROOT / 'configs/checkpoints.lock.json')[cfg['primary_generator']]
    cases = [{**c, 'split': 'development'} for c in fixtures()] + [
        {**c, 'split': 'lexical_confirmation'} for c in confirmation_fixtures()]
    parent, parent_hashes = verified_parent(cases, checkpoint, cfg)
    out = ROOT / 'outputs/acl/scope_instrument_controls_r01'
    out.mkdir(parents=True, exist_ok=True)
    identity = {'fixture_sha256': stable_hash(cases), 'control_hash': CONTROL_HASH,
        'checkpoint': checkpoint, 'seed': cfg['seed'], 'parent_hashes': parent_hashes,
        'worker_sha256': digest_file(__file__),
        'control_module_sha256': digest_file(ROOT / 'src/mindset/memory/scope_instrument_controls.py'),
        'model_config_sha256': digest_file(ROOT / 'configs/models.yaml'),
        'client_module_sha256': digest_file(ROOT / 'src/mindset/acl/clients.py'),
        'max_output_tokens': {'relation_only': 64, 'grounded_frames': 1024},
        'scope': 'same 48 exposed synthetic fixtures; causal output-cost control, not held-out QA',
        'expected_labels_in_prompts': False, 'full_runs_authorized': False}
    import fcntl
    with (out / 'worker.lock').open('a+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(out / 'audit.json', identity, immutable=True)
        record_environment(out / 'continuations' / os.environ['SLURM_JOB_ID'])
        parent_rows = [{**r, 'arm': 'original_frames', 'parent_reuse': True} for r in parent.values()]
        # Independent recomputation of rejection-aware statistics, not the
        # accepted-only false-conflict rate in the old report.
        write_json(out / 'parent_reaudit.json', summarize_controls(parent_rows))
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
        base_options = dict(client.options)
        done = unique_rows(out / 'measurements.jsonl')
        expected_ids = {f'{a}/{c["id"]}' for c in cases for a in ARMS}
        if set(done) - expected_ids:
            raise ValueError('Unexpected control case')
        for case in cases:
            for arm in ARMS:
                rid = f'{arm}/{case["id"]}'
                if rid in done:
                    continue
                prompt = control_messages(arm, case['left'], case['right'])
                count = prompt_tokens(tokenizer, prompt, thinking=False)
                client.options = {**base_options, 'response_format': {'type': 'json_schema',
                    'json_schema': {'name': 'source_scope_control', 'strict': True, 'schema': output_schema(arm)}}}
                c = client.complete(prompt, max_tokens=identity['max_output_tokens'][arm], seed=cfg['seed'])
                if c.input_tokens != count:
                    raise ValueError('Tokenization mismatch')
                error = None; payload = None; certificate = None
                try:
                    if c.incomplete:
                        raise ValueError('Capped or empty response')
                    payload = json.loads(c.text)
                    certificate = validate_control(arm, payload, case['left'], case['right'])
                except (ValueError, TypeError, KeyError) as exc:
                    error = str(exc)
                row = {**case, 'id': rid, 'case_id': case['id'], 'arm': arm, 'prediction': payload,
                    'raw_response': c.text, 'validation_error': error, 'certificate': certificate,
                    'input_tokens': c.input_tokens, 'output_tokens': c.output_tokens,
                    'cache_hit': c.cache_hit, 'parent_reuse': False, 'incomplete': c.incomplete,
                    'request_id': c.request_id, 'prompt_sha256': stable_hash(prompt),
                    'source_job_id': os.environ['SLURM_JOB_ID']}
                append_row(out / 'measurements.jsonl', row); done[rid] = row
                write_json(out / 'status.json', {'status': 'complete_synthetic' if set(done) == expected_ids else 'partial_synthetic',
                    'completed_cases': len(done), 'expected_cases': len(expected_ids),
                    'benchmark_questions': 0, 'hosted_API_calls': 0, 'full_runs_authorized': False})
                print(f'CONTROL {len(done)}/{len(expected_ids)} {rid} valid={error is None}', flush=True)
        if set(done) != expected_ids:
            raise ValueError('Incomplete control panel')
        report = summarize_controls(parent_rows + list(done.values()))
        write_json(out / 'analysis.json', report)
        print('RESULT', report, flush=True)


if __name__ == '__main__':
    run()
