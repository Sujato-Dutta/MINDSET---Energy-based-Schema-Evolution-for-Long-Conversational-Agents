"""Local source-scope instrument checks on independent synthetic pairs only."""
import json
import os
from pathlib import Path
from mindset.acl.pipeline import ROOT,server_client,request_json,record_environment
from mindset.acl.io import read_json,write_json,append_row,unique_rows,digest_file
from mindset.acl.clients import prompt_tokens
from mindset.memory.source_scope_contract import SCHEMA,CONTRACT_HASH,scope_messages,validate_measurement
from mindset.utils.hashing import stable_hash
from stress_source_semantics import fixtures
from source_scope_confirmation_fixtures import confirmation_fixtures


def run():
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('DGX Slurm only')
    from mindset.acl.pipeline import config
    cfg=config(); checkpoint=read_json(ROOT/'configs/checkpoints.lock.json')[cfg['primary_generator']]
    cases=[{**c,'split':'development'} for c in fixtures()]+[{**c,'split':'lexical_confirmation'} for c in confirmation_fixtures()]
    out=ROOT/'outputs/acl/source_scope_contract_r01'; out.mkdir(parents=True,exist_ok=True)
    identity={'scope':'24 development +24 lexically disjoint synthetic source-scope checks; same families, not OOD or benchmark confirmation',
        'fixture_sha256':stable_hash(cases),'contract_hash':CONTRACT_HASH,
        'contract_module_sha256':digest_file(ROOT/'src/mindset/memory/source_scope_contract.py'),
        'worker_sha256':digest_file(__file__),'checkpoint':checkpoint,'max_output_tokens':1024,'seed':cfg['seed'],
        'model_config_sha256':digest_file(ROOT/'configs/models.yaml'),
        'client_module_sha256':digest_file(ROOT/'src/mindset/acl/clients.py'),
        'decoding':'fixed seed, temperature zero, thinking off, explicit field definitions/schema',
        'quote_verification':'literal provenance only, NOT semantic truth or update correctness',
        'no_tuned_thresholds_or_benchmark_inputs':True,'full_runs_authorized':False}
    import fcntl
    with (out/'worker.lock').open('a+b') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(out/'audit.json',identity,immutable=True); record_environment(out/'continuations'/os.environ['SLURM_JOB_ID'])
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(checkpoint['checkpoint'],revision=checkpoint['revision'],
            cache_dir=str(Path.home()/'MINDSET_v2/.cache/gpu_prototype/huggingface/hub'),local_files_only=True)
        url=os.environ.get('MINDSET_LOCAL_BASE_URL','')
        if url.split('://')[-1].split(':')[0] not in ('localhost','127.0.0.1'): raise ValueError('Local-only model required')
        info=request_json(url.rstrip('/')+'/models')
        if {r['id'] for r in info['data']}!={checkpoint['checkpoint']}: raise ValueError('Wrong served model')
        client=server_client(cfg['primary_generator'],cfg,cache_path=out/'completion_cache.sqlite'); client.allow_incomplete=True
        client.options={**client.options,'response_format':{'type':'json_schema','json_schema':{'name':'source_scope','strict':True,'schema':SCHEMA}}}
        done=unique_rows(out/'measurements.jsonl')
        if set(done)-{c['id'] for c in cases}: raise ValueError('Unexpected synthetic case')
        for case in cases:
            if case['id'] in done: continue
            prompt=scope_messages(case['left'],case['right']); count=prompt_tokens(tokenizer,prompt,thinking=False)
            c=client.complete(prompt,max_tokens=1024,seed=cfg['seed'])
            if c.input_tokens!=count: raise ValueError('Tokenization mismatch')
            error=None; payload=None; certificate=None
            try:
                if c.incomplete: raise ValueError('Capped or empty response')
                payload=json.loads(c.text); certificate=validate_measurement(payload,case['left'],case['right'])
            except (ValueError,TypeError,KeyError) as exc: error=str(exc)
            row={**case,'prediction':payload,'raw_response':c.text,'validation_error':error,'certificate':certificate,
                 'input_tokens':c.input_tokens,'output_tokens':c.output_tokens,'cache_hit':c.cache_hit,'incomplete':c.incomplete,
                 'request_id':c.request_id,'prompt_sha256':stable_hash(prompt),'source_job_id':os.environ['SLURM_JOB_ID']}
            append_row(out/'measurements.jsonl',row); done[case['id']]=row
            write_json(out/'status.json',{'status':'complete_synthetic' if len(done)==len(cases) else 'partial_synthetic',
                'completed_cases':len(done),'expected_cases':len(cases),'benchmark_questions':0,'hosted_API_calls':0,
                'full_runs_authorized':False})
            print(f'SCOPE {len(done)}/{len(cases)} {case["id"]} valid={error is None}',flush=True)
        if set(done)!={c['id'] for c in cases}: raise ValueError('Incomplete synthetic panel')
        rows=list(done.values()); valid=[r for r in rows if r['validation_error'] is None]
        incompatible=[r for r in valid if r['prediction']['relationship']=='incompatible']
        false=[r for r in incompatible if not r['expected_contradiction']]
        misses=[r for r in valid if r['expected_contradiction'] and r['prediction']['relationship']!='incompatible']
        splits={}
        for split in ('development','lexical_confirmation'):
            subset=[r for r in rows if r['split']==split]; ok=[r for r in subset if r['validation_error'] is None]
            splits[split]={'cases':len(subset),'valid_literal_payloads':len(ok),'invalid_or_incomplete':len(subset)-len(ok),
                'false_conflicts':sum(not r['expected_contradiction'] and r['prediction']['relationship']=='incompatible' for r in ok),
                'missed_or_invalid_conflicts':sum(r['expected_contradiction'] and
                    (r['validation_error'] is not None or r['prediction']['relationship']!='incompatible') for r in subset),
                'underdetermined':sum(r['prediction']['relationship']=='underdetermined' for r in ok),
                'unresolved_frames':sum(not f['holder_quote'] or not f['predicate_quote'] or not f['value_quote']
                    for r in ok for f in r['prediction']['frames'])}
        report={'cases':len(cases),'splits':splits,'valid_literal_payloads':len(valid),'invalid_or_incomplete':len(rows)-len(valid),
            'false_conflicts':len(false),'missed_conflicts_in_valid_cases':len(misses),
            'underdetermined_cases':sum(r['prediction']['relationship']=='underdetermined' for r in valid),
            'fresh_recorded_calls':sum(not r['cache_hit'] for r in rows),
            'fresh_input_tokens':sum(r['input_tokens'] for r in rows if not r['cache_hit']),
            'fresh_output_tokens':sum(r['output_tokens'] for r in rows if not r['cache_hit']),
            'scope':'synthetic instrument diagnostic, not native QA, novel controller or benchmark error estimate',
            'cases_detail':rows,'full_runs_authorized':False}
        write_json(out/'analysis.json',report); print('RESULT', {k:v for k,v in report.items() if k!='cases_detail'},flush=True)


if __name__=='__main__': run()
