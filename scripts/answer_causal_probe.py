"""Bounded native-QA causal test; exact identical prompts reuse one answer.

This is quality evaluation, never efficiency or repeatability measurement.
LongMemEval native judge accuracy is unavailable until independently judged.
"""
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict
import argparse
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mindset.acl.io import read_json,write_json,digest_file,unique_rows,append_row
from mindset.acl.pipeline import config,record_environment,server_client,write_csv
from mindset.acl.clients import load_tokenizer,prompt_tokens,ANSWER_PROMPT_HASH
from mindset.datasets.locomo import CATEGORY_NAMES
from mindset.evaluation.locomo import locomo_native_score
from mindset.evaluation.metrics import token_f1
from mindset.utils.hashing import stable_hash

def score_prediction(row,result):
    category=next((int(k) for k,v in CATEGORY_NAMES.items() if v==row['category']),None)
    return {**row,'prediction':result.text,'generation':asdict(result),
        'native_f1':locomo_native_score(result.text,row['answer'],category) if row['dataset']=='locomo' else None,
        'secondary_token_f1':token_f1(result.text,row['answer']),
        'native_judge_accuracy':None,'judge_status':'not_judged',
        'scope':'development_quality_cached_identical_prompts_not_latency_or_repeatability'}

def summarize(out,expected):
    from mindset.acl.analysis import paired_stats
    from mindset.statistics.tests import holm_correction
    import numpy as np
    rows=list(unique_rows(out/'predictions.jsonl').values()); report=[]; statistics=[]
    for dataset in ('locomo','longmemeval'):
        arms=sorted({r['arm'] for r in expected.values() if r['dataset']==dataset})
        full={r['question_id']:r for r in rows if r['dataset']==dataset and r['arm']=='full'}
        family=[]
        for arm in arms:
            selected=[r for r in rows if r['dataset']==dataset and r['arm']==arm]
            if not selected: continue
            report.append({'dataset':dataset,'arm':arm,'questions':len(selected),'expected':50,
                'native_f1':float(np.mean([r['native_f1'] for r in selected])) if dataset=='locomo' else None,
                'secondary_token_f1':float(np.mean([r['secondary_token_f1'] for r in selected])),
                'native_judge_accuracy':None,'judge_status':'not_judged'})
            if dataset=='locomo' and len(selected)==50 and len(full)==50 and arm!='full':
                pairs=[(full[r['question_id']],r) for r in selected]
                stats=paired_stats([a['native_f1'] for a,b in pairs],[b['native_f1'] for a,b in pairs],
                                   [a['cluster'] for a,b in pairs],seed=0,resamples=10000)
                family.append({'dataset':dataset,'control':arm,'metric':'native_f1',**stats,
                    'interpretation':'exploratory repeated-look development, not confirmatory paper inference'})
        for row,p in zip(family,holm_correction([r['p'] for r in family])): row['holm_p']=p; statistics.append(row)
    if report: write_csv(out/'quality.csv',report)
    if statistics: write_csv(out/'exploratory_statistics.csv',statistics)
    # The request ledger survives a crash between completion and prediction
    # append. Counting only uncached prediction rows would undercount spending
    # when that row is later recovered from the exact-request cache.
    import sqlite3,json
    with sqlite3.connect(out/'exact_request_cache.sqlite') as ledger:
        completed_requests=[json.loads(r[0]) for r in ledger.execute('SELECT payload FROM local_completions')]
    write_json(out/'status.json',{'completed_cells':len(rows),'expected_cells':len(expected),
        'status':'complete_development_answers' if len(rows)==len(expected) else 'partial_resumable_development_answers',
        'completed_unique_answer_requests':len(completed_requests),
        'committed_prediction_cache_hits':sum(r['generation']['cache_hit'] for r in rows),
        'actual_answer_tokens':sum(r['input_tokens']+r['output_tokens'] for r in completed_requests),
        'usage_scope':'successful completions in durable request ledger; excludes failed attempts and server startup',
        'judge_calls':0,'native_longmemeval_accuracy_available':False,'full_runs_authorized':False})

def run(prepared,output):
    cfg=config(); prepared=Path(prepared); out=Path(output); audit=read_json(prepared/'audit.json')
    if read_json(prepared/'status.json')['status']!='complete_development': raise ValueError('Complete frozen causal evidence required')
    cohort=read_json(ROOT/'data/manifests/mindset_v2_development100.json')
    if audit['manifest_sha256']!=digest_file(ROOT/'data/manifests/mindset_v2_development100.json'):
        raise ValueError('Development manifest changed')
    lock=read_json(ROOT/'configs/checkpoints.lock.json')
    if audit['checkpoint_lock']!=lock or audit['answer_prompt_sha256']!=ANSWER_PROMPT_HASH: raise ValueError('Checkpoint/prompt changed')
    expected=unique_rows(prepared/'evidence.jsonl')
    expected_ids={f"{r['dataset']}/{a}/{r['id']}" for r in cohort['records'] for a in audit['arms']}
    if set(expected)!=expected_ids or len(cohort['records'])!=100: raise ValueError('Incomplete/unexpected causal question set')
    for r in expected.values():
        if stable_hash(r['messages'])!=r['prompt_sha256']: raise ValueError('Prompt bytes changed')
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'worker.lock').open('a+b') as worker:
        fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity={'prepared_evidence_sha256':digest_file(prepared/'evidence.jsonl'),'prepared_audit':audit,
            'worker_sha256':digest_file(Path(__file__)),'model':lock[cfg['primary_generator']],
            'generation':cfg['generation'],'thinking':False,'scope':'development_quality_not_efficiency',
            'reuse':'one completion per exact complete prompt/model/revision/seed/output-limit; never count cache-hit tokens as actual spend'}
        write_json(out/'audit.json',identity,immutable=True); record_environment(out)
        done=unique_rows(out/'predictions.jsonl')
        if set(done)-set(expected): raise ValueError('Unexpected resumed prediction IDs')
        tokenizer=load_tokenizer(lock[cfg['primary_generator']])
        client=server_client(cfg['primary_generator'],cfg,cache_path=out/'exact_request_cache.sqlite')
        for id,row in expected.items():
            if id in done: continue
            if prompt_tokens(tokenizer,row['messages'],thinking=False)+cfg['generation']['max_output_tokens']>16384:
                raise ValueError('Bounded development context exceeded, no silent truncation')
            result=client.complete(row['messages'],cfg['generation']['max_output_tokens'],cfg['seed'])
            scored=score_prediction(row,result); append_row(out/'predictions.jsonl',scored); done[id]=scored
            # Save progress without bootstrapping repeatedly after every answer.
            if len(done)%50==0: summarize(out,expected)
            print(f'{len(done)}/{len(expected)} causal answers; cache_hit={result.cache_hit}',flush=True)
        summarize(out,expected)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--prepared',required=True); p.add_argument('--output',required=True)
    args=p.parse_args()
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    run(args.prepared,args.output)

if __name__=='__main__': main()
