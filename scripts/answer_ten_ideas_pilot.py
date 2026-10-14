"""Matched local answers for the complete ten-idea development100 panel."""
import argparse
import os
from pathlib import Path
from mindset.acl.ten_ideas_pilot import ARMS
from mindset.acl.joint_iteration import protocol, groups_for
from mindset.acl.pipeline import ROOT, server_client, request_json, record_environment
from mindset.acl.io import read_json,write_json,unique_rows,append_row,digest_file
from mindset.acl.clients import ANSWER_PROMPT_HASH,load_tokenizer,prompt_tokens
from mindset.acl.development_answers import request_plan,summarize,inherit_cache
from mindset.datasets.locomo import CATEGORY_NAMES
from mindset.evaluation.locomo import locomo_native_score
from mindset.evaluation.metrics import token_f1
from mindset.baselines.event_rag import render_event
from mindset.utils.hashing import stable_hash

def run(prepared, output, inherit=None):
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('DGX Slurm only')
    manifest=ROOT/'data/manifests/mindset_v2_development100.json'
    calibration=ROOT/'outputs/acl/mindset_joint_v2_27058/calibration/calibration.json'
    cfg,cohort,_=protocol(manifest,calibration)
    prepared=Path(prepared); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    state=read_json(prepared/'status.json')
    if state['status']!='complete_development' or state['completed_cells']!=1100: raise ValueError('Complete original100 x11 required')
    cases={(dataset,c.id):c for dataset,cluster,group in groups_for(cohort,cfg) for c in group}
    rows=list(unique_rows(prepared/'retrieval.jsonl').values())
    planned=request_plan(rows,cases,methods=ARMS)
    checkpoint=read_json(ROOT/'configs/checkpoints.lock.json')[cfg['primary_generator']]
    audit={'scope':'ten_ideas_original100_matched_answers','prepared_sha256':digest_file(prepared/'retrieval.jsonl'),
        'manifest_sha256':digest_file(manifest),'calibration_sha256':digest_file(calibration),
        'checkpoint':checkpoint,'checkpoint_lock_sha256':digest_file(ROOT/'configs/checkpoints.lock.json'),
        'prompt_hash':ANSWER_PROMPT_HASH,'generation':cfg['generation'],'methods':ARMS,'plan_sha256':stable_hash(planned),
        'serving':'BF16 pinned checkpoint; explicit prefetch CPU offload; 20GB MIG; 16k cap; no quantization',
        'exposure':'repeated-look development, not confirmatory evaluation','source_adapter':'original_frozen_text_only',
        'cache_policy':'only identical full requests; not efficiency or independent repeatability',
        'native_lme_judge':'deferred','worker_sha256':digest_file(__file__),'full_runs_authorized':False}
    import fcntl
    with (out/'worker.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(out/'audit.json',audit,immutable=True)
        if inherit: inherit_cache(inherit,out,audit,strict_plan=False)
        record_environment(out/'continuations'/os.environ['SLURM_JOB_ID'])
        tokenizer=load_tokenizer(checkpoint)
        client=server_client(cfg['primary_generator'],cfg,cache_path=out/'completion_cache.sqlite')
        client.allow_incomplete=True
        info=request_json(os.environ['MINDSET_LOCAL_BASE_URL'].rstrip('/')+'/models')
        if {r['id'] for r in info['data']}!={checkpoint['checkpoint']}: raise ValueError('Wrong generator')
        done=unique_rows(out/'predictions.jsonl'); expected={r['id']:r for r in planned}
        if set(done)-set(expected): raise ValueError('Extra answer cells')
        source_tokens={}
        for item in planned:
            original={e.event_id:e for e in cases[(item['dataset'],item['question_id'])].events}
            for selected in item['selected']:
                key=(item['dataset'],item['cluster'],selected['event_id'])
                if key not in source_tokens: source_tokens[key]=tokenizer.encode(render_event(original[selected['event_id']]),add_special_tokens=False)
                start,end=selected['token_start'],selected['token_end']; tokens=source_tokens[key]
                if not 0<=start<end<=len(tokens) or tokenizer.decode(tokens[start:end])!=selected['text']:
                    raise ValueError('Generated or changed evidence')
            if item['id'] in done:
                if done[item['id']]['prompt_sha256']!=item['prompt_sha256']: raise ValueError('Changed resumed prompt')
                continue
            count=prompt_tokens(tokenizer,item['messages'],thinking=False)
            if count+cfg['generation']['max_output_tokens']>16384: raise ValueError('Context exceeds frozen diagnostic cap')
            completion=client.complete(item['messages'],max_tokens=cfg['generation']['max_output_tokens'],seed=cfg['seed'])
            if completion.input_tokens!=count: raise ValueError('Server/client prompt count mismatch')
            native=None
            if item['dataset']=='locomo':
                category=next(int(k) for k,v in CATEGORY_NAMES.items() if v==item['category'])
                native=locomo_native_score(completion.text,item['answer'],category)
            row={k:v for k,v in item.items() if k!='messages'}
            row.update(prediction=completion.text,native_f1=native,token_f1=token_f1(completion.text,item['answer']),
                input_tokens=completion.input_tokens,output_tokens=completion.output_tokens,
                completion_cache_hit=completion.cache_hit,answer_request_id=completion.request_id,
                finish_reason=completion.finish_reason,incomplete_response=completion.incomplete,
                source_job_id=os.environ['SLURM_JOB_ID'],status='development_not_paper')
            append_row(out/'predictions.jsonl',row); done[row['id']]=row
            if len(done)%25==0: summarize(out,planned,ARMS)
            print(f'ANSWER {len(done)}/{len(planned)} {item["id"]} cached={completion.cache_hit}',flush=True)
        summarize(out,planned,ARMS)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',default='outputs/acl/ten_ideas_r02/merged')
    p.add_argument('--output',default='outputs/acl/ten_ideas_r02/answers')
    p.add_argument('--inherit-successful-cache')
    a=p.parse_args(); run(a.prepared,a.output,a.inherit_successful_cache)
