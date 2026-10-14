"""Bounded 2x2 fixed-evidence LC50 source/reader diagnosis, DGX Slurm only."""
import argparse
from collections import Counter
import os
from pathlib import Path
import numpy as np
from mindset.acl.joint_iteration import protocol,groups_for
from mindset.acl.pipeline import ROOT,server_client,request_json,record_environment,write_csv
from mindset.acl.io import read_json,write_json,unique_rows,append_row,digest_file
from mindset.acl.clients import ANSWER_PROMPT_HASH,prompt_tokens
from mindset.acl.locomo_protocol_diagnostic import captioned_cases,READER_HASH
from mindset.acl.source_reader_factors import ARMS,build_plan
from mindset.datasets.locomo import CATEGORY_NAMES
from mindset.baselines.event_rag import render_event
from mindset.evaluation.locomo import locomo_native_score
from mindset.evaluation.metrics import token_f1
from mindset.utils.hashing import stable_hash

OUTPUT='outputs/acl/source_reader_factors_r01'


def summarize(out,planned):
    rows=list(unique_rows(out/'predictions.jsonl').values()); report=[]
    for arm in ARMS:
        subset=[r for r in rows if r['method']==arm]
        if not subset: continue
        report.append({'method':arm,'questions':len(subset),'expected':50,
            'native_f1':float(np.mean([r['native_f1'] for r in subset])),
            'generic_token_f1':float(np.mean([r['token_f1'] for r in subset])),
            'incomplete_responses':sum(r['incomplete_response'] for r in subset),
            'abstentions':sum('no information available' in r['prediction'].casefold() for r in subset),
            'mean_answer_words':float(np.mean([len(r['prediction'].split()) for r in subset])),
            'mean_evidence_tokens':float(np.mean([r['evidence_tokens'] for r in subset])),
            'logical_input_tokens':sum(r['input_tokens'] for r in subset),
            'logical_output_tokens':sum(r['output_tokens'] for r in subset),
            'scope':'fixed LC50 evidence IDs; shared evaluation factors, not method gains'})
    if report: write_csv(out/'comparison.csv',report)
    fresh=[r for r in rows if not r['reused_prior_artifact'] and not r['completion_cache_hit']]
    write_json(out/'status.json',{'status':'complete_development' if len(rows)==200 else 'partial_development',
        'completed_cells':len(rows),'expected_cells':200,'unique_questions':len({r['question_id'] for r in rows}),
        'fresh_recorded_requests':len(fresh),'fresh_input_tokens':sum(r['input_tokens'] for r in fresh),
        'fresh_output_tokens':sum(r['output_tokens'] for r in fresh),'judge_calls':0,'full_runs_authorized':False,
        'claim_scope':'not method contribution, not confirmation or efficiency'})
    if len(rows)==200:
        by={arm:{r['question_id']:r for r in rows if r['method']==arm} for arm in ARMS}
        expected={p['question_id'] for p in planned}
        if len(expected)!=50 or any(set(v)!=expected for v in by.values()): raise ValueError('Unpaired factors')
        contrast={}
        for a,b in (('text_phrase','text_generic'),('caption_generic','text_generic'),
                    ('caption_phrase','text_generic'),('caption_phrase','text_phrase')):
            differences=[by[a][q]['native_f1']-by[b][q]['native_f1'] for q in sorted(expected)]
            contrast[a+' minus '+b]={'mean_native_f1_difference':float(np.mean(differences)),
                'question_wins':sum(d>1e-12 for d in differences),'question_ties':sum(abs(d)<=1e-12 for d in differences),
                'question_losses':sum(d< -1e-12 for d in differences)}
        detail=[{'question_id':q,'category':by['text_generic'][q]['category'],
            'question':by['text_generic'][q]['question'],'reference_evaluator_only':by['text_generic'][q]['answer'],
            'predictions':{arm:by[arm][q]['prediction'] for arm in ARMS},
            'native_f1':{arm:by[arm][q]['native_f1'] for arm in ARMS}}
            for q in sorted(expected)]
        write_json(out/'analysis.json',{'summary':report,'paired_factor_contrasts':contrast,'cases':detail,
            'interpretation':'Source captions and answer instruction factors cannot be claimed as novel method gains. No semantic judge run.',
            'full_runs_authorized':False})


def run(output):
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('DGX Slurm only')
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    manifest=ROOT/'data/manifests/mindset_v2_development100.json'
    calibration=ROOT/'outputs/acl/mindset_joint_v2_27058/calibration/calibration.json'
    cfg,cohort,_=protocol(manifest,calibration)
    cases={('locomo',c.id):c for d,cluster,group in groups_for(cohort,cfg) if d=='locomo' for c in group}
    captioned=captioned_cases(cases,cfg,ROOT)
    lock=read_json(ROOT/'configs/checkpoints.lock.json'); checkpoint=lock[cfg['primary_generator']]
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(checkpoint['checkpoint'],revision=checkpoint['revision'],
        cache_dir=str(Path.home()/'MINDSET_v2/.cache/gpu_prototype/huggingface/hub'),local_files_only=True)
    prepared=ROOT/'outputs/acl/ten_ideas_r02/merged/retrieval.jsonl'
    planned=build_plan(list(unique_rows(prepared).values()),cases,captioned,tokenizer,cfg)
    prior=ROOT/'outputs/acl/ten_ideas_r02/answers'
    prior_audit=read_json(prior/'audit.json')
    identity={'manifest_sha256':digest_file(manifest),'calibration_sha256':digest_file(calibration),
        'checkpoint':checkpoint,'checkpoint_lock_sha256':digest_file(ROOT/'configs/checkpoints.lock.json'),
        'prompt_hash':ANSWER_PROMPT_HASH,'generation':cfg['generation']}
    for key,value in identity.items():
        if stable_hash(prior_audit[key])!=stable_hash(value): raise ValueError('Prior reader mismatch: '+key)
    old={(r['question_id'],r['prompt_sha256']):r for r in unique_rows(prior/'predictions.jsonl').values()
         if r['dataset']=='locomo' and r['method']=='joint'}
    audit={**identity,'scope':'LC50_fixed_ID_source_reader_factorial','methods':ARMS,'plan_sha256':stable_hash(planned),
        'prepared_sha256':digest_file(prepared),'prior_predictions_sha256':digest_file(prior/'predictions.jsonl'),
        'worker_sha256':digest_file(__file__),'factor_module_sha256':digest_file(ROOT/'src/mindset/acl/source_reader_factors.py'),
        'adapter_module_sha256':digest_file(ROOT/'src/mindset/acl/locomo_protocol_diagnostic.py'),
        'published_reader_hash':READER_HASH,'evidence_ids_and_order':'identical across all four factors',
        'caption_budget':'native caption facts add tokens; not claimed token matched or a method innovation',
        'source_sha256':digest_file(ROOT/cfg['benchmarks']['locomo']['path']),
        'interpretation':'shared evaluation diagnosis only, no retrieval/controller change or retuning',
        'serving':'pinned Qwen3.5-9B BF16 prefetch offload; 20GB MIG, 16k cap, thinking off',
        'judge_calls':0,'full_runs_authorized':False}
    import fcntl
    with (out/'worker.lock').open('a+b') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(out/'audit.json',audit,immutable=True); record_environment(out/'continuations'/os.environ['SLURM_JOB_ID'])
        client=server_client(cfg['primary_generator'],cfg,cache_path=out/'completion_cache.sqlite'); client.allow_incomplete=True
        info=request_json(os.environ['MINDSET_LOCAL_BASE_URL'].rstrip('/')+'/models')
        if {r['id'] for r in info['data']}!={checkpoint['checkpoint']}: raise ValueError('Wrong served model')
        done=unique_rows(out/'predictions.jsonl'); expected={r['id']:r for r in planned}
        if set(done)-set(expected): raise ValueError('Unexpected resumed factor cells')
        for item in planned:
            if item['id'] in done:
                if done[item['id']]['prompt_sha256']!=item['prompt_sha256']: raise ValueError('Changed resumed prompt')
                continue
            source=captioned if item['source_factor']=='native_captions' else cases
            events={e.event_id:e for e in source[('locomo',item['question_id'])].events}
            for s in item['selected']:
                tokens=tokenizer.encode(render_event(events[s['event_id']]),add_special_tokens=False)
                if not 0<=s['token_start']<s['token_end']<=len(tokens) or tokenizer.decode(tokens[s['token_start']:s['token_end']])!=s['text']:
                    raise ValueError('Changed/non-source evidence')
            count=prompt_tokens(tokenizer,item['messages'],thinking=False)
            if count+cfg['generation']['max_output_tokens']>16384: raise ValueError('Context cap exceeded')
            key=(item['question_id'],item['prompt_sha256']); reused=key in old
            if reused:
                response=old[key]; prediction=response['prediction']; inputs=response['input_tokens']; outputs=response['output_tokens']
                hit=True; incomplete=response.get('incomplete_response',False); request_id=response.get('answer_request_id')
            else:
                completion=client.complete(item['messages'],max_tokens=cfg['generation']['max_output_tokens'],seed=cfg['seed'])
                prediction=completion.text; inputs=completion.input_tokens; outputs=completion.output_tokens
                hit=completion.cache_hit; incomplete=completion.incomplete; request_id=completion.request_id
            if inputs!=count: raise ValueError('Server/client tokenization mismatch')
            category=next(int(k) for k,v in CATEGORY_NAMES.items() if v==item['category'])
            row={k:v for k,v in item.items() if k!='messages'}
            row.update(prediction=prediction,native_f1=locomo_native_score(prediction,item['answer'],category),
                token_f1=token_f1(prediction,item['answer']),input_tokens=inputs,output_tokens=outputs,
                completion_cache_hit=hit,reused_prior_artifact=reused,incomplete_response=incomplete,
                answer_request_id=request_id,source_job_id=os.environ['SLURM_JOB_ID'])
            append_row(out/'predictions.jsonl',row); done[row['id']]=row
            if len(done)%10==0: summarize(out,planned)
            print(f'FACTOR {len(done)}/200 {item["id"]} prior={reused} cached={hit}',flush=True)
        summarize(out,planned)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',default=OUTPUT)
    run(p.parse_args().output)
