"""Matched answers for the bounded 40-question transition packet pilot."""
import argparse
from collections import Counter
import os
from pathlib import Path
import numpy as np
from mindset.acl.joint_iteration import protocol,groups_for
from mindset.acl.pipeline import ROOT,server_client,request_json,record_environment,write_csv
from mindset.acl.io import read_json,write_json,unique_rows,append_row,digest_file
from mindset.acl.clients import ANSWER_PROMPT_HASH,prompt_tokens
from mindset.acl.development_answers import request_plan
from mindset.methods.transition_packets import ARMS
from mindset.datasets.locomo import CATEGORY_NAMES
from mindset.evaluation.locomo import locomo_native_score
from mindset.evaluation.metrics import token_f1
from mindset.baselines.event_rag import render_event
from mindset.utils.hashing import stable_hash

PREPARED='outputs/acl/transition_packets_r01'
PRIOR='outputs/acl/ten_ideas_r02/answers'
METRICS=('recall@8','precision@8','mrr@8','ndcg@8')


def summarize(out,planned):
    rows=list(unique_rows(out/'predictions.jsonl').values()); report=[]
    for dataset in ('locomo','longmemeval'):
        for arm in ARMS:
            subset=[r for r in rows if r['dataset']==dataset and r['method']==arm]
            if not subset: continue
            report.append({'dataset':dataset,'method':arm,'questions':len(subset),'expected':20,
                'native_f1':float(np.mean([r['native_f1'] for r in subset])) if dataset=='locomo' else None,
                'secondary_token_f1':float(np.mean([r['token_f1'] for r in subset])),
                **{m:float(np.mean([r['retrieval'][m] for r in subset])) for m in METRICS},
                'native_judge_accuracy':None,'judge_status':'deferred_by_user',
                'incomplete_responses':sum(r['incomplete_response'] for r in subset),
                'reused_prior_answer_cells':sum(r['reused_prior_artifact'] for r in subset),
                'logical_input_tokens':sum(r['input_tokens'] for r in subset),
                'logical_output_tokens':sum(r['output_tokens'] for r in subset),
                'scope':'paired exposed development40; NOT final paper or efficiency'})
    if report: write_csv(out/'comparison.csv',report)
    status={'status':'complete_development_answers' if len(rows)==len(planned) else 'partial_development_answers',
        'completed_cells':len(rows),'expected_cells':len(planned),
        'remaining_cells':len(planned)-len(rows),'reused_prior_answer_cells':sum(r['reused_prior_artifact'] for r in rows),
        'fresh_physical_answer_calls':sum(not r['reused_prior_artifact'] and not r['completion_cache_hit'] for r in rows),
        'new_recorded_input_tokens':sum(r['input_tokens'] for r in rows if not r['reused_prior_artifact'] and not r['completion_cache_hit']),
        'new_recorded_output_tokens':sum(r['output_tokens'] for r in rows if not r['reused_prior_artifact'] and not r['completion_cache_hit']),
        'judge_calls':0,'full_runs_authorized':False,'latency_scope':'cache/artifact reuse; no efficiency claims'}
    write_json(out/'status.json',status)
    if len(rows)==len(planned):
        paired={}
        for dataset in ('locomo','longmemeval'):
            baseline={r['question_id']:r for r in rows if r['dataset']==dataset and r['method']=='joint'}
            if len(baseline)!=20: raise ValueError('Invalid paired baseline')
            paired[dataset]={}
            for arm in ARMS[1:]:
                values={r['question_id']:r for r in rows if r['dataset']==dataset and r['method']==arm}
                if set(values)!=set(baseline): raise ValueError('Unpaired results')
                deltas={'secondary_token_f1':float(np.mean([values[q]['token_f1']-baseline[q]['token_f1'] for q in values])),
                    **{m:float(np.mean([values[q]['retrieval'][m]-baseline[q]['retrieval'][m] for q in values])) for m in METRICS}}
                if dataset=='locomo': deltas['native_f1']=float(np.mean([values[q]['native_f1']-baseline[q]['native_f1'] for q in values]))
                changes=[]
                for q in values:
                    if values[q]['prompt_sha256']==baseline[q]['prompt_sha256']: continue
                    changes.append({'question_id':q,'question':values[q]['question'],'reference_evaluator_only':values[q]['answer'],
                        'baseline_answer':baseline[q]['prediction'],'new_answer':values[q]['prediction'],
                        'native_f1_delta':values[q]['native_f1']-baseline[q]['native_f1'] if dataset=='locomo' else None,
                        'token_f1_delta':values[q]['token_f1']-baseline[q]['token_f1'],
                        'certificate':values[q]['certificate']})
                paired[dataset][arm]={'mean_differences':deltas,'changed_cases':changes}
            energy={r['question_id']:r for r in rows if r['dataset']==dataset and r['method']=='compact_energy'}
            control={r['question_id']:r for r in rows if r['dataset']==dataset and r['method']=='compact_no_energy'}
            paired[dataset]['energy_vs_no_energy']={'token_f1_difference':float(np.mean([energy[q]['token_f1']-control[q]['token_f1'] for q in energy])),
                'native_f1_difference':float(np.mean([energy[q]['native_f1']-control[q]['native_f1'] for q in energy])) if dataset=='locomo' else None}
        write_json(out/'analysis.json',{'summary':report,'paired':paired,'usage':status,
            'claim_scope':'descriptive pilot; no retuning, no confirmed significance, judge deferred, no physical-energy claims',
            'full_runs_authorized':False})


def run(prepared,output,resolve_only=False):
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('DGX Slurm only')
    manifest=ROOT/'data/manifests/mindset_v2_development100.json'
    calibration=ROOT/'outputs/acl/mindset_joint_v2_27058/calibration/calibration.json'
    cfg,cohort,_=protocol(manifest,calibration)
    prepared=Path(prepared); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    status=read_json(prepared/'status.json')
    if status['status']!='complete_development' or status['completed_cells']!=160: raise ValueError('Incomplete source screening')
    subset=read_json(prepared/'subset.json'); selected={(r['dataset'],r['id']) for r in subset['records']}
    if len(selected)!=40 or subset['source_manifest_sha256']!=digest_file(manifest): raise ValueError('Wrong pilot subset')
    if subset['records']!=[r for r in cohort['records'] if (r['dataset'],r['id']) in selected]: raise ValueError('Unknown subset records')
    cases={(d,c.id):c for d,cluster,group in groups_for(cohort,cfg) for c in group if (d,c.id) in selected}
    retrieval=unique_rows(prepared/'retrieval.jsonl'); planned=request_plan(list(retrieval.values()),cases,methods=ARMS)
    checkpoint=read_json(ROOT/'configs/checkpoints.lock.json')[cfg['primary_generator']]
    prior=ROOT/PRIOR; oldaudit=read_json(prior/'audit.json')
    compatible={'manifest_sha256':digest_file(manifest),'calibration_sha256':digest_file(calibration),
        'checkpoint':checkpoint,'checkpoint_lock_sha256':digest_file(ROOT/'configs/checkpoints.lock.json'),
        'prompt_hash':ANSWER_PROMPT_HASH,'generation':cfg['generation']}
    for key,value in compatible.items():
        if stable_hash(oldaudit[key])!=stable_hash(value): raise ValueError('Incompatible prior reader: '+key)
    if read_json(prior/'status.json')['status']!='complete_development_answers': raise ValueError('Prior answers not terminal')
    previous={}
    for r in unique_rows(prior/'predictions.jsonl').values():
        key=(r['dataset'],r['question_id'],r['prompt_sha256'])
        if key in previous and r['prediction']!=previous[key]['prediction']: raise ValueError('Inconsistent identical prior prompts')
        previous[key]=r
    audit={**compatible,'scope':'matched_transition_packet_answers_development40','methods':ARMS,
        'subset_sha256':digest_file(prepared/'subset.json'),'prepared_sha256':digest_file(prepared/'retrieval.jsonl'),
        'prepared_audit_sha256':digest_file(prepared/'audit.json'),'plan_sha256':stable_hash(planned),
        'prior_predictions_sha256':digest_file(prior/'predictions.jsonl'),'prior_audit_sha256':digest_file(prior/'audit.json'),
        'worker_sha256':digest_file(__file__),'source_adapter':'original_frozen_text_only',
        'serving':'BF16 pinned checkpoint; explicit prefetch CPU offload; 20GB MIG; 16k cap; no quantization',
        'cache_policy':'identical full prior prompts/readers only; fresh requests for changed packets; not efficiency',
        'native_judge':'deferred_by_user','full_runs_authorized':False}
    import fcntl
    with (out/'worker.lock').open('a+b') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(out/'audit.json',audit,immutable=True); record_environment(out/'continuations'/os.environ['SLURM_JOB_ID'])
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(checkpoint['checkpoint'],revision=checkpoint['revision'],
            cache_dir=str(Path.home()/'MINDSET_v2/.cache/gpu_prototype/huggingface/hub'),local_files_only=True)
        client=None
        if not resolve_only:
            if os.environ.get('MINDSET_LOCAL_BASE_URL','').split('://')[-1].split(':')[0] not in ('127.0.0.1','localhost'):
                raise ValueError('Local server required')
            info=request_json(os.environ['MINDSET_LOCAL_BASE_URL'].rstrip('/')+'/models')
            if {r['id'] for r in info['data']}!={checkpoint['checkpoint']}: raise ValueError('Wrong served model')
            client=server_client(cfg['primary_generator'],cfg,cache_path=out/'completion_cache.sqlite')
            client.allow_incomplete=True
        done=unique_rows(out/'predictions.jsonl'); expected={r['id']:r for r in planned}
        if set(done)-set(expected): raise ValueError('Unexpected resumed cells')
        for row in planned:
            if row['id'] in done:
                if done[row['id']]['prompt_sha256']!=row['prompt_sha256']: raise ValueError('Changed resumed answer')
                continue
            original={e.event_id:e for e in cases[(row['dataset'],row['question_id'])].events}
            for passage in row['selected']:
                ids=tokenizer.encode(render_event(original[passage['event_id']]),add_special_tokens=False)
                a,b=passage['token_start'],passage['token_end']
                if not 0<=a<b<=len(ids) or tokenizer.decode(ids[a:b])!=passage['text']: raise ValueError('Evidence is not an immutable source span')
            count=prompt_tokens(tokenizer,row['messages'],thinking=False)
            if count+cfg['generation']['max_output_tokens']>16384: raise ValueError('Context cap exceeded')
            key=(row['dataset'],row['question_id'],row['prompt_sha256']); reused=key in previous
            if reused:
                result=previous[key]
                if result['input_tokens']!=count: raise ValueError('Changed inherited tokenization')
                prediction=result['prediction']; input_tokens=result['input_tokens']; output_tokens=result['output_tokens']
                cache_hit=True; incomplete=result.get('incomplete_response',False); request_id=result.get('answer_request_id')
            else:
                if resolve_only: continue
                completion=client.complete(row['messages'],max_tokens=cfg['generation']['max_output_tokens'],seed=cfg['seed'])
                if completion.input_tokens!=count: raise ValueError('Server/client tokenizer mismatch')
                prediction=completion.text; input_tokens=completion.input_tokens; output_tokens=completion.output_tokens
                cache_hit=completion.cache_hit; incomplete=completion.incomplete; request_id=completion.request_id
            native=None
            if row['dataset']=='locomo':
                category=next(int(k) for k,v in CATEGORY_NAMES.items() if v==row['category'])
                native=locomo_native_score(prediction,row['answer'],category)
            saved={k:v for k,v in row.items() if k!='messages'}
            saved.update(prediction=prediction,native_f1=native,token_f1=token_f1(prediction,row['answer']),
                input_tokens=input_tokens,output_tokens=output_tokens,completion_cache_hit=cache_hit,
                incomplete_response=incomplete,reused_prior_artifact=reused,answer_request_id=request_id,
                source_job_id=os.environ['SLURM_JOB_ID'],certificate=retrieval[row['id']]['certificate'])
            append_row(out/'predictions.jsonl',saved); done[row['id']]=saved
            print(f'ANSWER {len(done)}/160 {row["id"]} reused_prior={reused}',flush=True)
        summarize(out,planned)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--prepared',default=PREPARED)
    p.add_argument('--output',default=PREPARED+'/answers'); p.add_argument('--resolve-only',action='store_true')
    a=p.parse_args(); run(a.prepared,a.output,a.resolve_only)
