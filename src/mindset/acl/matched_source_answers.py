"""Reusable original-development100 raw-source answer panels, DGX Slurm only.

Exact terminal-request reuse is a quality optimization, never latency evidence.
Shared published LC reader; unchanged generic LME reader; native judge deferred.
"""
import os
from pathlib import Path
import numpy as np
from mindset.acl.pipeline import ROOT,server_client,request_json,record_environment,write_csv
from mindset.acl.joint_iteration import protocol,groups_for
from mindset.acl.io import read_json,write_json,unique_rows,append_row,digest_file
from mindset.acl.clients import messages,ANSWER_PROMPT_HASH,prompt_tokens
from mindset.acl.locomo_protocol_diagnostic import short_phrase_messages,READER_HASH
from mindset.acl.development_answers import request_plan
from mindset.baselines.event_rag import render_event
from mindset.evaluation.locomo import locomo_native_score
from mindset.evaluation.metrics import token_f1
from mindset.datasets.locomo import CATEGORY_NAMES
from mindset.utils.hashing import stable_hash

METRICS=('recall@8','precision@8','mrr@8','ndcg@8')


def summarize(out,planned,arms,contrasts):
    rows=list(unique_rows(out/'predictions.jsonl').values()); expected={p['id'] for p in planned}
    if {r['id'] for r in rows}-expected: raise ValueError('Unexpected answer cell')
    report=[]
    for d in ('locomo','longmemeval'):
        for arm in arms:
            subset=[r for r in rows if r['dataset']==d and r['method']==arm]
            if not subset: continue
            report.append({'dataset':d,'method':arm,'questions':len(subset),
                'expected':sum(p['dataset']==d and p['method']==arm for p in planned),
                'native_f1':float(np.mean([r['native_f1'] for r in subset])) if d=='locomo' else None,
                'secondary_token_f1':float(np.mean([r['token_f1'] for r in subset])),
                **{m:float(np.mean([r['retrieval'][m] for r in subset])) for m in METRICS},
                'native_judge_accuracy':None,'judge_status':'deferred_by_user',
                'incomplete_responses':sum(r['incomplete_response'] for r in subset),
                'logical_input_tokens':sum(r['input_tokens'] for r in subset),
                'logical_output_tokens':sum(r['output_tokens'] for r in subset),
                'mean_logical_ce_pairs':float(np.mean([r['logical_ce_pairs'] for r in subset])),
                'scope':'paired exposed original100, shared reader, NOT paper confirmation or efficiency'})
    if report: write_csv(out/'comparison.csv',report)
    fresh=[r for r in rows if not r['reused_parent_artifact'] and not r['completion_cache_hit']]
    status={'status':'complete_development_answers' if {r['id'] for r in rows}==expected else 'partial_development_answers',
        'completed_cells':len(rows),'expected_cells':len(expected),'remaining_cells':len(expected)-len(rows),
        'reused_parent_cells':sum(r['reused_parent_artifact'] for r in rows),
        'fresh_recorded_calls':len(fresh),'fresh_recorded_input_tokens':sum(r['input_tokens'] for r in fresh),
        'fresh_recorded_output_tokens':sum(r['output_tokens'] for r in fresh),'judge_calls':0,'full_runs_authorized':False}
    write_json(out/'status.json',status)
    if status['remaining_cells']: return
    comparisons={}; changes=[]
    for d in ('locomo','longmemeval'):
        by={a:{r['question_id']:r for r in rows if r['dataset']==d and r['method']==a} for a in arms}
        ids={p['question_id'] for p in planned if p['dataset']==d}
        if any(set(v)!=ids for v in by.values()): raise ValueError('Unpaired panels')
        metric='native_f1' if d=='locomo' else 'token_f1'; comparisons[d]={}
        for a,b in contrasts:
            delta=[by[a][q][metric]-by[b][q][metric] for q in sorted(ids)]
            changed=[]
            for q in sorted(ids):
                if by[a][q]['prompt_sha256']==by[b][q]['prompt_sha256']: continue
                changed.append(q)
                changes.append({'dataset':d,'contrast':a+' minus '+b,'question_id':q,'question':by[a][q]['question'],
                    'reference_evaluator_only':by[a][q]['answer'],'new_prediction':by[a][q]['prediction'],
                    'reference_prediction':by[b][q]['prediction'],'score_delta':by[a][q][metric]-by[b][q][metric],
                    'new_source_ids':[s['event_id'] for s in by[a][q]['selected']],
                    'reference_source_ids':[s['event_id'] for s in by[b][q]['selected']]})
            clusters={by[a][q]['cluster'] for q in ids}
            comparisons[d][a+' minus '+b]={'metric':metric,'mean_difference':float(np.mean(delta)),
                'wins':sum(v>1e-12 for v in delta),'ties':sum(abs(v)<=1e-12 for v in delta),'losses':sum(v< -1e-12 for v in delta),
                'changed_prompts':changed,'per_cluster_mean_differences':{c:float(np.mean([by[a][q][metric]-by[b][q][metric]
                    for q in ids if by[a][q]['cluster']==c])) for c in sorted(clusters)},
                'ranking_mean_differences':{m:float(np.mean([by[a][q]['retrieval'][m]-by[b][q]['retrieval'][m] for q in ids])) for m in METRICS}}
    write_json(out/'analysis.json',{'summary':report,'paired':comparisons,'changed_cases':changes,'usage':status,
        'scope':'descriptive repeated-look controls; no novel-method or significance claim; LME token overlap is secondary',
        'full_runs_authorized':False})


def prior_answers(source,identity):
    """A sealed full-request parent, with independently reconstructed prompts."""
    import fcntl
    with (source/'worker.lock').open('rb') as handle:
        fcntl.flock(handle,fcntl.LOCK_SH|fcntl.LOCK_NB)
        if read_json(source/'status.json')['status']!='complete_development_answers': raise ValueError('Parent not terminal')
        audit=read_json(source/'audit.json')
        for k in ('manifest_sha256','calibration_sha256','checkpoint','checkpoint_lock_sha256','generation','seed',
                  'model_config_sha256','client_module_sha256','reader_hashes','source_sha256'):
            if stable_hash(audit[k])!=stable_hash(identity[k]): raise ValueError('Incompatible parent: '+k)
        result={}
        for r in unique_rows(source/'predictions.jsonl').values():
            prompt=short_phrase_messages(r['question'],r['evidence'],r['question_date'],category=r['category']) if r['dataset']=='locomo' else messages(r['question'],r['evidence'],r['question_date'])
            if stable_hash(prompt)!=r['prompt_sha256']: raise ValueError('Parent prompt cannot be reconstructed')
            key=(r['dataset'],r['question_id'],r['prompt_sha256'])
            if key in result and result[key]['prediction']!=r['prediction']: raise ValueError('Conflicting identical parent requests')
            result[key]=r
        return result,{'path':str(source),'audit_sha256':digest_file(source/'audit.json'),
                       'predictions_sha256':digest_file(source/'predictions.jsonl')}


def run_panel(prepared,output,arms,contrasts,scope,parent,*,
              source_adapter='original_frozen_text_only',plan_transform=None):
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('DGX Slurm only')
    prepared=Path(prepared); out=Path(output); parent=Path(parent)
    manifest=ROOT/'data/manifests/mindset_v2_development100.json'
    calibration=ROOT/'outputs/acl/mindset_joint_v2_27058/calibration/calibration.json'
    cfg,cohort,_=protocol(manifest,calibration)
    if cohort['count']!=100: raise ValueError('Original100 only')
    status=read_json(prepared/'status.json'); audit=read_json(prepared/'audit.json')
    if status['status']!='complete_development' or status['completed_cells']!=100*len(arms): raise ValueError('Incomplete source panel')
    for k,path in (('manifest_sha256',manifest),('calibration_sha256',calibration),('checkpoint_lock_sha256',ROOT/'configs/checkpoints.lock.json')):
        if audit[k]!=digest_file(path): raise ValueError('Preparation identity differs: '+k)
    if audit['source_adapter']!=source_adapter or tuple(audit['methods'])!=tuple(arms): raise ValueError('Wrong methods/source adapter')
    retrieval=unique_rows(prepared/'retrieval.jsonl')
    cases={(d,c.id):c for d,cluster,group in groups_for(cohort,cfg) for c in group}
    planned=[]
    for d in ('locomo','longmemeval'):
        planned+=request_plan([r for r in retrieval.values() if r['dataset']==d],{k:c for k,c in cases.items() if k[0]==d},
                              methods=arms,reader='locomo_short_phrase' if d=='locomo' else 'current')
    if len(planned)!=100*len(arms): raise ValueError('Wrong plan size')
    if plan_transform is not None:
        planned=plan_transform(planned,retrieval,cases)
    planned.sort(key=lambda r:(r['dataset'],r['question_id'],arms.index(r['method'])))
    checkpoint=read_json(ROOT/'configs/checkpoints.lock.json')[cfg['primary_generator']]
    identity={'manifest_sha256':digest_file(manifest),'calibration_sha256':digest_file(calibration),
        'checkpoint':checkpoint,'checkpoint_lock_sha256':digest_file(ROOT/'configs/checkpoints.lock.json'),
        'generation':cfg['generation'],'seed':cfg['seed'],'model_config_sha256':digest_file(ROOT/'configs/models.yaml'),
        'client_module_sha256':digest_file(ROOT/'src/mindset/acl/clients.py'),
        'reader_hashes':{'locomo':READER_HASH,'longmemeval':ANSWER_PROMPT_HASH},
        'source_sha256':{d:digest_file(ROOT/cfg['benchmarks'][d]['path']) for d in ('locomo','longmemeval')},
        'worker_sha256':digest_file(__file__),'request_plan_module_sha256':digest_file(ROOT/'src/mindset/acl/development_answers.py'),
        'reader_module_sha256':digest_file(ROOT/'src/mindset/acl/locomo_protocol_diagnostic.py'),
        'scope':scope,'methods':arms,'contrasts':contrasts,'plan_sha256':stable_hash(planned),
        'prepared_sha256':digest_file(prepared/'retrieval.jsonl'),'prepared_audit_sha256':digest_file(prepared/'audit.json'),
        'source_adapter':source_adapter,
        'plan_transform_sha256':digest_file(__import__('inspect').getfile(plan_transform)) if plan_transform else None,
        'serving':'pinned Qwen3.5-9B BF16 prefetch offload, 20GB MIG, 16k cap, thinking off',
        'cache_scope':'exact full request reuse, not latency or repeatability','full_runs_authorized':False}
    previous,parent_identity=prior_answers(parent,identity); identity['parent']=parent_identity
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'worker.lock').open('a+b') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(out/'audit.json',identity,immutable=True); record_environment(out/'continuations'/os.environ['SLURM_JOB_ID'])
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(checkpoint['checkpoint'],revision=checkpoint['revision'],
            cache_dir=str(Path.home()/'MINDSET_v2/.cache/gpu_prototype/huggingface/hub'),local_files_only=True)
        url=os.environ.get('MINDSET_LOCAL_BASE_URL','')
        if url.split('://')[-1].split(':')[0] not in ('localhost','127.0.0.1'): raise ValueError('Local-only reader required')
        info=request_json(url.rstrip('/')+'/models')
        if {r['id'] for r in info['data']}!={checkpoint['checkpoint']}: raise ValueError('Wrong served model')
        client=server_client(cfg['primary_generator'],cfg,cache_path=out/'completion_cache.sqlite'); client.allow_incomplete=True
        done=unique_rows(out/'predictions.jsonl'); expected={r['id']:r for r in planned}
        if set(done)-set(expected): raise ValueError('Unexpected resumed answers')
        for r in planned:
            if r['id'] in done:
                if done[r['id']]['prompt_sha256']!=r['prompt_sha256']: raise ValueError('Changed resumed prompt')
                continue
            events={e.event_id:e for e in cases[(r['dataset'],r['question_id'])].events}
            for s in r['selected']:
                tokens=tokenizer.encode(render_event(events[s['event_id']]),add_special_tokens=False)
                a,b=s['token_start'],s['token_end']
                if not 0<=a<b<=len(tokens) or tokenizer.decode(tokens[a:b])!=s['text']: raise ValueError('Non-source passage')
            count=prompt_tokens(tokenizer,r['messages'],thinking=False)
            if count+cfg['generation']['max_output_tokens']>16384: raise ValueError('Context cap exceeded')
            key=(r['dataset'],r['question_id'],r['prompt_sha256']); reused=key in previous
            if reused:
                old=previous[key]; text=old['prediction']; inputs=old['input_tokens']; outputs=old['output_tokens']
                hit=True; incomplete=old['incomplete_response']; request_id=old['answer_request_id']
            else:
                c=client.complete(r['messages'],max_tokens=cfg['generation']['max_output_tokens'],seed=cfg['seed'])
                text=c.text; inputs=c.input_tokens; outputs=c.output_tokens; hit=c.cache_hit; incomplete=c.incomplete; request_id=c.request_id
            if inputs!=count: raise ValueError('Tokenization mismatch')
            native=None
            if r['dataset']=='locomo':
                category=next(int(k) for k,v in CATEGORY_NAMES.items() if v==r['category'])
                native=locomo_native_score(text,r['answer'],category)
            saved={k:v for k,v in r.items() if k!='messages'}
            saved.update(prediction=text,native_f1=native,token_f1=token_f1(text,r['answer']),input_tokens=inputs,output_tokens=outputs,
                completion_cache_hit=hit,incomplete_response=incomplete,reused_parent_artifact=reused,answer_request_id=request_id,
                source_job_id=os.environ['SLURM_JOB_ID'],logical_ce_pairs=retrieval[r['id']]['logical_ce_pairs'])
            append_row(out/'predictions.jsonl',saved); done[r['id']]=saved
            if len(done)%12==0: summarize(out,planned,arms,contrasts)
            print(f'ANSWER {len(done)}/{len(planned)} {r["id"]} parent={reused} cache={hit}',flush=True)
        summarize(out,planned,arms,contrasts)
