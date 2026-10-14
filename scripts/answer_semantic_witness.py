"""Matched development100 source-witness answers; DGX Slurm only.

All LC arms use the same published short-phrase reader. LME native judging is
deferred, so its token F1 is secondary only. No references enter requests.
"""
import argparse
import os
from pathlib import Path
import numpy as np
from mindset.acl.joint_iteration import protocol,groups_for
from mindset.acl.pipeline import ROOT,server_client,request_json,record_environment,write_csv
from mindset.acl.io import read_json,write_json,unique_rows,append_row,digest_file
from mindset.acl.clients import ANSWER_PROMPT_HASH,prompt_tokens
from mindset.acl.locomo_protocol_diagnostic import READER_HASH
from mindset.acl.development_answers import request_plan
from mindset.methods.semantic_witness import ARMS
from mindset.datasets.locomo import CATEGORY_NAMES
from mindset.baselines.event_rag import render_event
from mindset.evaluation.locomo import locomo_native_score
from mindset.evaluation.metrics import token_f1
from mindset.utils.hashing import stable_hash

PREPARED='outputs/acl/semantic_witness_r01'
METRICS=('recall@8','precision@8','mrr@8','ndcg@8')
CONTRASTS=(('flat48','joint'),('semantic_pool','flat48'),('semantic_pool','no_verifier'),
           ('no_order','semantic_pool'),('semantic_cover','no_order'),('semantic_cover','joint'))


def build_plan(retrieval,cases):
    planned=[]
    for dataset in ('locomo','longmemeval'):
        local={k:c for k,c in cases.items() if k[0]==dataset}
        rows=[r for r in retrieval if r['dataset']==dataset]
        planned+=request_plan(rows,local,methods=ARMS,
                              reader='locomo_short_phrase' if dataset=='locomo' else 'current')
    return sorted(planned,key=lambda r:(r['dataset'],r['question_id'],ARMS.index(r['method'])))


def summarize(out,planned):
    rows=list(unique_rows(out/'predictions.jsonl').values()); expected={p['id'] for p in planned}
    if {r['id'] for r in rows}-expected: raise ValueError('Unexpected saved answer cell')
    report=[]
    for dataset in ('locomo','longmemeval'):
        for arm in ARMS:
            subset=[r for r in rows if r['dataset']==dataset and r['method']==arm]
            if not subset: continue
            report.append({'dataset':dataset,'method':arm,'questions':len(subset),
                'expected':sum(p['dataset']==dataset and p['method']==arm for p in planned),
                'native_f1':float(np.mean([r['native_f1'] for r in subset])) if dataset=='locomo' else None,
                'secondary_token_f1':float(np.mean([r['token_f1'] for r in subset])),
                **{m:float(np.mean([r['retrieval'][m] for r in subset])) for m in METRICS},
                'native_judge_accuracy':None,'judge_status':'deferred_by_user',
                'incomplete_responses':sum(r['incomplete_response'] for r in subset),
                'mean_answer_words':float(np.mean([len(r['prediction'].split()) for r in subset])),
                'logical_input_tokens':sum(r['input_tokens'] for r in subset),
                'logical_output_tokens':sum(r['output_tokens'] for r in subset),
                'scope':'exposed dev100; shared reader per dataset; NOT final paper or efficiency'})
    if report: write_csv(out/'comparison.csv',report)
    fresh=[r for r in rows if not r['reused_prior_artifact'] and not r['completion_cache_hit']]
    status={'status':'complete_development_answers' if {r['id'] for r in rows}==expected else 'partial_development_answers',
        'completed_cells':len(rows),'expected_cells':len(expected),'remaining_cells':len(expected)-len(rows),
        'reused_prior_answer_cells':sum(r['reused_prior_artifact'] for r in rows),
        'fresh_recorded_answer_calls':len(fresh),'fresh_recorded_input_tokens':sum(r['input_tokens'] for r in fresh),
        'fresh_recorded_output_tokens':sum(r['output_tokens'] for r in fresh),
        'judge_calls':0,'full_runs_authorized':False,'latency_scope':'exact request reuse; NOT efficiency'}
    write_json(out/'status.json',status)
    if status['remaining_cells']: return
    paired={}; cases=[]
    for dataset in ('locomo','longmemeval'):
        by={arm:{r['question_id']:r for r in rows if r['dataset']==dataset and r['method']==arm} for arm in ARMS}
        wanted={p['question_id'] for p in planned if p['dataset']==dataset}
        if any(set(values)!=wanted for values in by.values()): raise ValueError('Unpaired answer panels')
        metric='native_f1' if dataset=='locomo' else 'token_f1'; paired[dataset]={}
        for a,b in CONTRASTS:
            differences=[by[a][q][metric]-by[b][q][metric] for q in sorted(wanted)]
            clusters={by[a][q]['cluster'] for q in wanted}
            clustered={c:float(np.mean([by[a][q][metric]-by[b][q][metric] for q in wanted if by[a][q]['cluster']==c]))
                       for c in sorted(clusters)}
            paired[dataset][a+' minus '+b]={'metric':metric,'mean_difference':float(np.mean(differences)),
                'wins':sum(d>1e-12 for d in differences),'ties':sum(abs(d)<=1e-12 for d in differences),
                'losses':sum(d< -1e-12 for d in differences),'per_cluster_mean_differences':clustered,
                'changed_prompts':sum(by[a][q]['prompt_sha256']!=by[b][q]['prompt_sha256'] for q in wanted),
                'ranking_mean_differences':{m:float(np.mean([by[a][q]['retrieval'][m]-by[b][q]['retrieval'][m] for q in wanted])) for m in METRICS},
                'inference_scope':'descriptive repeated-look development; no confirmatory p-value'}
        for q in sorted(wanted):
            if all(by[a][q]['prompt_sha256']==by['joint'][q]['prompt_sha256'] for a in ARMS[1:]): continue
            cases.append({'dataset':dataset,'question_id':q,'question':by['joint'][q]['question'],
                'reference_evaluator_only':by['joint'][q]['answer'],
                'predictions':{a:by[a][q]['prediction'] for a in ARMS},
                'scores':{a:by[a][q][metric] for a in ARMS},
                'selected_ids':{a:[s['event_id'] for s in by[a][q]['selected']] for a in ARMS},
                'semantic_edges':by['semantic_cover'][q]['semantic_edges'],
                'certificate':by['semantic_cover'][q]['certificate']})
    write_json(out/'analysis.json',{'summary':report,'paired':paired,'changed_cases':cases,'usage':status,
        'claim_limit':'NLI report labels are proxies; no novel-method, causal truth, native LME accuracy or efficiency claim',
        'full_runs_authorized':False})


def verified_previous(cfg,manifest,calibration,checkpoint):
    """Terminal source artifacts only; prompt hash checked for every reused cell."""
    import fcntl
    if cfg['seed']!=cfg['generation']['seed']: raise ValueError('Ambiguous generation seed')
    compatible={'manifest_sha256':digest_file(manifest),'calibration_sha256':digest_file(calibration),
        'checkpoint':checkpoint,'checkpoint_lock_sha256':digest_file(ROOT/'configs/checkpoints.lock.json'),
        'prompt_hash':ANSWER_PROMPT_HASH,'generation':cfg['generation']}
    sources=((ROOT/'outputs/acl/source_reader_factors_r01','locomo','text_phrase','complete_development'),
             (ROOT/'outputs/acl/ten_ideas_r02/answers','longmemeval','joint','complete_development_answers'))
    previous={}; identities=[]
    for source,dataset,method,terminal in sources:
        with (source/'worker.lock').open('rb') as handle:
            fcntl.flock(handle,fcntl.LOCK_SH|fcntl.LOCK_NB)
            if read_json(source/'status.json')['status']!=terminal: raise ValueError('Prior reader not terminal')
            audit=read_json(source/'audit.json')
            for key,value in compatible.items():
                if stable_hash(audit[key])!=stable_hash(value): raise ValueError('Incompatible previous reader: '+key)
            if dataset=='locomo' and (audit['published_reader_hash']!=READER_HASH or
                audit['adapter_module_sha256']!=digest_file(ROOT/'src/mindset/acl/locomo_protocol_diagnostic.py')):
                raise ValueError('Published reader changed')
            for r in unique_rows(source/'predictions.jsonl').values():
                if r['dataset']!=dataset or r['method']!=method: continue
                from mindset.acl.clients import messages
                from mindset.acl.locomo_protocol_diagnostic import short_phrase_messages
                reconstructed=short_phrase_messages(r['question'],r['evidence'],r['question_date'],category=r['category']) if dataset=='locomo' else messages(r['question'],r['evidence'],r['question_date'])
                if stable_hash(reconstructed)!=r['prompt_sha256']: raise ValueError('Prior prompt cannot be reconstructed')
                key=(dataset,r['question_id'],r['prompt_sha256'])
                if key in previous: raise ValueError('Ambiguous prior answer')
                previous[key]=r
            identities.append({'path':str(source),'audit_sha256':digest_file(source/'audit.json'),
                               'predictions_sha256':digest_file(source/'predictions.jsonl'),
                               'selected_dataset':dataset,'selected_method':method})
    return previous,identities


def run(prepared,output):
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('DGX Slurm only')
    manifest=ROOT/'data/manifests/mindset_v2_development100.json'
    calibration=ROOT/'outputs/acl/mindset_joint_v2_27058/calibration/calibration.json'
    cfg,cohort,_=protocol(manifest,calibration); prepared=Path(prepared); out=Path(output)
    status=read_json(prepared/'status.json')
    if status['status']!='complete_development' or status['completed_cells']!=600:
        raise ValueError('Complete original100 retrieval panel required')
    if not status['changed_packets']: raise ValueError('No changed packets: answer spending gate closed')
    prior_audit=read_json(prepared/'audit.json')
    if prior_audit['manifest_sha256']!=digest_file(manifest) or prior_audit['calibration_sha256']!=digest_file(calibration):
        raise ValueError('Preparation identity mismatch')
    if prior_audit['source_adapter']!='original_frozen_text_only': raise ValueError('Unexpected source adapter')
    cases={(d,c.id):c for d,cluster,group in groups_for(cohort,cfg) for c in group}
    retrieval=unique_rows(prepared/'retrieval.jsonl'); planned=build_plan(list(retrieval.values()),cases)
    if len(planned)!=600: raise ValueError('Original100 six-arm panel required')
    checkpoint=read_json(ROOT/'configs/checkpoints.lock.json')[cfg['primary_generator']]
    previous,identities=verified_previous(cfg,manifest,calibration,checkpoint)
    out.mkdir(parents=True,exist_ok=True)
    audit={'manifest_sha256':digest_file(manifest),'calibration_sha256':digest_file(calibration),
        'checkpoint':checkpoint,'checkpoint_lock_sha256':digest_file(ROOT/'configs/checkpoints.lock.json'),
        'generation':cfg['generation'],'seed':cfg['seed'],'methods':ARMS,'plan_sha256':stable_hash(planned),
        'model_config_sha256':digest_file(ROOT/'configs/models.yaml'),
        'client_module_sha256':digest_file(ROOT/'src/mindset/acl/clients.py'),
        'reader_hashes':{'locomo':READER_HASH,'longmemeval':ANSWER_PROMPT_HASH},
        'source_sha256':{d:digest_file(ROOT/cfg['benchmarks'][d]['path']) for d in ('locomo','longmemeval')},
        'prepared_sha256':digest_file(prepared/'retrieval.jsonl'),'prepared_audit_sha256':digest_file(prepared/'audit.json'),
        'worker_sha256':digest_file(__file__),'previous_terminal_artifacts':identities,
        'request_plan_sha256':digest_file(ROOT/'src/mindset/acl/development_answers.py'),
        'reader_module_sha256':digest_file(ROOT/'src/mindset/acl/locomo_protocol_diagnostic.py'),
        'scope':'original exposed dev100, same reader per dataset, original raw text sources',
        'serving':'pinned Qwen3.5-9B BF16 prefetch offload, 20GB MIG, 16k cap, no thinking',
        'judge':'deferred_by_user','cache_policy':'exact same prompt/checkpoint/generation; not latency measurement',
        'full_runs_authorized':False}
    import fcntl
    with (out/'worker.lock').open('a+b') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(out/'audit.json',audit,immutable=True); record_environment(out/'continuations'/os.environ['SLURM_JOB_ID'])
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(checkpoint['checkpoint'],revision=checkpoint['revision'],
            cache_dir=str(Path.home()/'MINDSET_v2/.cache/gpu_prototype/huggingface/hub'),local_files_only=True)
        if os.environ.get('MINDSET_LOCAL_BASE_URL','').split('://')[-1].split(':')[0] not in ('localhost','127.0.0.1'):
            raise ValueError('Local-only server required')
        models=request_json(os.environ['MINDSET_LOCAL_BASE_URL'].rstrip('/')+'/models')
        if {r['id'] for r in models['data']}!={checkpoint['checkpoint']}: raise ValueError('Wrong served checkpoint')
        client=server_client(cfg['primary_generator'],cfg,cache_path=out/'completion_cache.sqlite'); client.allow_incomplete=True
        done=unique_rows(out/'predictions.jsonl'); expected={r['id']:r for r in planned}
        if set(done)-set(expected): raise ValueError('Unexpected resumed answer cells')
        for row in planned:
            if row['id'] in done:
                if done[row['id']]['prompt_sha256']!=row['prompt_sha256']: raise ValueError('Changed resumed prompt')
                continue
            original={e.event_id:e for e in cases[(row['dataset'],row['question_id'])].events}
            for s in row['selected']:
                ids=tokenizer.encode(render_event(original[s['event_id']]),add_special_tokens=False)
                a,b=s['token_start'],s['token_end']
                if not 0<=a<b<=len(ids) or tokenizer.decode(ids[a:b])!=s['text']:
                    raise ValueError('Non-source evidence span')
            count=prompt_tokens(tokenizer,row['messages'],thinking=False)
            if count+cfg['generation']['max_output_tokens']>16384: raise ValueError('Context cap exceeded')
            key=(row['dataset'],row['question_id'],row['prompt_sha256']); reused=key in previous
            if reused:
                old=previous[key]; prediction=old['prediction']; inputs=old['input_tokens']; outputs=old['output_tokens']
                cache_hit=True; incomplete=old.get('incomplete_response',False); rid=old.get('answer_request_id')
            else:
                c=client.complete(row['messages'],max_tokens=cfg['generation']['max_output_tokens'],seed=cfg['seed'])
                prediction=c.text; inputs=c.input_tokens; outputs=c.output_tokens
                cache_hit=c.cache_hit; incomplete=c.incomplete; rid=c.request_id
            if count!=inputs: raise ValueError('Tokenization mismatch')
            native=None
            if row['dataset']=='locomo':
                category=next(int(k) for k,v in CATEGORY_NAMES.items() if v==row['category'])
                native=locomo_native_score(prediction,row['answer'],category)
            saved={k:v for k,v in row.items() if k!='messages'}
            saved.update(prediction=prediction,native_f1=native,token_f1=token_f1(prediction,row['answer']),
                input_tokens=inputs,output_tokens=outputs,completion_cache_hit=cache_hit,incomplete_response=incomplete,
                reused_prior_artifact=reused,answer_request_id=rid,source_job_id=os.environ['SLURM_JOB_ID'],
                certificate=retrieval[row['id']]['certificate'],semantic_edges=retrieval[row['id']]['semantic_edges'])
            append_row(out/'predictions.jsonl',saved); done[row['id']]=saved
            if len(done)%12==0: summarize(out,planned)
            print(f'ANSWER {len(done)}/600 {row["id"]} prior={reused} cache={cache_hit}',flush=True)
        summarize(out,planned)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--prepared',default=PREPARED)
    p.add_argument('--output',default=PREPARED+'/answers'); args=p.parse_args(); run(args.prepared,args.output)
