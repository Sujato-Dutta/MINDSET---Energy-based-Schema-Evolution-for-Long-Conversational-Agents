"""Single frozen development100 A/B; all work requires DGX Slurm."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np

from mindset.acl.joint_iteration import protocol, groups_for
from mindset.acl.pipeline import ROOT, local_stack, server_client, request_json, rank_metrics, record_environment, write_csv
from mindset.acl.io import read_json, write_json, append_row, unique_rows, digest_file
from mindset.acl.clients import load_tokenizer, messages, prompt_tokens, ANSWER_PROMPT_HASH
from mindset.acl.evidence import EvidenceCache
from mindset.baselines.event_rag import render_event
from mindset.methods.compact import CompactConfig
from mindset.methods.joint import JointMindsetMemory
from mindset.methods.mindset import _render_event
from mindset.methods.query_state_agent import (inventory, planner_schema, planner_messages,
    execute, fallback, retrieve, state_fingerprint, PLANNER_PROMPT)
from mindset.datasets.locomo import CATEGORY_NAMES
from mindset.evaluation.locomo import locomo_native_score
from mindset.evaluation.metrics import token_f1
from mindset.utils.hashing import stable_hash

ARMS = ('joint', 'state_agent')
OUTPUT = 'outputs/acl/query_state_agent_r01'
METRICS = ('recall@8', 'precision@8', 'mrr@8', 'ndcg@8')


def summarize(out):
    rows = list(unique_rows(out/'predictions.jsonl').values()); report = []
    for dataset in ('locomo', 'longmemeval'):
        for method in ARMS:
            subset = [r for r in rows if r['dataset'] == dataset and r['method'] == method]
            if not subset: continue
            fresh = [r for r in subset if not r['answer_cache_hit'] and not r['planner_cache_hit']]
            bound = [r for r in subset if r['execution']['bound']]
            item = {'dataset': dataset, 'method': method, 'questions': len(subset), 'expected': 50,
                **{m: float(np.mean([r['metrics'][m] for r in subset])) for m in METRICS},
                'native_f1': float(np.mean([r['native_f1'] for r in subset])) if dataset == 'locomo' else None,
                'secondary_token_f1': float(np.mean([r['token_f1'] for r in subset])),
                'native_judge_accuracy': None, 'judge_status': 'deferred_by_user',
                'binding_rate': len(bound)/len(subset),
                'actual_version_link_rate': sum(r['execution']['transition_link_used'] for r in subset)/len(subset),
                'actual_branch_link_rate': sum(r['execution']['branch_link_used'] for r in subset)/len(subset),
                'retrieval_p50_ms': float(np.quantile([r['retrieval_seconds']*1000 for r in fresh], .5)) if fresh else None,
                'retrieval_p95_ms': float(np.quantile([r['retrieval_seconds']*1000 for r in fresh], .95)) if fresh else None,
                'end_to_end_p50_ms': float(np.quantile([r['end_to_end_seconds']*1000 for r in fresh], .5)) if fresh else None,
                'end_to_end_p95_ms': float(np.quantile([r['end_to_end_seconds']*1000 for r in fresh], .95)) if fresh else None,
                'latency_fresh_cells': len(fresh), 'latency_scope': 'single-pass, shared history index; diagnostic, not repeated paper efficiency',
                'logical_answer_tokens': sum(r['input_tokens']+r['output_tokens'] for r in subset),
                'logical_planner_tokens': sum(r['planner_input_tokens']+r['planner_output_tokens'] for r in subset),
                'incomplete_answers': sum(r['answer_incomplete'] for r in subset)}
            report.append(item)
    if report: write_csv(out/'comparison.csv', report)
    write_json(out/'status.json', {'completed_cells': len(rows), 'expected_cells': 200,
        'unique_questions': len({(r['dataset'],r['question_id']) for r in rows}),
        'status': 'complete_development' if len(rows) == 200 else 'partial_development',
        'native_judge': 'deferred_by_user', 'full_runs_authorized': False})
    if len(rows) == 200:
        paired = {}
        for dataset in ('locomo', 'longmemeval'):
            a = {r['question_id']: r for r in rows if r['dataset']==dataset and r['method']=='joint'}
            b = {r['question_id']: r for r in rows if r['dataset']==dataset and r['method']=='state_agent'}
            if set(a)!=set(b) or len(a)!=50: raise ValueError('Nonpaired pilot')
            keys = ('native_f1','token_f1') if dataset=='locomo' else ('token_f1',)
            differences = {k: float(np.mean([b[q][k]-a[q][k] for q in a])) for k in keys}
            differences.update({m:float(np.mean([b[q]['metrics'][m]-a[q]['metrics'][m] for q in a])) for m in METRICS})
            changes = []
            for q in a:
                if a[q]['prediction'] != b[q]['prediction'] or a[q]['selected'] != b[q]['selected']:
                    changes.append({'question_id':q,'question':a[q]['question'],'category':a[q]['category'],
                        'reference':a[q]['answer'],'baseline_answer':a[q]['prediction'],'agent_answer':b[q]['prediction'],
                        'token_f1_delta': b[q]['token_f1']-a[q]['token_f1'],
                        'native_f1_delta': b[q]['native_f1']-a[q]['native_f1'] if dataset=='locomo' else None,
                        'packet_changed':a[q]['selected']!=b[q]['selected'], 'execution':b[q]['execution']})
            paired[dataset] = {'differences':differences,'changed_cases':changes,
                'passes_secondary_token_f1_gate': differences['token_f1']>=.03,
                'official_judge_gate':'not_tested', 'scope':'repeated-look development, not confirmatory'}
        write_json(out/'analysis.json', {'paired':paired,'summary':report,'full_runs_authorized':False})
    return report


def inherit_answers(source,out,audit):
    """Terminal prior requests only; SQLite keys still require full-body equality."""
    import fcntl,sqlite3
    source=Path(source); prior=read_json(source/'audit.json')
    if read_json(source/'status.json')['status']!='complete_development': raise ValueError('Prior A/B not terminal')
    for key in ('manifest_sha256','calibration_sha256','checkpoint','checkpoint_lock_sha256',
                'answer_prompt_hash','generation','source_adapter'):
        if stable_hash(prior[key])!=stable_hash(audit[key]): raise ValueError('Incompatible prior answers: '+key)
    with (source/'worker.lock').open('rb') as handle:
        fcntl.flock(handle,fcntl.LOCK_SH|fcntl.LOCK_NB)
        for arm in ARMS:
            old=source/f'answer_{arm}_cache.sqlite'; destination=out/f'answer_{arm}_cache.sqlite'
            if destination.exists(): continue
            with sqlite3.connect('file:'+str(old)+'?mode=ro',uri=True) as before:
                with sqlite3.connect(destination) as after: before.backup(after)
    write_json(out/'answer_inheritance.json',{'source':str(source),'prior_audit_sha256':digest_file(source/'audit.json'),
        'policy':'exact full answer request including checkpoint revision; no planner-response inheritance; not efficiency'},immutable=True)


def run(output,planner_profile='original',inherit=None):
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('DGX Slurm only')
    if planner_profile not in ('original','explicit'): raise ValueError('Unknown planner profile')
    prompt_builder=planner_messages
    if planner_profile=='explicit':
        from mindset.acl.planner_contract import explicit_messages,FIELD_CONTRACT
        prompt_builder=explicit_messages
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    manifest=ROOT/'data/manifests/mindset_v2_development100.json'
    calibration=ROOT/'outputs/acl/mindset_joint_v2_27058/calibration/calibration.json'
    cfg, cohort, calibrated = protocol(manifest, calibration)
    if cohort['count']!=100 or cfg['primary_generator']!='qwen35_9b': raise ValueError('Wrong frozen pilot')
    parameters=CompactConfig(**{**{k:v for k,v in cfg['mindset'].items() if k!='profile'},**calibrated['selected_parameters']})
    checkpoint=read_json(ROOT/'configs/checkpoints.lock.json')[cfg['primary_generator']]
    frozen_paths=[ROOT/'src/mindset'/p for p in ('methods/joint.py','methods/compact.py','methods/mindset.py','memory/energy.py')]
    audit={'scope':'query_state_agent_frozen100_ab','methods':ARMS,'manifest_sha256':digest_file(manifest),
        'calibration_sha256':digest_file(calibration),'parameters':asdict(parameters),'checkpoint':checkpoint,
        'checkpoint_lock_sha256':digest_file(ROOT/'configs/checkpoints.lock.json'),
        'frozen_controller_sha256':{str(p.relative_to(ROOT)):digest_file(p) for p in frozen_paths},
        'planner_source_sha256':digest_file(ROOT/'src/mindset/methods/query_state_agent.py'),
        'worker_sha256':digest_file(__file__),'answer_prompt_hash':ANSWER_PROMPT_HASH,
        'planner_prompt_hash':stable_hash(PLANNER_PROMPT),'generation':cfg['generation'],
        'planner_max_tokens':256,'planner_calls_per_question':1,'answer_calls_per_arm_question':1,
        'budgets':{'ce_candidates':48,'raw_episodes':8,'planner_inventory_roots':24,'planner_inventory_with_links':48,'state_witnesses':2},
        'source_adapter':'original_frozen_text_only','native_judge':'deferred_by_user','benchmark_retuning':False,
        'exposure':'repeated-look development; not pristine holdout','full_runs_authorized':False,
        'binding_limitation':'observed structural validation; planner semantics not independently certified',
        'latency_scope':'fresh single-pass, no result/CE memoization; repeated paper efficiency not claimed'}
    if planner_profile=='explicit':
        audit.update(planner_profile='explicit_six_field_contract',
            planner_contract_sha256=digest_file(ROOT/'src/mindset/acl/planner_contract.py'),
            planner_prompt_hash=stable_hash(PLANNER_PROMPT+'\n\n'+FIELD_CONTRACT))
    if inherit:
        audit.update(answer_cache_policy='terminal prior exact full requests; not efficiency',
            inherited_answer_source_audit_sha256=digest_file(Path(inherit)/'audit.json'),
            latency_scope='quality screening with exact answer-cache reuse; no paired efficiency claim')
    import fcntl
    with (out/'worker.lock').open('a+b') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(out/'audit.json',audit,immutable=True); record_environment(out/'continuations'/os.environ['SLURM_JOB_ID'])
        if inherit: inherit_answers(inherit,out,audit)
        done=unique_rows(out/'predictions.jsonl'); plans=unique_rows(out/'plans.jsonl')
        expected={f"{r['dataset']}/{m}/{r['id']}" for r in cohort['records'] for m in ARMS}
        if set(done)-expected: raise ValueError('Extra A/B cells')
        # Model caches remain separate; no downloads and no secrets transfer.
        gpu_cache=os.environ['HF_HOME']; os.environ['HF_HOME']=str(Path.home()/'MINDSET/.cache/huggingface')
        encoder, ce=local_stack(cfg,cache=False)
        os.environ['HF_HOME']=gpu_cache
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(checkpoint['checkpoint'],revision=checkpoint['revision'],
            cache_dir=str(Path(gpu_cache)/'hub'),local_files_only=True)
        # Separate arm caches avoid measuring B's fallback with a free A answer.
        answer_clients={arm:server_client(cfg['primary_generator'],cfg,
            cache_path=out/f'answer_{arm}_cache.sqlite') for arm in ARMS}
        planner_client=server_client(cfg['primary_generator'],cfg,cache_path=out/'planner_cache.sqlite')
        planner_client.allow_incomplete=True
        for client in answer_clients.values(): client.allow_incomplete=True
        info=request_json(os.environ['MINDSET_LOCAL_BASE_URL'].rstrip('/')+'/models')
        if {r['id'] for r in info['data']}!={checkpoint['checkpoint']}: raise ValueError('Wrong served checkpoint')
        prior=unique_rows(ROOT/'outputs/acl/ten_ideas_r02/merged/retrieval.jsonl')
        for dataset, cluster, cases in groups_for(cohort,cfg):
            if all(f'{dataset}/{m}/{c.id}' in done for c in cases for m in ARMS): continue
            events=cases[0].events
            print(f'BUILD {dataset}/{cluster} episodes={len(events)}',flush=True)
            vectors=encoder.encode([_render_event(e) for e in events],batch_size=64,normalize_embeddings=True,
                convert_to_numpy=True,show_progress_bar=False)
            memory=JointMindsetMemory(parameters,encoder=encoder,local_reranker=ce,
                precomputed_embeddings={e.event_id:v for e,v in zip(events,vectors,strict=True)})
            memory.ingest(events); memory.validate_invariants(); before=state_fingerprint(memory)
            write_json(out/'state'/f'{dataset}-{cluster}.json', {'state_sha256':before,
                'episodes':len(memory.episodes),'schemas':len(memory.schemas),
                'operations':[r.operation for r in memory.operation_log.all()]},immutable=True)
            write_json(out/'continuations'/os.environ['SLURM_JOB_ID']/'ingestion'/f'{dataset}-{cluster}.json',
                {'telemetry':memory.telemetry()})
            index=EvidenceCache(events,encoder,tokenizer,cfg)
            for case in cases:
                for arm in ARMS:
                    key=f'{dataset}/{arm}/{case.id}'
                    if key in done: continue
                    started=perf_counter(); planner=None; planner_cached=False
                    execution=fallback('baseline_no_planner')
                    if arm=='joint': hits=memory.retrieve(case.question,8)
                    else:
                        rows=inventory(memory,case.question); pmsgs=prompt_builder(case.question,rows,case.question_date)
                        phash=stable_hash(pmsgs); plan_key=f'{dataset}/{case.id}'
                        if plan_key in plans:
                            saved=plans[plan_key]
                            if saved['prompt_sha256']!=phash: raise ValueError('Changed resumed planner inventory')
                            plan=saved['plan']; planner_cached=True
                            planner=saved['completion']
                        else:
                            planner_client.options['response_format']={'type':'json_schema','json_schema':{
                                'name':'memory_access_plan','strict':True,'schema':planner_schema(rows)}}
                            count=prompt_tokens(tokenizer,pmsgs,thinking=False)
                            if count+256>16384:
                                plan=None; planner={'input_tokens':0,'output_tokens':0,'latency_seconds':0,
                                    'cache_hit':False,'incomplete':False,'reason':'inventory_context_overflow'}
                            else:
                                result=planner_client.complete(pmsgs,max_tokens=256,seed=cfg['seed'])
                                if result.input_tokens!=count: raise ValueError('Planner token count mismatch')
                                planner=asdict(result); planner_cached=result.cache_hit
                                try: plan=json.loads(result.text) if not result.incomplete else None
                                except json.JSONDecodeError: plan=None
                            saved={'id':plan_key,'prompt_sha256':phash,'inventory':rows,'messages':pmsgs,
                                'plan':plan,'completion':planner,'source_job_id':os.environ['SLURM_JOB_ID']}
                            append_row(out/'plans.jsonl',saved); plans[plan_key]=saved
                        execution=execute(memory,case.question,plan,rows)
                        hits=retrieve(memory,case.question,execution,8)
                    selected,packet=index.packet(hits,case.question); retrieval_seconds=perf_counter()-started
                    candidate_ids={memory._episode_order[i] for i in memory.last_candidate_pool}
                    if len(candidate_ids)>48 or any(s['event_id'] not in candidate_ids for s in selected):
                        raise ValueError('Evidence bypasses bounded CE pool')
                    if arm=='joint' and selected!=prior[key]['selected']: raise ValueError('Frozen baseline evidence regression')
                    # Exact immutable source spans, never planner/schema text.
                    original={e.event_id:e for e in events}
                    for s in selected:
                        ids=tokenizer.encode(render_event(original[s['event_id']]),add_special_tokens=False)
                        if not 0<=s['token_start']<s['token_end']<=len(ids) or tokenizer.decode(ids[s['token_start']:s['token_end']])!=s['text']:
                            raise ValueError('Non-source answer evidence')
                    msgs=messages(case.question,packet,case.question_date)
                    count=prompt_tokens(tokenizer,msgs,thinking=False)
                    if count+cfg['generation']['max_output_tokens']>16384: raise ValueError('Answer context cap exceeded')
                    completion=answer_clients[arm].complete(msgs,max_tokens=cfg['generation']['max_output_tokens'],seed=cfg['seed'])
                    if completion.input_tokens!=count: raise ValueError('Answer token count mismatch')
                    native=None
                    if dataset=='locomo':
                        category=next(int(k) for k,v in CATEGORY_NAMES.items() if v==case.category)
                        native=locomo_native_score(completion.text,case.answer,category)
                    row={'id':key,'dataset':dataset,'method':arm,'question_id':case.id,'cluster':cluster,
                        'category':case.category,'question':case.question,'question_date':case.question_date,'answer':case.answer,
                        'selected':selected,'evidence':packet,'metrics':rank_metrics(selected,case,8),
                        'prediction':completion.text,'native_f1':native,'token_f1':token_f1(completion.text,case.answer),
                        'prompt_sha256':stable_hash(msgs),'input_tokens':completion.input_tokens,'output_tokens':completion.output_tokens,
                        'answer_cache_hit':completion.cache_hit,'answer_incomplete':completion.incomplete,
                        'answer_latency_seconds':completion.latency_seconds,'answer_request_id':completion.request_id,
                        'planner_cache_hit':planner_cached,'planner_input_tokens':planner['input_tokens'] if planner else 0,
                        'planner_output_tokens':planner['output_tokens'] if planner else 0,
                        'planner_latency_seconds':planner['latency_seconds'] if planner else 0,
                        'retrieval_seconds':retrieval_seconds,'end_to_end_seconds':retrieval_seconds+completion.latency_seconds,
                        'execution':execution,'candidate_count':len(candidate_ids),'state_sha256':before,
                        'status':'development_not_paper','source_job_id':os.environ['SLURM_JOB_ID']}
                    append_row(out/'predictions.jsonl',row); done[key]=row
                    summarize(out)
                    print(f'CELL {len(done)}/200 {key} bound={execution["bound"]} reason={execution["reason"]}',flush=True)
            if state_fingerprint(memory)!=before: raise ValueError('Query mutated frozen state')
        if set(done)!=expected: raise ValueError('Incomplete A/B')
        return summarize(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',default=OUTPUT)
    parser.add_argument('--planner-profile',choices=('original','explicit'),default='original')
    parser.add_argument('--inherit-answers')
    args=parser.parse_args(); run(args.output,args.planner_profile,args.inherit_answers)
