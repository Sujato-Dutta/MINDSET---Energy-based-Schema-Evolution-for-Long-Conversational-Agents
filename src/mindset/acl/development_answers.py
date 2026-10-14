"""Bounded, resumable native-answer diagnosis; not paper-frozen execution."""
from dataclasses import asdict
from pathlib import Path
import os
from mindset.acl.joint_iteration import protocol,groups_for
from mindset.acl.pipeline import ROOT,server_client,record_environment,write_csv
from mindset.acl.clients import messages,prompt_tokens,load_tokenizer,ANSWER_PROMPT_HASH
from mindset.acl.io import read_json,write_json,unique_rows,append_row,digest_file
from mindset.evaluation.metrics import token_f1
from mindset.evaluation.locomo import locomo_native_score
from mindset.datasets.locomo import CATEGORY_NAMES
from mindset.utils.hashing import stable_hash

METHODS=('joint','flat48','lightmem')
REGRESSION_METHODS=('joint','flat48','retained_v1','historical_ids')


def request_plan(rows,cases,methods=METHODS,*,reader='current'):
    """References remain evaluator metadata and never enter the request."""
    planned=[]; expected={(d,q,m) for d,q in cases for m in methods}; seen=set()
    for row in rows:
        if row['method'] not in methods: continue
        key=(row['dataset'],row['question_id'],row['method'])
        if key not in expected or key in seen: raise ValueError('Unexpected/duplicate development cell')
        seen.add(key); case=cases[key[:2]]
        if not row['selected'] or len(row['selected'])>8: raise ValueError('Invalid evidence packet size')
        original={e.event_id:e for e in case.events}
        for selected in row['selected']:
            if selected['event_id'] not in original: raise ValueError('Selected source not in frozen history')
            event=original[selected['event_id']]
            if selected['speaker']!=event.speaker or selected['timestamp']!=event.timestamp:
                raise ValueError('Source attribution mismatch')
        packet='\n\n'.join(f'[{i}] {s["text"]}' for i,s in enumerate(row['selected'],1))
        if packet!=row['evidence']: raise ValueError('Packet serialization mismatch')
        if reader=='current': prompt=messages(case.question,packet,case.question_date)
        elif reader=='locomo_short_phrase':
            if key[0]!='locomo': raise ValueError('LoCoMo instruction cannot be applied to LongMemEval')
            from mindset.acl.locomo_protocol_diagnostic import short_phrase_messages
            prompt=short_phrase_messages(case.question,packet,case.question_date,category=case.category)
        else: raise ValueError('Unknown diagnostic reader protocol')
        planned.append({'id':f'{key[0]}/{key[2]}/{key[1]}','dataset':key[0],'method':key[2],
            'question_id':key[1],'cluster':case.cluster,'category':case.category,'question':case.question,
            'question_date':case.question_date,'answer':case.answer,'retrieval':row['metrics'],
            'messages':prompt,'prompt_sha256':stable_hash(prompt),'evidence':packet,'selected':row['selected']})
    if seen!=expected: raise ValueError('Incomplete original development100 method panel')
    return sorted(planned,key=lambda row:row['id'])


def summarize(out,planned,methods=METHODS):
    rows=list(unique_rows(out/'predictions.jsonl').values()); report=[]
    for dataset in ('locomo','longmemeval'):
        for method in methods:
            subset=[r for r in rows if r['dataset']==dataset and r['method']==method]
            if not subset: continue
            expected=sum(p['dataset']==dataset and p['method']==method for p in planned)
            report.append({'dataset':dataset,'method':method,'questions':len(subset),'expected':expected,
                'native_f1':sum(r['native_f1'] for r in subset)/len(subset) if dataset=='locomo' else None,
                'secondary_token_f1':sum(r['token_f1'] for r in subset)/len(subset),
                'native_judge_accuracy':None,'judge_status':'not_run',
                'incomplete_response_count':sum(r.get('incomplete_response',False) for r in subset),
                'logical_input_tokens':sum(r['input_tokens'] for r in subset),
                'logical_output_tokens':sum(r['output_tokens'] for r in subset),
                'scope':'exposed development100; NOT paper results or efficiency'})
    if report: write_csv(out/'answer_metrics.csv',report)
    import sqlite3,json
    completed_requests=[]
    if (out/'completion_cache.sqlite').exists():
        with sqlite3.connect(out/'completion_cache.sqlite') as cache:
            completed_requests=[json.loads(r[0]) for r in cache.execute('SELECT payload FROM local_completions')]
    successful=[r for r in completed_requests if not r.get('incomplete',False)]
    write_json(out/'status.json',{'status':'complete_development_answers' if len(rows)==len(planned) else 'partial_development_answers',
        'completed_cells':len(rows),'expected_cells':len(planned),
        'unique_questions':len({(p['dataset'],p['question_id']) for p in planned}),
        'successful_unique_local_calls':len(successful),
        'actual_successful_input_tokens':sum(r['input_tokens'] for r in successful),
        'actual_successful_output_tokens':sum(r['output_tokens'] for r in successful),
        'persisted_unique_responses':len(completed_requests),
        'persisted_incomplete_responses':len(completed_requests)-len(successful),
        'actual_persisted_input_tokens':sum(r['input_tokens'] for r in completed_requests),
        'actual_persisted_output_tokens':sum(r['output_tokens'] for r in completed_requests),
        'usage_scope':'persisted responses include inherited successful requests and new capped/empty responses; failed/lost requests not inferred; not all usage is newly incurred',
        'judge_calls':0,'full_runs_authorized':False,
        'latency_scope':'completion cache/dedup/resume allowed; NOT efficiency/independent repeatability'})


def inherit_cache(source,out,identity,*,strict_plan=True):
    """Reuse only successful identical requests from a terminal prior run."""
    import sqlite3,fcntl
    source=Path(source); prior=read_json(source/'audit.json')
    keys=['manifest_sha256','calibration_sha256','checkpoint','checkpoint_lock_sha256','prompt_hash','generation','serving']
    if strict_plan: keys+=['prepared_sha256','methods','plan_sha256']
    for key in keys:
        # JSON turns tuple method lists into lists; compare canonical content.
        if stable_hash(prior[key])!=stable_hash(identity[key]): raise ValueError('Incompatible inheritance: '+key)
    destination=out/'completion_cache.sqlite'
    inherited={'source':str(source.resolve()),'source_audit_sha256':digest_file(source/'audit.json'),
               'strict_plan':strict_plan,
               'policy':'successes only; checkpoint/prompt/generation match; each hit requires exact full request key; no logical prediction inheritance'}
    write_json(out/'inheritance.json',inherited,immutable=True)
    if destination.exists(): return
    with (source/'worker.lock').open('rb') as handle:
        fcntl.flock(handle,fcntl.LOCK_SH|fcntl.LOCK_NB)  # Never copy from a live writer.
        with sqlite3.connect('file:'+str(source/'completion_cache.sqlite')+'?mode=ro',uri=True) as old:
            import json
            if any(json.loads(r[0]).get('incomplete',False) for r in old.execute('SELECT payload FROM local_completions')):
                raise ValueError('Inheritance source contains incomplete responses; successful cache required')
            with sqlite3.connect(destination) as new: old.backup(new)


def regression_plan(evidence,cases,prepared,*,reader='current'):
    """Three full LC panels plus all shared historical IDs, never cherry-picked."""
    audit=read_json(Path(prepared)/'audit.json')
    source=ROOT/'outputs/acl/legacy_score_inputs_r01/predictions.jsonl'
    if audit['kind']!='retained_v1_regression_diagnosis' or digest_file(source)!=audit['legacy_predictions_sha256']:
        raise ValueError('Wrong regression preparation or historical source')
    from mindset.acl.io import read_rows
    historical={r['id'] for r in read_rows(source)}
    cases={k:c for k,c in cases.items() if k[0]=='locomo'}
    if len(cases)!=50: raise ValueError('Regression requires the original LC50')
    planned=[]
    if {r['method'] for r in evidence}!=set(REGRESSION_METHODS): raise ValueError('Unexpected regression arms')
    for method in REGRESSION_METHODS:
        selected={k:c for k,c in cases.items() if method!='historical_ids' or k[1] in historical}
        planned+=request_plan(evidence,selected,methods=(method,),reader=reader)
    return sorted(planned,key=lambda row:row['id'])


def run(manifest,calibration,prepared,output,inherit=None,regression=False,reader='current',captions=False):
    cfg,cohort,_=protocol(manifest,calibration)
    if cohort['count']!=100: raise ValueError('Only the original development100 is authorized here')
    if cfg['primary_generator']!='qwen35_9b': raise ValueError('Primary checkpoint substitution prohibited')
    prepared=Path(prepared); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    if read_json(prepared/'status.json')['status']!='complete_development': raise ValueError('Incomplete prepared development')
    evidence=list(unique_rows(prepared/'retrieval.jsonl').values()); cases={}
    for dataset,cluster,queries in groups_for(cohort,cfg):
        for case in queries: cases[(dataset,case.id)]=case
    if reader!='current' or captions:
        if not regression: raise ValueError('Protocol factors are restricted to the bounded LC regression panel')
        if inherit: raise ValueError('Changed protocol factors require a fresh request cache, not inheritance')
    from mindset.acl.locomo_protocol_diagnostic import CAPTION_ADAPTER,READER_HASH,REFERENCE_URL,captioned_cases
    if read_json(prepared/'audit.json').get('source_adapter')!=(CAPTION_ADAPTER if captions else None):
        raise ValueError('Prepared source adapter differs from requested diagnostic')
    if captions: cases=captioned_cases(cases,cfg,ROOT)
    methods=REGRESSION_METHODS if regression else METHODS
    planned=regression_plan(evidence,cases,prepared,reader=reader) if regression else request_plan(evidence,cases)
    lock=read_json(ROOT/'configs/checkpoints.lock.json')
    checkpoint=lock[cfg['primary_generator']]
    prompt_hash=ANSWER_PROMPT_HASH if reader=='current' else READER_HASH
    identity={'scope':'retained_v1_regression_diagnosis' if regression else 'bounded_development100_native_answer_diagnosis',
        'prepared_sha256':digest_file(prepared/'retrieval.jsonl'),
        'manifest_sha256':digest_file(manifest),'calibration_sha256':digest_file(calibration),
        'worker_sha256':digest_file(__file__),'clients_sha256':digest_file(ROOT/'src/mindset/acl/clients.py'),
        'checkpoint':checkpoint,'checkpoint_lock_sha256':digest_file(ROOT/'configs/checkpoints.lock.json'),
        'prompt_hash':prompt_hash,'reader_protocol':reader,'source_adapter':CAPTION_ADAPTER if captions else 'original_frozen_text_only',
        'reader_reference':REFERENCE_URL if reader!='current' else None,
        'reader_limitation':'published question instruction adaptation; shared local serving/evidence/cap, NOT exact original full evaluation pipeline',
        'generation':cfg['generation'],'methods':methods,
        'plan_sha256':stable_hash(planned),'question_count':len({(r['dataset'],r['question_id']) for r in planned}),
        'logical_cells':len(planned),
        'bounded_response_policy':'score exact answer content at fixed cap, including truncated content; empty content scores zero; keep finish/usage flags',
        'cache_policy':'identical requests including auditable capped/empty diagnostic responses; NOT efficiency or repeatability',
        'serving':'BF16 pinned checkpoint; explicit prefetch CPU offload; 20GB MIG; 16k cap; no quantization',
        'exposure':'repeated-look known development, not confirmatory unseen evaluation','full_runs_authorized':False}
    import fcntl
    with (out/'worker.lock').open('a+b') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(out/'audit.json',identity,immutable=True)
        if inherit: inherit_cache(inherit,out,identity,strict_plan=not regression)
        record_environment(out/'continuations'/os.environ['SLURM_JOB_ID'])
        for dataset in sorted({p['dataset'] for p in planned}):
            write_json(out/dataset/'run_config.json',{'prepared_config':{'dataset':dataset,'status':'development_not_paper'},
                'model':cfg['primary_generator'],'checkpoint':checkpoint,'prompt_hash':prompt_hash,
                'generation':cfg['generation'],'development_audit':identity},immutable=True)
        done=unique_rows(out/'predictions.jsonl'); expected={r['id']:r for r in planned}
        if set(done)-set(expected): raise ValueError('Unexpected resumed cells')
        for key,row in done.items():
            if row['prompt_sha256']!=expected[key]['prompt_sha256']: raise ValueError('Resumed prompt changed')
        tokenizer=load_tokenizer(checkpoint); client=server_client(cfg['primary_generator'],cfg,cache_path=out/'completion_cache.sqlite')
        client.allow_incomplete=True
        # Each saved packet passage must be an exact decoded token span of its
        # immutable source, not a same-ID generated summary or changed text.
        from mindset.baselines.event_rag import render_event
        source_tokens={}
        for item in planned:
            original={e.event_id:e for e in cases[(item['dataset'],item['question_id'])].events}
            for selected in item['selected']:
                key=(item['dataset'],item['cluster'],selected['event_id'])
                if key not in source_tokens: source_tokens[key]=tokenizer.encode(render_event(original[selected['event_id']]),add_special_tokens=False)
                start,end=selected['token_start'],selected['token_end']; tokens=source_tokens[key]
                if not 0<=start<end<=len(tokens) or tokenizer.decode(tokens[start:end])!=selected['text']:
                    raise ValueError('Saved packet is not an exact immutable source token span')
        # Verify the server uses the explicit logical repository name, not a
        # different hosted fallback. Job startup pins both model/tokenizer SHA.
        from mindset.acl.pipeline import request_json
        info=request_json(os.environ['MINDSET_LOCAL_BASE_URL'].rstrip('/')+'/models')
        if {r['id'] for r in info['data']}!={checkpoint['checkpoint']}: raise ValueError('Server identifier mismatch')
        for item in planned:
            if item['id'] not in done:
                count=prompt_tokens(tokenizer,item['messages'],thinking=False)
                if count+cfg['generation']['max_output_tokens']>16384: raise ValueError('Context cap exceeded; no evidence truncation')
                result=client.complete(item['messages'],max_tokens=cfg['generation']['max_output_tokens'],seed=cfg['seed'])
                if result.input_tokens!=count: raise ValueError('Client/server token counts differ')
                native=None
                if item['dataset']=='locomo':
                    category=next(int(k) for k,v in CATEGORY_NAMES.items() if v==item['category'])
                    native=locomo_native_score(result.text,item['answer'],category)
                row={k:v for k,v in item.items() if k!='messages'}
                row.update(prediction=result.text,native_f1=native,token_f1=token_f1(result.text,item['answer']),
                    input_tokens=result.input_tokens,output_tokens=result.output_tokens,
                    completion_cache_hit=result.cache_hit,answer_request_id=result.request_id,
                    finish_reason=result.finish_reason,incomplete_response=result.incomplete,
                    answer_latency_seconds=result.latency_seconds,source_job_id=os.environ['SLURM_JOB_ID'],
                    status='development_not_paper',metric_scope='LME token F1 secondary; native LME judge not run')
                append_row(out/'predictions.jsonl',row); done[row['id']]=row
            # Repair only missing mirrors after an interrupted successful root
            # commit. No regeneration or overwriting of existing predictions.
            destination=out/item['dataset']/'predictions.jsonl'
            if item['id'] not in unique_rows(destination): append_row(destination,done[item['id']])
            summarize(out,planned,methods); print(f'{len(done)}/{len(planned)} native-development cells',flush=True)
        summarize(out,planned,methods)
