"""Independent complete-panel provenance/component audit; DGX Slurm only."""
import os
from pathlib import Path
from collections import Counter
import numpy as np
from mindset.acl.pipeline import ROOT,record_environment
from mindset.acl.joint_iteration import protocol,groups_for
from mindset.acl.io import read_json,write_json,unique_rows,digest_file
from mindset.methods.semantic_witness import ARMS
from mindset.evaluation.metrics import token_f1
from mindset.acl.pipeline import rank_metrics
from mindset.baselines.event_rag import render_event

METRICS=('recall@8','precision@8','mrr@8','ndcg@8')
CONTRASTS=(('flat48','joint'),('semantic_pool','flat48'),('semantic_pool','no_verifier'),
           ('no_order','semantic_pool'),('semantic_cover','no_order'))


def paired_components(rows):
    """Paired descriptive arithmetic; no unpaired partial means/significance."""
    by={(r['dataset'],r['method'],r['question_id']):r for r in rows}
    if len(by)!=len(rows): raise ValueError('Duplicate component cells')
    reports={}
    for dataset in ('locomo','longmemeval'):
        panels={a:{r['question_id']:r for r in rows if r['dataset']==dataset and r['method']==a} for a in ARMS}
        ids=set(panels['joint'])
        if not ids or any(set(p)!=ids for p in panels.values()): raise ValueError('Incomplete paired components')
        reports[dataset]={}
        for a,b in CONTRASTS:
            membership=sum({s['event_id'] for s in panels[a][q]['selected']}!={s['event_id'] for s in panels[b][q]['selected']} for q in ids)
            order=sum([s['event_id'] for s in panels[a][q]['selected']]!=[s['event_id'] for s in panels[b][q]['selected']] for q in ids)
            reports[dataset][a+' minus '+b]={'paired_questions':len(ids),'membership_changes':membership,
                'order_or_membership_changes':order,'packet_changes':sum(panels[a][q]['evidence']!=panels[b][q]['evidence'] for q in ids),
                'mean_token_difference':float(np.mean([panels[a][q]['evidence_tokens']-panels[b][q]['evidence_tokens'] for q in ids])),
                **{m:float(np.mean([panels[a][q]['metrics'][m]-panels[b][q]['metrics'][m] for q in ids])) for m in METRICS},
                'candidate_recall@48':float(np.mean([panels[a][q]['candidate_recall@48']-panels[b][q]['candidate_recall@48'] for q in ids]))}
        active=[]
        for q in sorted(ids):
            cert=panels['no_order'][q]['certificate']; proposals=cert['proposals']; i=cert['selected_proposal']
            best=min(range(len(proposals)),key=lambda j:(tuple(proposals[j]['cost']),j))
            if i!=best: raise ValueError('Not minimum over logged finite proposals')
            if sorted(proposals[i]['selected_ids'])!=sorted(s['event_id'] for s in panels['no_order'][q]['selected']):
                raise ValueError('Selected source membership does not match certificate')
            full=panels['semantic_cover'][q]['certificate']
            if full!=cert: raise ValueError('Ordering ablation changed objective/proposals')
            if sorted(s['event_id'] for s in panels['semantic_cover'][q]['selected'])!=sorted(proposals[i]['selected_ids']):
                raise ValueError('Ordering changed membership')
            if i:
                active.append({'question_id':q,'mode':cert['mode'],'selected_proposal':i,
                    'base_cost':proposals[0]['cost'],'selected_cost':proposals[i]['cost'],
                    'missing_witness_reduction':proposals[0]['cost'][0]-proposals[i]['cost'][0],
                    'redundancy_reduction':proposals[0]['cost'][1]-proposals[i]['cost'][1],
                    'selected_ids':proposals[i]['selected_ids'],'target_edges':cert['target_edges']})
        reports[dataset]['objective_diagnostic']={'selected_nondefault_proposals':len(active),'active_cases':active,
            'queries_reducing_missing_witnesses':sum(r['missing_witness_reduction']>0 for r in active),
            'queries_reducing_report_redundancy':sum(r['redundancy_reduction']>0 for r in active),
            'intent_counts':dict(Counter(panels['no_order'][q]['certificate']['mode'] for q in ids)),
            'contrast_queries':sum(any(e['relation']=='contrast' for e in panels['joint'][q]['semantic_edges']) for q in ids),
            'interpretation':'Finite logged-proposal minimum only; NLI labels are not truth or verified state transitions'}
    return reports


def run():
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('DGX Slurm only')
    source=ROOT/'outputs/acl/semantic_witness_r01'
    manifest=ROOT/'data/manifests/mindset_v2_development100.json'
    calibration=ROOT/'outputs/acl/mindset_joint_v2_27058/calibration/calibration.json'
    cfg,cohort,_=protocol(manifest,calibration)
    if read_json(source/'status.json')['status']!='complete_development': raise ValueError('Source screening not terminal')
    import fcntl
    with (source/'worker.lock').open('rb') as handle:
        fcntl.flock(handle,fcntl.LOCK_SH|fcntl.LOCK_NB)
        rows=list(unique_rows(source/'retrieval.jsonl').values()); file_hash=digest_file(source/'retrieval.jsonl')
    expected={f"{r['dataset']}/{arm}/{r['id']}" for r in cohort['records'] for arm in ARMS}
    if {r['id'] for r in rows}!=expected or len(rows)!=600: raise ValueError('Wrong complete original100 panel')
    checkpoint=read_json(ROOT/'configs/checkpoints.lock.json')[cfg['evidence_tokenizer']]
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(checkpoint['checkpoint'],revision=checkpoint['revision'],
        cache_dir=str(Path.home()/'MINDSET_v2/.cache/gpu_prototype/huggingface/hub'),local_files_only=True)
    cases={(d,c.id):c for d,cluster,group in groups_for(cohort,cfg) for c in group}
    checked_passages=0; checked_edges=0; edge_examples=[]
    for row in rows:
        case=cases[(row['dataset'],row['question_id'])]; events={e.event_id:e for e in case.events}
        if row['candidate_count']>48 or len(row['selected'])>8: raise ValueError('Hard budget exceeded')
        packet='\n\n'.join(f'[{i}] {s["text"]}' for i,s in enumerate(row['selected'],1))
        if packet!=row['evidence']: raise ValueError('Unverifiable packet serialization')
        count=len(tokenizer.encode(packet,add_special_tokens=False))
        if count!=row['evidence_tokens'] or count>row['matched_token_ceiling']: raise ValueError('Unverifiable token ceiling')
        for s in row['selected']:
            event=events[s['event_id']]; tokens=tokenizer.encode(render_event(event),add_special_tokens=False)
            a,b=s['token_start'],s['token_end']
            if not 0<=a<b<=len(tokens) or tokenizer.decode(tokens[a:b])!=s['text']: raise ValueError('Non-source token span')
            if (s['speaker'],s['timestamp'],s['session_id'])!=(event.speaker,event.timestamp,event.session_id):
                raise ValueError('Source attribution mismatch')
            checked_passages+=1
        if rank_metrics(row['selected'],case,8)!=row['metrics']: raise ValueError('Saved ranking metrics disagree with sources')
        for edge in row['semantic_edges']:
            left,right=events[edge['left']],events[edge['right']]
            for side,event in (('left',left),('right',right)):
                a,b=edge[side+'_span']
                if not 0<=a<b<=len(event.text) or event.text[a:b]!=edge[side+'_text']:
                    raise ValueError('Invented source semantic span')
            if left.speaker.casefold()!=right.speaker.casefold() or left.speaker.casefold()!=edge['speaker']:
                raise ValueError('Invalid same-author report attribution')
            checked_edges+=1
            if row['method']=='joint':
                edge_examples.append({'dataset':row['dataset'],'question_id':case.id,'question':case.question,
                    'edge':edge,'source_timestamps':[left.timestamp,right.timestamp],
                    'warning':'Unadjudicated model proxy; different contexts/times need not imply world-state replacement'})
    if digest_file(source/'retrieval.jsonl')!=file_hash: raise ValueError('Sealed screening changed')
    report={'source_sha256':file_hash,'source_audit_sha256':digest_file(source/'audit.json'),
        'cells':600,'questions':100,'checked_raw_passages':checked_passages,'checked_literal_edge_spans':checked_edges,
        'paired_components':paired_components(rows),'unadjudicated_source_edge_examples':edge_examples,
        'scope':'independent provenance/arithmetic audit, exposed dev100; no QA gain or first-ever claim',
        'full_runs_authorized':False}
    out=ROOT/'outputs/acl'/f'semantic_witness_audit_{os.environ["SLURM_JOB_ID"]}'
    record_environment(out); write_json(out/'audit.json',report)
    print('AUDITED',report['cells'],report['questions'],report['checked_raw_passages'],flush=True)
    for dataset,components in report['paired_components'].items():
        print('COMPONENTS',dataset,components,flush=True)


if __name__=='__main__': run()
