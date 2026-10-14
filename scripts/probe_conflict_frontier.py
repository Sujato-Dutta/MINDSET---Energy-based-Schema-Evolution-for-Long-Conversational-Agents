"""Energy-free, matched KU78 development retrieval with raw recency controls."""
from pathlib import Path
from collections import defaultdict
import argparse
import json
import os
import sys
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from mindset.acl.knowledge_update import protocol
from mindset.acl.joint_iteration import groups_for
from mindset.acl.pipeline import local_stack,record_environment,rank_metrics,write_csv
from mindset.acl.io import read_json,write_json,unique_rows,digest_file,append_row
from mindset.acl.runtime import CachedEncoder
from mindset.acl.evidence import EvidenceCache
from mindset.acl.clients import load_tokenizer,messages,ANSWER_PROMPT_HASH
from mindset.methods.report_grounding import PredicateEquivalence,inject_witnesses
from mindset.methods.conflict_frontier import ConflictFrontierCatalog,LeanEpisodeIndex,select_frontier,known_time
from mindset.methods.base import MemoryHit
from mindset.utils.hashing import stable_hash

ARMS=('frontier','no_opaque_tail','no_contrast','latest_report','raw_latest_two','common_flat','flat_original')
METRICS=('recall@8','precision@8','mrr@8','ndcg@8')


def summarize(out,count):
    rows=list(unique_rows(out/'evidence.jsonl').values())
    summary=[]; pairs=[]
    primary={r['question_id']:r for r in rows if r['arm']=='frontier'}
    for arm in ARMS:
        subset=[r for r in rows if r['arm']==arm]
        if not subset: continue
        summary.append({'arm':arm,'questions':len(subset),
            **{m:float(np.mean([r['retrieval'][m] for r in subset])) for m in METRICS},
            'candidate_recall@48':float(np.mean([r['candidate_recall'] for r in subset])),
            'mean_evidence_tokens':float(np.mean([r['evidence_tokens'] for r in subset]))})
        if arm!='frontier':
            paired=[(primary[r['question_id']],r) for r in subset if r['question_id'] in primary]
            pairs.append({'control':arm,'pairs':len(paired),
                'identical_complete_prompts':sum(a['prompt_sha256']==b['prompt_sha256'] for a,b in paired),
                **{m+'_delta':float(np.mean([a['retrieval'][m]-b['retrieval'][m] for a,b in paired])) for m in METRICS}})
    if summary: write_csv(out/'retrieval.csv',summary)
    if pairs: write_csv(out/'paired_packets.csv',pairs)
    write_json(out/'status.json',{'completed_cells':len(rows),'expected_cells':count*len(ARMS),
        'status':'complete_development' if len(rows)==count*len(ARMS) else 'partial_development',
        'full_runs_authorized':False,'qa_calls':0,'judge_calls':0,'energy_used':False})


def main(a):
    cfg,cohort,_=protocol(a.manifest,a.calibration); out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    p=Path(a.projections); n=Path(a.nli)
    if read_json(p/'status.json')['status']!='complete_projection': raise ValueError('Incomplete input')
    if read_json(n/'status.json')['status']!='complete_nli_cache': raise ValueError('Incomplete predicate cache')
    if read_json(n/'audit.json')['projection_sha256']!=digest_file(p/'reports.jsonl'): raise ValueError('NLI/input mismatch')
    frozen_nli=read_json(ROOT/'outputs/acl/report_nli_27346/audit.json')
    actual_nli=read_json(n/'audit.json')
    if any(actual_nli[k]!=frozen_nli[k] for k in ('repository','revision','label_mapping','model_files')):
        raise ValueError('Frozen predicate model changed')
    sources=defaultdict(dict); queries=defaultdict(dict)
    for row in unique_rows(p/'reports.jsonl').values():
        key=(row['dataset'],row['cluster'])
        if row['kind']=='episode': sources[key][row['source_id']]=row
        else: queries[key][row['question_id']]=row
    if {(d,r['question_id']) for (d,c),qs in queries.items() for r in qs.values()}!={(r['dataset'],r['id']) for r in cohort['records']}:
        raise ValueError('Frozen cohort mismatch')
    with (n/'pairs.jsonl').open() as f: eq=PredicateEquivalence([json.loads(line) for line in f])
    prior_path=ROOT/'outputs/acl/broad_grounding_validation_27370/evidence.jsonl'
    prior={r['question_id']:r for r in unique_rows(prior_path).values() if r['arm']=='previous'}
    write_json(out/'audit.json',{'manifest_sha256':digest_file(a.manifest),'projection_sha256':digest_file(p/'reports.jsonl'),
        'nli_audit':read_json(n/'audit.json'),'nli_sha256':digest_file(n/'pairs.jsonl'),
        'worker_sha256':digest_file(__file__),'controller_sha256':digest_file(ROOT/'src/mindset/methods/conflict_frontier.py'),
        'typed_catalog_sha256':digest_file(ROOT/'src/mindset/methods/typed_grounding.py'),
        'contract_sha256':digest_file(ROOT/'src/mindset/memory/query_contracts.py'),
        'prior_sha256':digest_file(prior_path),'prior_control':'previous unchanged flat48',
        'arms':ARMS,'candidate_budget':48,'episode_budget':8,
        'new_fitted_scalars':0,'answer_prompt_sha256':ANSWER_PROMPT_HASH,'energy_used':False,
        'source_scope':'same raw source prefix; exact typed projection; chronological opaque guard',
        'exposure':'already exposed KU78, repeated-look development, NOT unseen confirmation',
        'timing_scope':'quality run, history-local embedding reuse, projection time excluded; NOT paper efficiency',
        'full_runs_authorized':False,'qa_calls':0},immutable=True)
    record_environment(out); encoder,reranker=local_stack(cfg,cache=False); encoder=CachedEncoder(encoder)
    tokenizer=load_tokenizer(read_json(ROOT/'configs/checkpoints.lock.json')[cfg['evidence_tokenizer']]); completed=0
    for dataset,cluster,cases in groups_for(cohort,cfg):
        events=cases[0].events; key=(dataset,cluster); started=perf_counter()
        index=LeanEpisodeIndex(events,encoder); catalog=ConflictFrontierCatalog(sources[key],events,eq)
        write_json(out/'structure'/f'{cluster}.json',{'report_groups':catalog.groups,'decisions':catalog.audit,
            'index_build_seconds':perf_counter()-started,'timing_scope':'quality only; reused projection'})
        cache=EvidenceCache(events,encoder,tokenizer,cfg)
        for case in cases:
            qr=queries[key][case.id]
            if qr['text']!=case.question: raise ValueError('Question mismatch')
            flat=index.candidates(case.question,encoder); choices={}
            for arm,kwargs in (('frontier',{}),('no_opaque_tail',{'opaque':False}),('no_contrast',{'contrast':False}),
                               ('latest_report',{'opaque':False,'contrast':False})):
                ids,certificate=catalog.frontier(qr,**kwargs); choices[arm]=(ids,certificate)
            # Strong simple control: same literal scope, no parsed-report basis.
            deadline=known_time(case.question_date)
            raw=[eid for eid in catalog.scoped_sources(qr['report_query']['contract'])
                 if known_time(catalog.by_id[eid].timestamp) is not None
                 and (deadline is None or known_time(catalog.by_id[eid].timestamp)<=deadline)]
            raw.sort(key=lambda eid:(known_time(catalog.by_id[eid].timestamp),catalog.source_order[eid]),reverse=True)
            if qr['report_query']['mode']!='state_current': raw=[]
            choices['raw_latest_two']=(raw[:2],{'selector':'raw_latest_two','reserved_ids':raw[:2],'world_truth_certified':False})
            choices['common_flat']=([],{'reserved_ids':[]}); choices['flat_original']=([],{'reserved_ids':[]})
            anchors=list(dict.fromkeys(index.by_id[eid] for ids,_ in choices.values() for eid in ids))
            common=inject_witnesses(flat,anchors); union=sorted(set(flat)|set(common))
            scores=reranker.predict([(case.question,index.documents[i]) for i in union],batch_size=8,show_progress_bar=False)
            if len(scores)!=len(union) or not np.isfinite(scores).all(): raise ValueError('Bad CE scores')
            scoremap=dict(zip(union,scores,strict=True)); tie={i:index.ids[i] for i in union}
            units={e.event_id:e.session_id if case.retrieval_unit=='session' else e.event_id for e in events}
            for arm in ARMS:
                pool=flat if arm=='flat_original' else common; ids,certificate=choices[arm]
                chosen=select_frontier(pool,[scoremap[i] for i in pool],[index.by_id[eid] for eid in ids],tie_keys=tie)
                hits=[MemoryHit(index.ids[i],index.documents[i],float(scoremap[i]),(index.ids[i],),'raw_source_frontier') for i in chosen]
                selected,packet=cache.packet(hits,case.question); prompt=messages(case.question,packet,case.question_date)
                row={'id':f'{arm}/{case.id}','dataset':dataset,'cluster':cluster,'question_id':case.id,
                    'question':case.question,'question_date':case.question_date,'category':case.category,'answer':case.answer,
                    'arm':arm,'selected':selected,'evidence':packet,'messages':prompt,'prompt_sha256':stable_hash(prompt),
                    'evidence_tokens':len(tokenizer.encode(packet,add_special_tokens=False)),
                    'candidate_ids':[index.ids[i] for i in pool],
                    'candidate_recall':len({units[index.ids[i]] for i in pool}&set(case.gold))/len(set(case.gold)),
                    'retrieval':rank_metrics(selected,case,8),'frontier_certificate':certificate}
                if arm=='flat_original' and row['prompt_sha256']!=prior[case.id]['prompt_sha256']:
                    raise ValueError('Lean index did not reproduce prior flat48 prompt')
                append_row(out/'evidence.jsonl',row); completed+=1
            summarize(out,cohort['count']); print(f'{completed}/{cohort["count"]*len(ARMS)} frontier cells',flush=True)
        encoder.cache.clear()
    summarize(out,cohort['count'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default=str(ROOT/'data/manifests/mindset_v2_knowledge_update_validation.json'))
    for name in ('calibration','projections','nli','output'): p.add_argument('--'+name,required=True)
    a=p.parse_args()
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    main(a)
