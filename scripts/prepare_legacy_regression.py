"""DGX-only, source-audited retained-v1 versus joint LC50 diagnostic.

Not an exact historical checkout: use retained v1 defaults and recorded read
settings, and separately reuse all shared historical selected episode IDs.
No labels or answers enter controller construction or source selection.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def historical_hits(case,record):
    """Saved selected IDs, not gold IDs. Reject changed questions/references."""
    from mindset.methods.base import MemoryHit
    from mindset.baselines.event_rag import render_event
    from mindset.datasets.locomo import CATEGORY_NAMES
    if record['id']!=case.id or record['question']!=case.question or record['answer']!=case.answer:
        raise ValueError('Historical question/reference changed')
    if CATEGORY_NAMES[str(record['category'])]!=case.category: raise ValueError('Historical category changed')
    original={e.event_id:e for e in case.events}
    ids=record['retrieved_ids']
    if len(ids)>8 or len(set(ids))!=len(ids) or any(i not in original for i in ids):
        raise ValueError('Invalid historical selected episode IDs')
    return [MemoryHit(i,render_event(original[i]),float(len(ids)-rank),(i,),'historical_selected_raw_episode',{})
            for rank,i in enumerate(ids)]


def audit_answers(current,legacy,cases,out):
    from mindset.acl.io import write_json
    from mindset.evaluation.locomo import locomo_native_score,locomo_normalize
    from mindset.datasets.locomo import CATEGORY_NAMES
    from mindset.acl.pipeline import write_csv
    summary=[]; details=[]
    for method in ('joint','flat48'):
        subset=[r for r in current if r['dataset']=='locomo' and r['method']==method]
        if {r['question_id'] for r in subset}!=set(cases): raise ValueError('Incomplete LC50 answer panel')
        for row in subset:
            case=cases[row['question_id']]
            category=next(int(k) for k,v in CATEGORY_NAMES.items() if v==case.category)
            score=locomo_native_score(row['prediction'],case.answer,category)
            if abs(score-row['native_f1'])>1e-12: raise ValueError('Native answer score changed')
            chosen=[s['event_id'] for s in row['selected']]; gold=set(case.gold)
            reference=set(locomo_normalize(case.answer).split())
            evidence=set(locomo_normalize(row['evidence']).split())
            old=legacy.get(case.id)
            details.append({'method':method,'question_id':case.id,'category':case.category,'cluster':case.cluster,
                'question':case.question,'reference':case.answer,'prediction':row['prediction'],'native_f1':score,
                'gold_ids':case.gold,'selected_ids':chosen,'missing_gold_ids':sorted(gold-set(chosen)),
                'retrieval':row['retrieval'],'selected':row['selected'],
                'lexical_reference_coverage_proxy':len(reference&evidence)/len(reference) if reference else None,
                'abstention':('no information available' in row['prediction'].casefold()),
                'historical_prediction':old['prediction'] if old else None,
                'historical_selected_ids':old['retrieved_ids'] if old else None,
                'interpretation':'annotation presence and lexical overlap are not proof of semantic evidence sufficiency'})
        ds=[d for d in details if d['method']==method]
        summary.append({'method':method,'questions':len(ds),'native_f1':sum(d['native_f1'] for d in ds)/len(ds),
            'abstentions':sum(d['abstention'] for d in ds),'zero_f1':sum(d['native_f1']==0 for d in ds),
            'all_gold_ids_selected':sum(not d['missing_gold_ids'] for d in ds),
            'zero_f1_with_all_gold_ids_selected':sum(d['native_f1']==0 and not d['missing_gold_ids'] for d in ds),
            **{m:sum(d['retrieval'][m] for d in ds)/len(ds) for m in ('recall@8','precision@8','mrr@8','ndcg@8')}})
    write_json(out/'answer_error_audit.json',{'scope':'All LC50; evaluator-only references; no generation or tuning',
        'summary':summary,'cases':sorted(details,key=lambda d:(d['method'],d['native_f1'],d['question_id']))},immutable=True)
    write_csv(out/'answer_error_summary.csv',summary)
    for row in summary: print('ANSWER_AUDIT',row,flush=True)
    for row in sorted((d for d in details if d['method']=='joint'),key=lambda d:(d['native_f1'],d['question_id']))[:8]:
        print('INSPECT_CASE',row,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    from mindset.acl.io import read_json,read_rows,write_json,unique_rows,append_row,digest_file
    from mindset.acl.joint_iteration import protocol,groups_for
    from mindset.acl.pipeline import local_stack,record_environment,rank_metrics,write_csv
    from mindset.acl.runtime import CachedEncoder
    from mindset.acl.clients import load_tokenizer
    from mindset.acl.evidence import EvidenceCache
    from mindset.methods.mindset import MindsetMemory,MindsetConfig
    from mindset.methods.rerank_cache import CachedReranker
    from mindset.utils.hashing import stable_hash
    manifest=ROOT/'data/manifests/mindset_v2_development100.json'
    calibration=ROOT/'outputs/acl/mindset_joint_v2_27058/calibration/calibration.json'
    cfg,cohort,_=protocol(manifest,calibration)
    legacy_dir=ROOT/'outputs/acl/legacy_score_inputs_r01'; args=read_json(legacy_dir/'run_config.json')['arguments']
    if (args['method'],args['ablation'],args['mindset_retrieval_mode'],args['top_k'],args['candidate_k'])!=('mindset','full','cross_encoder_only',8,40):
        raise ValueError('Unexpected historical experiment')
    old_config=read_json(legacy_dir/'run_config.json')
    if old_config['provenance']['dataset_sha256']!=digest_file(ROOT/cfg['benchmarks']['locomo']['path']):
        raise ValueError('Historical source differs')
    legacy={r['id']:r for r in read_rows(legacy_dir/'predictions.jsonl')}
    if len(legacy)!=700: raise ValueError('Historical question panel incomplete/duplicate')
    cases={}; groups=[]
    for dataset,cluster,queries in groups_for(cohort,cfg):
        if dataset=='locomo': groups.append((cluster,queries)); cases.update({c.id:c for c in queries})
    prepared=ROOT/'outputs/acl/mindset_joint_v2_27058/quality'
    if read_json(prepared/'status.json')['status']!='complete_development': raise ValueError('Incomplete current preparation')
    reference=list(unique_rows(prepared/'retrieval.jsonl').values())
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    parameters=MindsetConfig(ablation=args['ablation'],retrieval_mode=args['mindset_retrieval_mode'],retrieval_candidate_k=args['candidate_k'])
    import fcntl
    with (out/'worker.lock').open('a+b') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        audit={'kind':'retained_v1_regression_diagnosis','manifest_sha256':digest_file(manifest),
            'legacy_predictions_sha256':digest_file(legacy_dir/'predictions.jsonl'),
            'legacy_run_config_sha256':digest_file(legacy_dir/'run_config.json'),
            'prepared_joint_sha256':digest_file(prepared/'retrieval.jsonl'),
            'worker_sha256':digest_file(__file__),'retained_v1_sha256':digest_file(ROOT/'src/mindset/methods/mindset.py'),
            'parameters':asdict(parameters),'checkpoints':read_json(ROOT/'configs/checkpoints.lock.json'),
            'limitations':['Retained code reconstruction; exact historical source checkout unavailable',
                'Historical selected IDs are reused with current raw formatting, not recovered original API prompt',
                'Historical encoder/CE revisions were not recorded; current pinned revisions are used',
                'Historical and v2 internal CE candidate budgets are not assumed compute-matched'],
            'methods':{'joint':50,'flat48':50,'retained_v1':50,'historical_ids':len(set(cases)&set(legacy))},
            'selection':'original LC50; historical_ids arm uses every available intersection, never score selection',
            'qa_calls':0,'judge_calls':0,'retuned':False,'full_runs_authorized':False}
        write_json(out/'audit.json',audit,immutable=True); record_environment(out)
        audit_answers(list(unique_rows(ROOT/'outputs/acl/development100_answers_r01/predictions.jsonl').values()),legacy,cases,out)
        done=unique_rows(out/'retrieval.jsonl')
        for r in reference:
            if r['dataset']=='locomo' and r['method'] in ('joint','flat48') and r['id'] not in done:
                append_row(out/'retrieval.jsonl',r); done[r['id']]=r
        encoder,reranker=local_stack(cfg,cache=False); encoder=CachedEncoder(encoder,enabled=True)
        reranker=CachedReranker(reranker,stable_hash(audit['checkpoints']['reranker']),out/'pair_scores.sqlite')
        tokenizer=load_tokenizer(audit['checkpoints'][cfg['evidence_tokenizer']])
        for cluster,queries in groups:
            cluster_expected={f'locomo/{m}/{c.id}' for c in queries for m in ('retained_v1','historical_ids')
                              if m!='historical_ids' or c.id in legacy}
            if cluster_expected<=set(done): continue
            events=queries[0].events
            # Match the old VectorRAGIndex embedding serialization, not a new schema summary.
            vectors=encoder.encode([f"[{e.timestamp or 'date unknown'}] {e.speaker}: {e.text}" for e in events],
                batch_size=64,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
            memory=MindsetMemory(parameters,encoder=encoder,local_reranker=reranker,
                precomputed_embeddings={e.event_id:v for e,v in zip(events,vectors)})
            memory.ingest(events)
            snapshot=out/'structure'/f'{cluster}.json'
            invariants=memory.validate_invariants()
            if snapshot.exists():
                if read_json(snapshot)['invariants']!=invariants: raise ValueError('Reconstructed state invariants changed')
            else: write_json(snapshot,{'invariants':invariants,'telemetry':memory.telemetry()},immutable=True)
            index=EvidenceCache(events,encoder,tokenizer,cfg)
            for case in queries:
                for method in ('retained_v1','historical_ids'):
                    key=f'locomo/{method}/{case.id}'
                    if key in done or (method=='historical_ids' and case.id not in legacy): continue
                    if method=='historical_ids': hits=historical_hits(case,legacy[case.id]); candidate_count=None
                    else:
                        pool=memory.retrieve_candidate_pool(case.question); candidate_count=len(pool)
                        hits=memory.select_candidate_pool(case.question,pool,mode='cross_encoder_only',top_k=40)
                    selected,packet=index.packet(hits,case.question)
                    row={'id':key,'dataset':'locomo','method':method,'question_id':case.id,'cluster':cluster,
                        'category':case.category,'selected':selected,'evidence':packet,'metrics':rank_metrics(selected,case,8),
                        'candidate_count':candidate_count,'evidence_tokens':len(tokenizer.encode(packet,add_special_tokens=False)),
                        'status':'development_not_paper'}
                    if case.id in legacy:
                        row['historical_selected_ids']=legacy[case.id]['retrieved_ids']
                        row['matches_historical_selected_ids']=[s['event_id'] for s in selected]==legacy[case.id]['retrieved_ids']
                    append_row(out/'retrieval.jsonl',row); done[key]=row
                    print(f'PREPARED {len(done)}/{sum(audit["methods"].values())} {key}',flush=True)
            encoder.cache.clear()
        expected={f'locomo/{m}/{q}' for m in audit['methods'] for q in cases if m!='historical_ids' or q in legacy}
        if set(done)!=expected: raise ValueError('Regression panel incomplete or contains extra IDs')
        report=[]
        for method,count in audit['methods'].items():
            subset=[r for r in done.values() if r['method']==method]
            report.append({'method':method,'questions':len(subset),
                **{k:sum(r['metrics'][k] for r in subset)/len(subset) for k in ('recall@8','precision@8','mrr@8','ndcg@8')},
                'mean_evidence_tokens':sum(r['evidence_tokens'] for r in subset)/len(subset),
                'mean_ce_candidates':sum(r['candidate_count'] for r in subset)/len(subset) if method=='retained_v1' else None})
        write_csv(out/'retrieval.csv',report)
        write_json(out/'status.json',{'status':'complete_development','completed_cells':len(done),'expected_cells':len(expected),
            'unique_questions':50,'qa_calls':0,'judge_calls':0,'full_runs_authorized':False})


if __name__=='__main__': main()
