"""Does the state substrate earn its cost in source association? DGX Slurm only.

Four complete original100 retrieval panels. No NLI, schema generation, answer
generation, outcome-dependent admission or fitted mixture. Controls are NOT a
new-method novelty claim and Joint-vs-flat seeds is not an energy-only ablation.
"""
import os
from pathlib import Path
import numpy as np
from mindset.acl.pipeline import ROOT,local_stack,rank_metrics,write_csv,record_environment
from mindset.acl.joint_iteration import protocol,groups_for
from mindset.acl.io import read_json,write_json,unique_rows,append_row,digest_file
from mindset.acl.runtime import CachedEncoder
from mindset.acl.evidence import EvidenceCache
from mindset.methods.conflict_frontier import LeanEpisodeIndex
from mindset.methods.base import MemoryHit
from mindset.methods.semantic_witness import source_spans,semantic_pairs,extend_pool

ARMS=('joint_ref','expanded_joint_ref','flat_ce_seed','hybrid_seed')
METRICS=('recall@8','precision@8','mrr@8','ndcg@8')


def summarize(out,expected):
    rows=list(unique_rows(out/'retrieval.jsonl').values()); reports=[]
    if set(r['id'] for r in rows)-expected: raise ValueError('Unexpected source-seed cell')
    for d in ('locomo','longmemeval'):
        base={r['question_id']:r for r in rows if r['dataset']==d and r['method']=='expanded_joint_ref'}
        for a in ARMS:
            subset=[r for r in rows if r['dataset']==d and r['method']==a]
            if not subset: continue
            reports.append({'dataset':d,'method':a,'questions':len(subset),'expected':50,
                **{m:float(np.mean([r['metrics'][m] for r in subset])) for m in METRICS},
                'packet_changes_from_expanded_joint':sum(r['evidence']!=base[r['question_id']]['evidence'] for r in subset),
                'mean_evidence_tokens':float(np.mean([r['evidence_tokens'] for r in subset])),
                'mean_logical_ce_pairs':float(np.mean([r['logical_ce_pairs'] for r in subset])),
                'scope':'source-substrate/seed controls, exposed original100, NOT novelty or efficiency'})
    if reports: write_csv(out/'comparison.csv',reports)
    write_json(out/'status.json',{'status':'complete_development' if len(rows)==len(expected) else 'partial_development',
        'completed_cells':len(rows),'expected_cells':len(expected),'qa_calls':0,'NLI_calls':0,'judge_calls':0,
        'full_runs_authorized':False})


def run():
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('DGX Slurm only')
    manifest=ROOT/'data/manifests/mindset_v2_development100.json'
    calibration=ROOT/'outputs/acl/mindset_joint_v2_27058/calibration/calibration.json'
    cfg,cohort,_=protocol(manifest,calibration)
    source=ROOT/'outputs/acl/semantic_witness_r01'; audit=read_json(source/'audit.json')
    if read_json(source/'status.json')['status']!='complete_development': raise ValueError('Prior screening not sealed')
    if audit['manifest_sha256']!=digest_file(manifest) or audit['calibration_sha256']!=digest_file(calibration):
        raise ValueError('Prior protocol differs')
    if audit['selector_sha256']!=digest_file(ROOT/'src/mindset/methods/semantic_witness.py'):
        raise ValueError('Source-span/neighbor contract changed')
    prior=unique_rows(source/'retrieval.jsonl')
    expected={f"{r['dataset']}/{a}/{r['id']}" for r in cohort['records'] for a in ARMS}
    out=ROOT/'outputs/acl/raw_source_seeds_r01'; out.mkdir(parents=True,exist_ok=True)
    identity={'manifest_sha256':digest_file(manifest),'calibration_sha256':digest_file(calibration),
        'worker_sha256':digest_file(__file__),'source_module_sha256':audit['selector_sha256'],
        'reference_retrieval_sha256':digest_file(source/'retrieval.jsonl'),
        'checkpoint_lock_sha256':digest_file(ROOT/'configs/checkpoints.lock.json'),'methods':ARMS,
        'source_adapter':'original_frozen_text_only','limits':{'final_pool':48,'passages':8,'seed_episodes':4,'neighbors_per_seed':4},
        'ce_budget_scope':'48 final candidates; CE-selected seeds require an additional initial 48-pair pass; hybrid seeds do not',
        'no_NLI_or_new_state_construction':True,'no_reference_or_gold_admission':True,
        'claim_scope':'state-substrate and seed controls, NOT pure energy ablation or novelty claim',
        'quality_cache':'exact embedding reuse; shared scores NOT timing/efficiency',
        'full_runs_authorized':False}
    import fcntl
    with (out/'worker.lock').open('a+b') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(out/'audit.json',identity,immutable=True); record_environment(out/'continuations'/os.environ['SLURM_JOB_ID'])
        encoder,ce=local_stack(cfg,cache=False); encoder=CachedEncoder(encoder,enabled=True)
        from transformers import AutoTokenizer
        checkpoint=read_json(ROOT/'configs/checkpoints.lock.json')[cfg['evidence_tokenizer']]
        tokenizer=AutoTokenizer.from_pretrained(checkpoint['checkpoint'],revision=checkpoint['revision'],
            cache_dir=str(Path.home()/'MINDSET_v2/.cache/gpu_prototype/huggingface/hub'),local_files_only=True)
        done=unique_rows(out/'retrieval.jsonl')
        if set(done)-expected: raise ValueError('Unexpected resumed source-seed cells')
        for dataset,cluster,cases in groups_for(cohort,cfg):
            if all(f'{dataset}/{a}/{c.id}' in done for c in cases for a in ARMS): continue
            events=cases[0].events; index=LeanEpisodeIndex(events,encoder)
            documents=dict(zip(index.ids,index.documents,strict=True)); spans=source_spans(events)
            vectors=encoder.encode([s.text for s in spans],batch_size=64,normalize_embeddings=True,
                convert_to_numpy=True,show_progress_bar=False) if spans else np.empty((0,index.vectors.shape[1]))
            print(f'BUILD {dataset}/{cluster} episodes={len(events)} spans={len(spans)}',flush=True)
            for case in cases:
                key=f'{dataset}/{{}}/{case.id}'; joint=prior[key.format('joint')]; expanded=prior[key.format('no_verifier')]
                flat_ref=prior[key.format('flat48')]; flat=[index.ids[i] for i in index.candidates(case.question,encoder,48)]
                qvec=encoder.encode([case.question],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)[0]
                budget=joint['evidence_tokens']; local_cfg={**cfg,'retrieval':{**cfg['retrieval'],'candidate_k':48,'evidence_tokens':budget}}
                packet_index=EvidenceCache(events,encoder,tokenizer,local_cfg); scores={}
                sources={'expanded_joint_ref':[s['event_id'] for s in joint['selected']],
                         'flat_ce_seed':[s['event_id'] for s in flat_ref['selected']], 'hybrid_seed':flat}
                calculated={}
                for arm,seeds in sources.items():
                    pairs=semantic_pairs(index,spans,vectors,qvec,seeds,events,question_date=case.question_date)
                    links=[{'left':spans[a].eid,'right':spans[b].eid} for a,b in pairs]
                    pool=extend_pool(flat,links)
                    if tuple(pool) not in scores:
                        values=np.asarray(ce.predict([(case.question,documents[e]) for e in pool],
                            batch_size=cfg['mindset']['rerank_batch_size'],show_progress_bar=False))
                        if len(values)!=len(pool) or not np.isfinite(values).all(): raise ValueError('Bad CE scores')
                        scores[tuple(pool)]=dict(zip(pool,map(float,values),strict=True))
                    score=scores[tuple(pool)]; ranking=sorted(pool,key=lambda e:(-score[e],e))
                    hits=[MemoryHit(e,documents[e],score[e],(e,),'raw_source_seed',{}) for e in ranking]
                    selected,packet=packet_index.packet(hits,case.question)
                    calculated[arm]=(selected,packet,pool,seeds[:4],pairs,score)
                # Eliminating NLI must exactly replay the successful unverified arm.
                if calculated['expanded_joint_ref'][0]!=expanded['selected'] or calculated['expanded_joint_ref'][1]!=expanded['evidence']:
                    raise ValueError('NLI-free replay differs from prior unverified packets')
                for arm in ARMS:
                    rid=f'{dataset}/{arm}/{case.id}'
                    if rid in done: continue
                    if arm=='joint_ref':
                        selected,packet=joint['selected'],joint['evidence']; pool=[]; seeds=[]; pairs=[]; score={}
                        ce_pairs=48; construction='reference frozen Joint state; not reconstructed here'
                    else:
                        selected,packet,pool,seeds,pairs,score=calculated[arm]
                        ce_pairs=len(pool)+(0 if arm=='hybrid_seed' else 48)
                        construction='Joint-derived seed reference' if arm=='expanded_joint_ref' else 'no Joint state'
                    count=len(tokenizer.encode(packet,add_special_tokens=False))
                    if count>budget or len(selected)>8 or len(pool)>48: raise ValueError('Final packet budget violated')
                    row={'id':rid,'dataset':dataset,'method':arm,'question_id':case.id,'cluster':cluster,'category':case.category,
                        'selected':selected,'evidence':packet,'evidence_tokens':count,'matched_token_ceiling':budget,
                        'metrics':rank_metrics(selected,case,8),'candidate_ids':pool,'CE_scores':score,
                        'seed_ids':seeds,'source_pair_ids':[[spans[a].eid,spans[b].eid] for a,b in pairs],
                        'logical_ce_pairs':ce_pairs,'state_construction_dependency':construction,
                        'NLI_calls':0,'qa_calls':0,'scope':'exposed original100 source-seed controls'}
                    append_row(out/'retrieval.jsonl',row); done[rid]=row
                summarize(out,expected); print(f'QUERY {case.id} cells={len(done)}/400',flush=True)
            encoder.cache.clear()
        if {r['id'] for r in done.values()}!=expected: raise ValueError('Incomplete original100 controls')
        summarize(out,expected)


if __name__=='__main__': run()
