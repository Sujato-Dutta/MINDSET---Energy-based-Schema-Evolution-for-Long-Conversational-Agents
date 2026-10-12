"""DGX development-only matched-budget controller tests, never parameter tuning."""
from pathlib import Path
from dataclasses import replace
from collections import defaultdict
import argparse
import os
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mindset.methods.joint import JointMindsetMemory,typed_features,utc_time
from mindset.methods.compact import CompactConfig
from mindset.memory.energy import unit_vector
from mindset.methods.base import MemoryHit
from mindset.retrieval.hybrid import ranked_indices,reciprocal_rank_fusion
from mindset.acl.joint_iteration import protocol,groups_for
from mindset.acl.pipeline import local_stack,record_environment,source_fingerprint,write_csv,rank_metrics
from mindset.acl.runtime import CachedEncoder
from mindset.acl.evidence import EvidenceCache
from mindset.acl.clients import load_tokenizer,messages,ANSWER_PROMPT_HASH
from mindset.acl.io import read_json,write_json,digest_file,append_row,unique_rows
from mindset.utils.hashing import stable_hash

ARMS=('full','preselected','rule','nearest_merge','no_context','no_complexity','no_hysteresis','no_provenance','flat48')

class RuleMemory(JointMindsetMemory):
    """Fixed feasible-action rules; no fitted energy selection or novelty claim."""
    def __init__(self,*args,**kwargs): super().__init__(*args,joint_targets=False,**kwargs)
    def _plans(self,schema,episode):
        plans=super()._plans(schema,episode); signals=plans[0]['signals']
        preferred='split' if signals.contextual_potential else 'supersede' if signals.temporal_potential else 'reinforce'
        for p in plans: p['cost']=float(p['action']!=preferred)
        return plans
    def _ingest_episode(self,episode):
        super()._ingest_episode(episode)
        cert=self._decision_certificates.get(episode.episode_id)
        if not cert: return
        record=self.operation_log.all()[-1]; margin=self.config.switch_margin
        for p in cert['feasible_candidates']:
            p['policy_score']=p['adjusted_energy']
            p['adjusted_energy']=p['terms']['total']+margin*(p['action']=='supersede')
        chosen=next(p for p in cert['feasible_candidates'] if p['schema_id']==record.target_schema_id and p['action']==record.operation)
        cert.update(policy='fixed_rule_control',selected_adjusted_energy=chosen['adjusted_energy'],
                    optimality_gap=max(0.,chosen['adjusted_energy']-min(p['adjusted_energy'] for p in cert['feasible_candidates'])))
        self.last_joint_candidates=tuple(cert['feasible_candidates'])

class NearestMergeMemory(JointMindsetMemory):
    """Owner/predicate-blocked nearest-centroid merge, without memory transitions."""
    def _ingest_episode(self,episode):
        episode=replace(episode,features=typed_features(episode.speaker,episode.text)); self.episodes[episode.episode_id]=episode
        topic=f'{episode.speaker.casefold()}::{episode.features.primary_slot}'
        pool=[self.schemas[i] for i in self._by_topic[topic] if self.schemas[i].status=='active']
        if not pool:
            s=self._create_schema(episode,action='new'); self._record_transition(episode,'new',None,(s.schema_id,),{},None); return
        target=max(pool,key=lambda s:(float(s.centroid@episode.vector),s.schema_id))
        target.support_episode_ids.append(episode.episode_id); target._add_features(episode.features)
        self._sums[target.schema_id]+=episode.vector; total=self._sums[target.schema_id]
        target.centroid=unit_vector(total)
        target.mean_distortion=float(np.clip((1-np.linalg.norm(total)/len(target.support_episode_ids))/2,0,1))
        self._latest[target.schema_id]=max(self._latest[target.schema_id],utc_time(episode.timestamp))
        self._record_transition(episode,'reinforce',target.schema_id,(target.schema_id,),{},None)

def construct(arm,events,cfg,parameters,encoder,reranker):
    options={**{k:v for k,v in cfg['mindset'].items() if k!='profile'},**parameters}
    if arm.startswith('no_'): options['ablation']=arm
    cls=RuleMemory if arm=='rule' else NearestMergeMemory if arm=='nearest_merge' else JointMindsetMemory
    kwargs={'joint_targets':False} if arm=='preselected' else {}
    m=cls(CompactConfig(**options),encoder=encoder,local_reranker=reranker,**kwargs); m.ingest(events)
    return m

def flat_pool(m,q):
    vector=unit_vector(m._encode([q])[0]); dense=m._episode_vectors@vector; lexical=m._bm25.scores(q)
    fused=reciprocal_rank_fusion([ranked_indices(dense),ranked_indices(lexical)])
    return sorted(fused,key=lambda i:(-fused[i],i))[:min(48,len(dense))]

def summary(out,methods):
    rows=list(unique_rows(out/'evidence.jsonl').values()); report=[]; differences=[]
    for dataset in ('locomo','longmemeval'):
        full={r['question_id']:r for r in rows if r['dataset']==dataset and r['arm']=='full'}
        for arm in methods:
            selected=[r for r in rows if r['dataset']==dataset and r['arm']==arm]
            if not selected: continue
            result={'dataset':dataset,'arm':arm,'questions':len(selected),'expected':50}
            for metric in ('recall@8','precision@8','mrr@8','ndcg@8'):
                result[metric]=float(np.mean([r['retrieval'][metric] for r in selected]))
            report.append(result)
            pairs=[(full[r['question_id']],r) for r in selected if r['question_id'] in full]
            if arm!='full' and pairs:
                differences.append({'dataset':dataset,'control':arm,'paired_questions':len(pairs),
                    'identical_complete_answer_prompts':sum(a['prompt_sha256']==b['prompt_sha256'] for a,b in pairs),
                    'identical_ordered_sources':sum([s['event_id'] for s in a['selected']]==[s['event_id'] for s in b['selected']] for a,b in pairs),
                    **{metric+'_delta':float(np.mean([a['retrieval'][metric]-b['retrieval'][metric] for a,b in pairs]))
                       for metric in ('recall@8','precision@8','mrr@8','ndcg@8')}})
    if report: write_csv(out/'retrieval.csv',report)
    if differences: write_csv(out/'causal_opportunity.csv',differences)
    write_json(out/'status.json',{'completed_cells':len(rows),'expected_cells':100*len(methods),
        'status':'complete_development' if len(rows)==100*len(methods) else 'partial_development',
        'qa_calls':0,'judge_calls':0,'full_runs_authorized':False,
        'scope':'matched-budget repeated-look development; identical full prompts preclude attributing QA differences to that memory change'})

def run(manifest,calibration,output):
    cfg,cohort,artifact=protocol(manifest,calibration); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'worker.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        audit={'scope':'development100_matched_budget_causal_probe','source_fingerprint':source_fingerprint(),
            'worker_sha256':digest_file(Path(__file__)),'calibration_sha256':digest_file(calibration),
            'manifest_sha256':digest_file(manifest),'checkpoint_lock':read_json(ROOT/'configs/checkpoints.lock.json'),
            'parameters':artifact['selected_parameters'],'arms':list(ARMS),'answer_prompt_sha256':ANSWER_PROMPT_HASH,
            'budgets':{'logical_candidates_per_arm':48,'raw_episodes':8,'evidence_tokens':8000},
            'score_reuse':'identical query/raw-episode CE scores shared only in quality comparisons, never latency',
            'labels':'retrieval annotations used only after evidence selection; no state-error or judge labels fabricated',
            'lightmem':'not modified or included in this within-MINDSET causal test','full_runs_authorized':False}
        write_json(out/'audit.json',audit,immutable=True); record_environment(out)
        done=unique_rows(out/'evidence.jsonl')
        expected={f"{r['dataset']}/{a}/{r['id']}" for r in cohort['records'] for a in ARMS}
        if set(done)-expected: raise ValueError('Unexpected resumed IDs')
        encoder,reranker=local_stack(cfg,cache=False); encoder=CachedEncoder(encoder,enabled=True)
        tokenizer=load_tokenizer(read_json(ROOT/'configs/checkpoints.lock.json')[cfg['evidence_tokenizer']])
        for dataset,cluster,queries in groups_for(cohort,cfg):
            pending=[q for q in queries if any(f'{dataset}/{a}/{q.id}' not in done for a in ARMS)]
            if not pending: continue
            memories={}
            for arm in ARMS:
                if arm=='flat48': memories[arm]=memories['full']; continue
                print(f'Build {dataset}/{cluster}/{arm}',flush=True)
                memories[arm]=construct(arm,queries[0].events,cfg,artifact['selected_parameters'],encoder,reranker)
                m=memories[arm]; inv=m.validate_invariants(); records=m.operation_log.all()
                write_json(out/'structure'/f'{dataset}-{cluster}-{arm}.json',{'invariants':inv,'schema_count':len(m.schemas),
                    'operation_counts':{a:sum(r.operation==a for r in records) for a in ('reinforce','supersede','split','new')},
                    'historical_schemas':sum(s.status=='historical' for s in m.schemas.values()),
                    'exception_stores':sum(r.stored_as_exception for r in records),
                    'decision_policy':'fixed_rule' if arm=='rule' else 'nearest_merge' if arm=='nearest_merge' else 'energy',
                    'state_gold_accuracy':None,'state_gold_status':'no independent natural-state labels supplied'})
            index=EvidenceCache(queries[0].events,encoder,tokenizer,cfg)
            for case in pending:
                arms=[a for a in ARMS if f'{dataset}/{a}/{case.id}' not in done]
                pools={a:flat_pool(memories[a],case.question) if a=='flat48' else memories[a].candidate_pool(case.question)[0] for a in arms}
                union=sorted(set(i for pool in pools.values() for i in pool))
                base=memories['full']; scores=np.asarray(reranker.predict([(case.question,base._documents[i]) for i in union],
                      batch_size=8,show_progress_bar=False),dtype=float)
                if len(scores)!=len(union) or not np.isfinite(scores).all(): raise ValueError('Invalid CE scores')
                by_index=dict(zip(union,scores,strict=True))
                for arm,pool in pools.items():
                    order=sorted(pool,key=lambda i:(-by_index[i],base._episode_order[i]))
                    hits=[MemoryHit(base._episode_order[i],base._documents[i],by_index[i],(base._episode_order[i],),'causal_raw_source',{}) for i in order]
                    selected,packet=index.packet(hits,case.question)
                    prompt=messages(case.question,packet,case.question_date)
                    row={'id':f'{dataset}/{arm}/{case.id}','dataset':dataset,'question_id':case.id,'cluster':cluster,
                        'category':case.category,'arm':arm,'question':case.question,'question_date':case.question_date,
                        'answer':case.answer,'selected':selected,'evidence':packet,'messages':prompt,
                        'prompt_sha256':stable_hash(prompt),'candidate_count':len(pool),'shared_union_pairs':len(union),
                        'retrieval':rank_metrics(selected,case,8),'status':'development_not_paper'}
                    append_row(out/'evidence.jsonl',row); done[row['id']]=row
                summary(out,ARMS); print(f'{len(done)}/900 causal cells complete',flush=True)
            encoder.cache.clear(); del memories,index
        summary(out,ARMS)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--calibration',required=True); p.add_argument('--output',required=True)
    p.add_argument('--manifest',default=str(ROOT/'data/manifests/mindset_v2_development100.json')); args=p.parse_args()
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): p.error('All experiment computation requires DGX Slurm')
    run(args.manifest,args.calibration,args.output)

if __name__=='__main__': main()
