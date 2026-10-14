"""Fixed development100, source-only ten-idea screening. DGX Slurm only.

All arms use the frozen encoder, CE, 48-candidate/8-source budgets and reader.
Cached CE work is permitted for quick QUALITY screening, never latency claims.
No full study, benchmark retuning or hosted judging is performed here.
"""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import os
import numpy as np
from mindset.acl.joint_iteration import protocol, groups_for
from mindset.acl.pipeline import ROOT, local_stack, rank_metrics, record_environment, write_csv
from mindset.acl.io import read_json, write_json, append_row, unique_rows, digest_file
from mindset.acl.clients import load_tokenizer
from mindset.acl.runtime import CachedEncoder
from mindset.acl.evidence import EvidenceCache
from mindset.methods.compact import CompactConfig
from mindset.methods.joint import JointMindsetMemory
from mindset.methods.mindset import _render_event
from mindset.methods.rerank_cache import CachedReranker
from mindset.utils.hashing import stable_hash

ARMS=('joint','idea01_rwse','idea02_cdm','idea03_mdl','idea04_fqrt',
      'idea05_cmep','idea06_pcmt','idea07_signed','idea08_compiler',
      'idea09_spectral','idea10_repair')
METRICS=('recall@8','precision@8','mrr@8','ndcg@8')

def controller(arm, parameters, **kwargs):
    from mindset.methods.ideas_window import ReversibleWindowMemory
    from mindset.methods.ideas_cdm_mdl import ConformalDeferredMemory, MDLDescriptionMemory
    from mindset.methods.ideas_regret_program_proof import FutureQueryRegretMemory, CompositionalEditMemory, ProofCarryingMemory
    from mindset.methods.ideas_signed_temporal_spectral_repair import SignedRetrievalMemory, TemporalCompilerMemory, SpectralBifurcationMemory, CounterfactualRepairMemory
    types=dict(zip(ARMS, (JointMindsetMemory,ReversibleWindowMemory,ConformalDeferredMemory,
        MDLDescriptionMemory,FutureQueryRegretMemory,CompositionalEditMemory,ProofCarryingMemory,
        SignedRetrievalMemory,TemporalCompilerMemory,SpectralBifurcationMemory,CounterfactualRepairMemory)))
    return types[arm](parameters, **kwargs)

def summarize(out, arms, expected=None):
    rows=list(unique_rows(out/'retrieval.jsonl').values()); report=[]
    for dataset in ('locomo','longmemeval'):
        for arm in arms:
            subset=[r for r in rows if r['dataset']==dataset and r['method']==arm]
            if not subset: continue
            wanted=sum(key.startswith(f'{dataset}/{arm}/') for key in expected) if expected is not None else 50
            item={'dataset':dataset,'method':arm,'questions':len(subset),'expected':wanted,
                **{key:float(np.mean([r['metrics'][key] for r in subset])) for key in METRICS},
                'candidate_pool_recall@48':float(np.mean([r['candidate_pool_recall@48'] for r in subset])),
                'mean_evidence_tokens':float(np.mean([r['evidence_tokens'] for r in subset])),
                'diagnostic_retrieval_p50_ms':float(np.quantile([r['diagnostic_query_seconds']*1000 for r in subset],.50)),
                'diagnostic_retrieval_p95_ms':float(np.quantile([r['diagnostic_query_seconds']*1000 for r in subset],.95)),
                'latency_scope':'cache-enabled single-pass screening; NOT paper efficiency',
                'native_answer_metric':'not generated yet','judge_status':'deferred'}
            item['retrieval_score']=.35*item['recall@8']+.25*item['mrr@8']+.25*item['ndcg@8']+.15*item['precision@8']
            report.append(item)
    if report: write_csv(out/'retrieval.csv',report)
    total=len(expected) if expected is not None else 100*len(arms)
    write_json(out/'status.json',{'status':'complete_development' if len(rows)==total else 'partial_development',
        'completed_cells':len(rows),'expected_cells':total,'methods':list(arms),
        'unique_questions':len({(r['dataset'],r['question_id']) for r in rows}),
        'qa_calls':0,'judge_calls':0,'full_runs_authorized':False})
    return report

def run(arms, output, shard=0, shards=1):
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('DGX Slurm only')
    if not arms or any(a not in ARMS for a in arms) or len(set(arms))!=len(arms): raise ValueError('Invalid pilot arms')
    if not 0<=shard<shards: raise ValueError('Invalid history shard')
    manifest=ROOT/'data/manifests/mindset_v2_development100.json'
    calibration=ROOT/'outputs/acl/mindset_joint_v2_27058/calibration/calibration.json'
    cfg, cohort, calibrated=protocol(manifest,calibration)
    if cohort['count']!=100: raise ValueError('Original 100-question pilot only')
    parameters=CompactConfig(**{**{k:v for k,v in cfg['mindset'].items() if k!='profile'},**calibrated['selected_parameters']})
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'worker.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source_paths=[ROOT/'src/mindset/methods'/name for name in ('idea_transition_tools.py','ideas_window.py',
            'ideas_cdm_mdl.py','ideas_regret_program_proof.py','ideas_signed_temporal_spectral_repair.py')]
        audit={'kind':'ten_ideas_bounded_pilot','manifest_sha256':digest_file(manifest),
            'calibration_sha256':digest_file(calibration),'parameters':asdict(parameters),
            'checkpoints':read_json(ROOT/'configs/checkpoints.lock.json'),'configuration':cfg,
            'controller_source_sha256':{p.name:digest_file(p) for p in source_paths},
            'worker_sha256':digest_file(__file__),'methods':list(arms),'shard':shard,'shards':shards,'top_k':8,'ce_candidates':48,
            'source_adapter':'original_frozen_text_only','source_reference':'unchanged development100; repeated-look, not pristine holdout',
            'natural_answers_or_gold_used_for_controller':False,'benchmark_tuning':False,
            'quality_cache_policy':'share exact embedding and query-pair CE scores, not independent timing',
            'full_runs_authorized':False,'judge_calls':0}
        write_json(out/'audit.json',audit,immutable=True); record_environment(out/'continuations'/os.environ['SLURM_JOB_ID'])
        encoder, ce=local_stack(cfg,cache=False); encoder=CachedEncoder(encoder,enabled=True)
        ce=CachedReranker(ce,stable_hash(audit['checkpoints']['reranker']),out/'pair_scores.sqlite')
        tokenizer=load_tokenizer(audit['checkpoints'][cfg['evidence_tokenizer']])
        cdm=None
        if 'idea02_cdm' in arms:
            from mindset.methods.ideas_cdm_mdl import calibrate_cdm
            path=out/'cdm_calibration.json'
            if path.exists(): cdm=read_json(path)
            else:
                cdm=calibrate_cdm(encoder,parameters)
                write_json(path,cdm,immutable=True)
        done=unique_rows(out/'retrieval.jsonl')
        groups=list(groups_for(cohort,cfg))
        groups=[g for i,g in enumerate(groups) if i%shards==shard]
        expected={f'{d}/{a}/{c.id}' for d,cluster,cases in groups for c in cases for a in arms}
        if set(done)-expected: raise ValueError('Extra pilot cells')
        for dataset, cluster, cases in groups:
            events=cases[0].events
            vectors=encoder.encode([_render_event(e) for e in events],batch_size=64,normalize_embeddings=True,
                                   convert_to_numpy=True,show_progress_bar=False)
            lookup={e.event_id:v for e,v in zip(events,vectors,strict=True)}
            for arm in arms:
                pending=[c for c in cases if f'{dataset}/{arm}/{c.id}' not in done]
                if not pending: continue
                print(f'BUILD {dataset}/{cluster}/{arm} episodes={len(events)}',flush=True)
                kwargs={'encoder':encoder,'local_reranker':ce,'precomputed_embeddings':lookup}
                if arm=='idea02_cdm': kwargs['calibration']=cdm
                memory=controller(arm,parameters,**kwargs)
                started=perf_counter(); memory.ingest(events); write_seconds=perf_counter()-started
                if memory._episode_order!=[e.event_id for e in events]: raise ValueError('Immutable source order changed')
                invariants=memory.validate_invariants()
                write_json(out/'structure'/f'{dataset}-{cluster}-{arm}.json',{'episodes':len(events),
                    'controller_write_seconds':write_seconds,'ingest_seconds_per_episode':write_seconds/len(events),
                    'invariants':invariants,'telemetry':memory.telemetry()})
                index=EvidenceCache(events,encoder,tokenizer,cfg)
                for case in pending:
                    prior_pairs=memory._retrieval_profile.get('query_episode_pairs',0)
                    start=perf_counter(); hits=memory.retrieve(case.question,8)
                    selected,packet=index.packet(hits,case.question); duration=perf_counter()-start
                    candidate_ids={memory._episode_order[i] for i in memory.last_candidate_pool}
                    if len(candidate_ids)>48 or any(s['event_id'] not in candidate_ids for s in selected):
                        raise ValueError('Final evidence bypasses bounded scored candidate pool')
                    unit={e.event_id:e.session_id if case.retrieval_unit=='session' else e.event_id for e in events}
                    pool={unit[memory._episode_order[i]] for i in memory.last_candidate_pool}
                    gold=set(case.gold)
                    row={'id':f'{dataset}/{arm}/{case.id}','dataset':dataset,'method':arm,'question_id':case.id,
                        'cluster':cluster,'category':case.category,'metrics':rank_metrics(selected,case,8),
                        'selected':selected,'evidence':packet,'evidence_tokens':len(tokenizer.encode(packet,add_special_tokens=False)),
                        'candidate_count':len(memory.last_candidate_pool),'candidate_pool_recall@48':len(pool&gold)/len(gold) if gold else 0.,
                        'diagnostic_query_seconds':duration,'idea_certificate':getattr(memory,'last_idea_certificate',{}),
                        'logical_ce_pair_count':memory._retrieval_profile.get('query_episode_pairs',0)-prior_pairs,
                        'deterministic_answer_diagnostic':getattr(memory,'deterministic_answer',None),
                        'status':'repeated-look development; not paper', 'qa_calls':0,'judge_calls':0}
                    append_row(out/'retrieval.jsonl',row); done[row['id']]=row
                    print(f'CELL {len(done)}/{len(expected)} {row["id"]}',flush=True)
                write_json(out/'query_structure'/f'{dataset}-{cluster}-{arm}.json',{
                    'telemetry_after_queries':memory.telemetry()})
                summarize(out,arms,expected); del memory,index
            encoder.cache.clear()
        if set(done)!=expected: raise ValueError('Incomplete pilot')
        return summarize(out,arms,expected)
