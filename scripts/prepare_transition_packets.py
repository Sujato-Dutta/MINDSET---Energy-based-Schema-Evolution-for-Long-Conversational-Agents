"""Zero-answer-call source-grounded transition packet pilot on dev100 subset."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from mindset.acl.joint_iteration import protocol,groups_for
from mindset.acl.pipeline import ROOT,local_stack,rank_metrics,record_environment,write_csv
from mindset.acl.io import read_json,read_rows,write_json,append_row,unique_rows,digest_file
from mindset.acl.evidence import EvidenceCache
from mindset.methods.compact import CompactConfig
from mindset.methods.joint import JointMindsetMemory,utc_time
from mindset.methods.mindset import _render_event
from mindset.methods.base import MemoryHit
from mindset.methods.query_state_agent import state_fingerprint
from mindset.methods.transition_packets import ARMS,TransitionCatalog,inject,select_packet
from mindset.acl.iteration import seeded_key

OUT='outputs/acl/transition_packets_r01'
PROJECTION='outputs/acl/participant_frontier_prepare_27392/projections'
METRICS=('recall@8','precision@8','mrr@8','ndcg@8')


def select_subset(groups):
    selected=[]
    locomo=[g for g in groups if g[0]=='locomo']
    if len(locomo)!=10: raise ValueError('Expected original ten conversations')
    for dataset,cluster,cases in locomo:
        ranked=sorted(cases,key=lambda c:(c.category!='temporal',seeded_key(17,c.id)))
        selected.extend((dataset,c.id) for c in ranked[:2])
    priority={'knowledge-update':0,'temporal-reasoning':1,'multi-session':2}
    lme=[c for d,cluster,cases in groups if d=='longmemeval' for c in cases]
    lme.sort(key=lambda c:(priority.get(c.category,3),seeded_key(17,c.id)))
    selected.extend(('longmemeval',c.id) for c in lme[:20])
    if len(set(selected))!=40: raise ValueError('Only 20+20 authorized subset')
    return set(selected)


class PacketBuilder:
    def __init__(self,memory,index,query,scores,budget,mode='ordinary',witness_values=None):
        self.memory=memory; self.index=index; self.query=query; self.scores=scores
        self.budget=budget; self.mode=mode
        self.witness_values=witness_values or {}
        if index.cfg['retrieval']['evidence_tokens']!=budget: raise ValueError('Unmatched packet ceiling')
    def token_count(self,text): return len(self.index.tokenizer.encode(text,add_special_tokens=False))
    def witness_ids(self,selected):
        clean=lambda value:' '.join(value.casefold().split())
        return {s['event_id'] for s in selected if any(clean(v) in clean(s['text'])
            for v in self.witness_values.get(s['event_id'],()) if v.strip())}
    def __call__(self,ids):
        ids=list(dict.fromkeys(e for e in ids if e in self.scores))
        hits=[MemoryHit(e,self.memory._documents[self.memory._index_by_id[e]],self.scores[e],(e,),
            'transition_packet_raw_source',{}) for e in ids]
        selected,packet=self.index.packet(hits,self.query)
        # Both compact arms use identical order policy. Never silently exceed
        # the exact ordinary packet's token ceiling after renumbering evidence.
        if self.mode=='change':
            ordered=sorted(selected,key=lambda s:(utc_time(s['timestamp']),self.memory._index_by_id[s['event_id']]))
            rendered='\n\n'.join(f'[{i}] {s["text"]}' for i,s in enumerate(ordered,1))
            if self.token_count(rendered)<=self.budget: selected,packet=ordered,rendered
        if self.token_count(packet)>self.budget: raise ValueError('Matched token ceiling exceeded')
        return selected,packet


def summarize(out):
    rows=list(unique_rows(out/'retrieval.jsonl').values()); report=[]
    for dataset in ('locomo','longmemeval'):
        for arm in ARMS:
            subset=[r for r in rows if r['dataset']==dataset and r['method']==arm]
            if not subset: continue
            baseline={r['question_id']:r for r in rows if r['dataset']==dataset and r['method']=='joint'}
            report.append({'dataset':dataset,'method':arm,'questions':len(subset),'expected':20,
                **{m:float(np.mean([r['metrics'][m] for r in subset])) for m in METRICS},
                'bound_queries':sum(r['certificate']['bound'] for r in subset),
                'packet_changes':sum(r['evidence']!=baseline[r['question_id']]['evidence'] for r in subset),
                'mean_evidence_tokens':float(np.mean([r['evidence_tokens'] for r in subset])),
                'qa_calls':0,'judge_calls':0,'scope':'repeated-look development; not paper'})
    if report: write_csv(out/'retrieval.csv',report)
    changed=sum(r['method']!='joint' and r['evidence']!=next(b['evidence'] for b in rows if
        b['dataset']==r['dataset'] and b['question_id']==r['question_id'] and b['method']=='joint') for r in rows)
    write_json(out/'status.json',{'status':'complete_development' if len(rows)==160 else 'partial_development',
        'completed_cells':len(rows),'expected_cells':160,'changed_packets':changed,
        'answer_gate':'bounded_answers_allowed_if_any_packet_changed' if changed else 'no_new_answer_calls_needed_yet',
        'qa_calls':0,'judge_calls':0,'full_runs_authorized':False})
    return report


def run(output):
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('DGX Slurm only')
    manifest=ROOT/'data/manifests/mindset_v2_development100.json'
    calibration=ROOT/'outputs/acl/mindset_joint_v2_27058/calibration/calibration.json'
    cfg,cohort,calibrated=protocol(manifest,calibration); groups=list(groups_for(cohort,cfg))
    selected=select_subset(groups); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    records=[r for r in cohort['records'] if (r['dataset'],r['id']) in selected]
    subset={'records':records,'count':40,'source_manifest_sha256':digest_file(manifest),
        'selection':'seed17; two per LoCoMo conversation, temporal first; LME native KU/temporal category first',
        'selection_uses_scores_answers_or_gold':False,'pristine_holdout':False}
    write_json(out/'subset.json',subset,immutable=True)
    parameters=CompactConfig(**{**{k:v for k,v in cfg['mindset'].items() if k!='profile'},**calibrated['selected_parameters']})
    projection=ROOT/PROJECTION
    if read_json(projection/'status.json')['status']!='complete_projection': raise ValueError('Incomplete source projection')
    paudit=read_json(projection/'audit.json')
    for name,sha in paudit['module_sha256'].items():
        if digest_file(ROOT/'src/mindset/memory'/name)!=sha: raise ValueError('Projection definition changed')
    source_rows=defaultdict(dict)
    for r in read_rows(projection/'reports.jsonl'):
        if r['kind']=='episode':
            key=(r['dataset'],r['cluster'])
            if r['source_id'] in source_rows[key]: raise ValueError('Duplicate projected source')
            source_rows[key][r['source_id']]=r
    frozen=('methods/joint.py','methods/compact.py','methods/mindset.py','memory/energy.py')
    audit={'kind':'bounded_transition_packets','manifest_sha256':digest_file(manifest),'subset_sha256':digest_file(out/'subset.json'),
        'calibration_sha256':digest_file(calibration),'parameters':asdict(parameters),
        'source_projection_sha256':digest_file(projection/'reports.jsonl'),'source_projection_audit':paudit,
        'frozen_controller_sha256':{p:digest_file(ROOT/'src/mindset'/p) for p in frozen},
        'selector_sha256':digest_file(ROOT/'src/mindset/methods/transition_packets.py'),'worker_sha256':digest_file(__file__),
        'checkpoint_lock_sha256':digest_file(ROOT/'configs/checkpoints.lock.json'),'methods':ARMS,
        'budget':'48 CE candidates, up to 8 episodes, per-query ceiling = ordinary actual evidence tokens',
        'selection':'fixed lexicographic cost; no learned/tuned coefficients; no gold admission',
        'source_adapter':'original_frozen_text_only','claim_scope':'report provenance, not world truth or physical thermodynamics',
        'quality_score_reuse':'same expanded CE scores across three treatment arms; not efficiency benchmark',
        'qa_calls':0,'judge_calls':0,'full_runs_authorized':False}
    import fcntl
    with (out/'worker.lock').open('a+b') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(out/'audit.json',audit,immutable=True); record_environment(out/'continuations'/os.environ['SLURM_JOB_ID'])
        encoder,ce=local_stack(cfg,cache=False)
        from transformers import AutoTokenizer
        checkpoint=read_json(ROOT/'configs/checkpoints.lock.json')[cfg['evidence_tokenizer']]
        tokenizer=AutoTokenizer.from_pretrained(checkpoint['checkpoint'],revision=checkpoint['revision'],
            cache_dir=str(Path.home()/'MINDSET_v2/.cache/gpu_prototype/huggingface/hub'),local_files_only=True)
        prior=unique_rows(ROOT/'outputs/acl/ten_ideas_r02/merged/retrieval.jsonl'); done=unique_rows(out/'retrieval.jsonl')
        expected={f'{d}/{arm}/{qid}' for d,qid in selected for arm in ARMS}
        if set(done)-expected: raise ValueError('Unexpected resumed cells')
        for dataset,cluster,cases in groups:
            cases=[c for c in cases if (dataset,c.id) in selected]
            if not cases or all(f'{dataset}/{arm}/{c.id}' in done for c in cases for arm in ARMS): continue
            events=cases[0].events
            print(f'BUILD {dataset}/{cluster} questions={len(cases)} episodes={len(events)}',flush=True)
            vectors=encoder.encode([_render_event(e) for e in events],batch_size=64,normalize_embeddings=True,
                convert_to_numpy=True,show_progress_bar=False)
            memory=JointMindsetMemory(parameters,encoder=encoder,local_reranker=ce,
                precomputed_embeddings={e.event_id:v for e,v in zip(events,vectors,strict=True)})
            memory.ingest(events); before=state_fingerprint(memory)
            if set(source_rows[(dataset,cluster)])!=set(memory.episodes): raise ValueError('Incomplete source projection')
            catalog=TransitionCatalog(memory,source_rows[(dataset,cluster)])
            write_json(out/'catalogs'/f'{dataset}-{cluster}.json',{'source_reports':len(catalog.reports),
                'contrast_types':dict(Counter(e.kind for e in catalog.edges)),
                'ambiguous_time_groups':catalog.ambiguous_time_groups,
                'edges':[asdict(e) for e in catalog.edges], 'schema_count':len(memory.schemas),
                'state_sha256':before},immutable=True)
            for case in cases:
                started=perf_counter(); hits=memory.retrieve(case.question,8)
                index=EvidenceCache(events,encoder,tokenizer,cfg)
                base_selected,base_packet=index.packet(hits,case.question)
                baseline=prior[f'{dataset}/joint/{case.id}']
                if base_selected!=baseline['selected']: raise ValueError('Frozen baseline changed')
                budget=len(tokenizer.encode(base_packet,add_special_tokens=False))
                original_pool=[memory._episode_order[i] for i in memory.last_candidate_pool]
                base_order=[h.memory_id for h in hits]
                edges,mode=catalog.bind(case.question,base_order,memory,case.question_date)
                expanded=inject(original_pool,edges)
                scores=np.asarray(ce.predict([(case.question,memory._documents[memory._index_by_id[e]]) for e in expanded],
                    batch_size=parameters.rerank_batch_size,show_progress_bar=False))
                if len(scores)!=len(expanded) or not np.isfinite(scores).all(): raise ValueError('Malformed CE scores')
                score={e:float(s) for e,s in zip(expanded,scores,strict=True)}
                ranking=sorted(expanded,key=lambda e:(-score[e],e))
                local_cfg={**cfg,'retrieval':{**cfg['retrieval'],'evidence_tokens':budget,'candidate_k':48}}
                local_index=EvidenceCache(events,encoder,tokenizer,local_cfg)
                unit={e.event_id:e.session_id if case.retrieval_unit=='session' else e.event_id for e in events}
                for arm in ARMS:
                    key=f'{dataset}/{arm}/{case.id}'
                    if key in done: continue
                    if arm=='joint' or not edges:
                        packet_selected,packet=base_selected,base_packet
                        certificate={'bound':False,'reason':'ordinary_retrieval' if arm=='joint' else 'no_grounded_contrast',
                            'energy_used':False,'proposals':[]}
                        pool=original_pool
                    else:
                        pack=PacketBuilder(memory,local_index,case.question,score,budget,
                            mode if arm.startswith('compact') else 'ordinary',catalog.source_values(edges[0]))
                        packet_selected,packet,certificate=select_packet(arm,ranking,edges,mode,case.question,catalog,pack,
                            question_date=case.question_date)
                        pool=expanded
                    selected_ids={s['event_id'] for s in packet_selected}
                    if not selected_ids.issubset(pool) or len(pool)>48 or len(packet_selected)>8:
                        raise ValueError('Evidence escaped fair scored budget')
                    count=len(tokenizer.encode(packet,add_special_tokens=False))
                    if count>budget: raise ValueError('Treatment has larger context budget')
                    gold=set(case.gold); units={unit[e] for e in pool}
                    row={'id':key,'dataset':dataset,'method':arm,'question_id':case.id,'cluster':cluster,'category':case.category,
                        'metrics':rank_metrics(packet_selected,case,8),'selected':packet_selected,'evidence':packet,
                        'evidence_tokens':count,'matched_token_ceiling':budget,'candidate_count':len(pool),
                        'candidate_pool_recall@48':len(gold&units)/len(gold) if gold else 0,
                        'certificate':certificate,'admitted_contrasts':[asdict(e) for e in edges],'intent':mode,
                        'shared_query_compute_seconds':perf_counter()-started,
                        'status':'development_not_paper','qa_calls':0,'judge_calls':0}
                    append_row(out/'retrieval.jsonl',row); done[key]=row
                summarize(out)
                print(f'QUERY {case.id} edges={len(edges)} cells={len(done)}/160',flush=True)
            if state_fingerprint(memory)!=before: raise ValueError('Frozen state mutated')
        if set(done)!=expected: raise ValueError('Incomplete pilot')
        summarize(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',default=OUT)
    run(p.parse_args().output)
