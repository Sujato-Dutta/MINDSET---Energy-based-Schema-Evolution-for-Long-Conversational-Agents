"""Original development100, slot-free semantic witness retrieval screening."""
import argparse
from collections import Counter
from dataclasses import asdict
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from mindset.acl.joint_iteration import protocol,groups_for
from mindset.acl.pipeline import ROOT,local_stack,rank_metrics,write_csv,record_environment
from mindset.acl.io import read_json,write_json,unique_rows,append_row,digest_file
from mindset.acl.runtime import CachedEncoder
from mindset.acl.evidence import EvidenceCache
from mindset.acl.local_semantics import SourceNLI
from mindset.methods.conflict_frontier import LeanEpisodeIndex
from mindset.methods.compact import CompactConfig
from mindset.methods.joint import JointMindsetMemory
from mindset.methods.mindset import _render_event
from mindset.methods.base import MemoryHit
from mindset.methods.semantic_witness import ARMS,source_spans,semantic_pairs,classified_edges,extend_pool,choose_packet
from mindset.methods.query_state_agent import state_fingerprint
from mindset.utils.hashing import stable_hash

OUTPUT='outputs/acl/semantic_witness_r01'
METRICS=('recall@8','precision@8','mrr@8','ndcg@8')


class Pack:
    def __init__(self,index,documents,query,scores,budget):
        self.index=index; self.documents=documents; self.query=query; self.scores=scores; self.budget=budget
    def token_count(self,text): return len(self.index.tokenizer.encode(text,add_special_tokens=False))
    def __call__(self,ids):
        ids=list(dict.fromkeys(e for e in ids if e in self.scores))
        hits=[MemoryHit(e,self.documents[e],self.scores[e],(e,),'semantic_witness_raw_source',{}) for e in ids]
        selected,packet=self.index.packet(hits,self.query)
        if self.token_count(packet)>self.budget: raise ValueError('Common token ceiling exceeded')
        return selected,packet


def summarize(out,expected):
    rows=list(unique_rows(out/'retrieval.jsonl').values()); reports=[]
    baseline={(r['dataset'],r['question_id']):r for r in rows if r['method']=='joint'}
    for d in ('locomo','longmemeval'):
        for arm in ARMS:
            selected=[r for r in rows if r['dataset']==d and r['method']==arm]
            if not selected: continue
            reports.append({'dataset':d,'method':arm,'questions':len(selected),'expected':50,
                **{m:float(np.mean([r['metrics'][m] for r in selected])) for m in METRICS},
                'candidate_recall@48':float(np.mean([r['candidate_recall@48'] for r in selected])),
                'packet_changes':sum(r['evidence']!=baseline[(d,r['question_id'])]['evidence'] for r in selected),
                'mean_evidence_tokens':float(np.mean([r['evidence_tokens'] for r in selected])),
                'contrast_edge_queries':sum(any(e['relation']=='contrast' for e in r['semantic_edges']) for r in selected),
                'scope':'same dev100, original text source; shared scores/caches, NOT paper efficiency'})
    if reports: write_csv(out/'retrieval.csv',reports)
    changed=sum(r['method']!='joint' and r['evidence']!=baseline[(r['dataset'],r['question_id'])]['evidence'] for r in rows)
    write_json(out/'status.json',{'status':'complete_development' if len(rows)==len(expected) else 'partial_development',
        'completed_cells':len(rows),'expected_cells':len(expected),'changed_packets':changed,
        'answer_gate':'changed_raw_packets_required','qa_calls':0,'judge_calls':0,'full_runs_authorized':False})


def run(output):
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('DGX Slurm only')
    manifest=ROOT/'data/manifests/mindset_v2_development100.json'
    calibration=ROOT/'outputs/acl/mindset_joint_v2_27058/calibration/calibration.json'
    cfg,cohort,cal=protocol(manifest,calibration); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    expected={f"{r['dataset']}/{arm}/{r['id']}" for r in cohort['records'] for arm in ARMS}
    params=CompactConfig(**{**{k:v for k,v in cfg['mindset'].items() if k!='profile'},**cal['selected_parameters']})
    audit={'scope':'semantic_witness_source_retrieval_dev100','manifest_sha256':digest_file(manifest),
        'calibration_sha256':digest_file(calibration),'parameters':asdict(params),'methods':ARMS,
        'checkpoint_lock_sha256':digest_file(ROOT/'configs/checkpoints.lock.json'),
        'worker_sha256':digest_file(__file__),'selector_sha256':digest_file(ROOT/'src/mindset/methods/semantic_witness.py'),
        'semantic_loader_sha256':digest_file(ROOT/'src/mindset/acl/local_semantics.py'),
        'frozen_joint_sha256':digest_file(ROOT/'src/mindset/methods/joint.py'),
        'limits':{'CE_candidates':48,'raw_passages':8,'anchors':4,'source_neighbors_per_anchor':4,'sentence_spans_per_episode':12},
        'source_adapter':'original_frozen_text_only','quality_cache':'exact embeddings and NLI pairs; not efficiency',
        'cost':'finite lexicographic witness coverage/redundancy/CE-rank/tokens; no fitted mixtures',
        'claim_limit':'NLI is a semantic proxy, not world truth; no global minimum or first-ever claim',
        'reader_next_stage':'published short phrase shared across LoCoMo arms; generic LME, judge deferred',
        'no_answer_gold_or_category_admission':True,'full_runs_authorized':False}
    import fcntl
    with (out/'worker.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(out/'audit.json',audit,immutable=True); record_environment(out/'continuations'/os.environ['SLURM_JOB_ID'])
        encoder,ce=local_stack(cfg,cache=False); encoder=CachedEncoder(encoder,enabled=True)
        nli=SourceNLI(ROOT,out/'semantic_pairs.sqlite'); write_json(out/'nli_audit.json',nli.identity,immutable=True)
        from transformers import AutoTokenizer
        checkpoint=read_json(ROOT/'configs/checkpoints.lock.json')[cfg['evidence_tokenizer']]
        tokenizer=AutoTokenizer.from_pretrained(checkpoint['checkpoint'],revision=checkpoint['revision'],
            cache_dir=str(Path.home()/'MINDSET_v2/.cache/gpu_prototype/huggingface/hub'),local_files_only=True)
        done=unique_rows(out/'retrieval.jsonl')
        if set(done)-expected: raise ValueError('Unexpected resumed cells')
        prior=unique_rows(ROOT/'outputs/acl/ten_ideas_r02/merged/retrieval.jsonl')
        for dataset,cluster,cases in groups_for(cohort,cfg):
            if all(f'{dataset}/{arm}/{c.id}' in done for c in cases for arm in ARMS): continue
            events=cases[0].events; print(f'BUILD {dataset}/{cluster} episodes={len(events)}',flush=True)
            index=LeanEpisodeIndex(events,encoder); documents=dict(zip(index.ids,index.documents,strict=True))
            memory=JointMindsetMemory(params,encoder=encoder,local_reranker=ce,
                precomputed_embeddings={e.event_id:v for e,v in zip(events,index.vectors,strict=True)})
            memory.ingest(events); before=state_fingerprint(memory)
            spans=source_spans(events)
            vectors=encoder.encode([s.text for s in spans],batch_size=64,normalize_embeddings=True,
                convert_to_numpy=True,show_progress_bar=False) if spans else np.empty((0,index.vectors.shape[1]))
            write_json(out/'source_index'/f'{dataset}-{cluster}.json',{'episodes':len(events),
                'source_spans':len(spans),'span_hash':stable_hash([asdict(s) for s in spans]),'frozen_state_sha256':before},immutable=True)
            original={e.event_id:e for e in events}; order={e.event_id:i for i,e in enumerate(events)}
            for case in cases:
                hits=memory.retrieve(case.question,8); base_index=EvidenceCache(events,encoder,tokenizer,cfg)
                base_selected,base_packet=base_index.packet(hits,case.question)
                if base_selected!=prior[f'{dataset}/joint/{case.id}']['selected']: raise ValueError('Joint baseline changed')
                budget=len(tokenizer.encode(base_packet,add_special_tokens=False))
                local_cfg={**cfg,'retrieval':{**cfg['retrieval'],'candidate_k':48,'evidence_tokens':budget}}
                packet_index=EvidenceCache(events,encoder,tokenizer,local_cfg)
                flat=[index.ids[i] for i in index.candidates(case.question,encoder,48)]
                qvec=encoder.encode([case.question],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)[0]
                pairs=semantic_pairs(index,spans,vectors,qvec,[h.memory_id for h in hits],events,question_date=case.question_date)
                texts=[(spans[a].text,spans[b].text) for a,b in pairs]
                directed=[p for a,b in texts for p in ((a,b),(b,a))]
                labels=nli.predict(directed); edges=classified_edges(pairs,spans,labels,events)
                unverified=[{'left':spans[a].eid,'right':spans[b].eid,'relation':'embedding_neighbor'} for a,b in pairs]
                semantic=extend_pool(flat,edges); broad=extend_pool(flat,unverified)
                scores={}
                for pool in (flat,semantic,broad):
                    key=tuple(pool)
                    if key in scores: continue
                    values=np.asarray(ce.predict([(case.question,documents[e]) for e in pool],
                        batch_size=params.rerank_batch_size,show_progress_bar=False))
                    if len(values)!=len(pool) or not np.isfinite(values).all(): raise ValueError('Malformed CE scores')
                    scores[key]=dict(zip(pool,map(float,values),strict=True))
                unit={e.event_id:e.session_id if case.retrieval_unit=='session' else e.event_id for e in events}
                for arm in ARMS:
                    rid=f'{dataset}/{arm}/{case.id}'
                    if rid in done: continue
                    if arm=='joint':
                        selected,packet=base_selected,base_packet; pool=[memory._episode_order[i] for i in memory.last_candidate_pool]
                        certificate={'policy':'frozen_joint'}
                    else:
                        pool=flat if arm=='flat48' else broad if arm=='no_verifier' else semantic
                        score=scores[tuple(pool)]; ranking=sorted(pool,key=lambda e:(-score[e],e))
                        pack=Pack(packet_index,documents,case.question,score,budget)
                        selected,packet,certificate=choose_packet(arm,ranking,edges,case.question,pack,order)
                    count=len(tokenizer.encode(packet,add_special_tokens=False))
                    if count>budget or len(selected)>8 or len(pool)>48 or any(s['event_id'] not in pool for s in selected):
                        raise ValueError('Fair source/candidate/token bound violated')
                    gold=set(case.gold); seen={unit[e] for e in pool}
                    row={'id':rid,'dataset':dataset,'method':arm,'question_id':case.id,'cluster':cluster,'category':case.category,
                        'selected':selected,'evidence':packet,'evidence_tokens':count,'matched_token_ceiling':budget,
                        'metrics':rank_metrics(selected,case,8),'candidate_recall@48':len(gold&seen)/len(gold) if gold else 0.,
                        'candidate_count':len(pool),'semantic_edges':edges,'pair_count':len(pairs),
                        'pair_labels':labels,'certificate':certificate,'status':'exposed_development_not_paper',
                        'qa_calls':0,'judge_calls':0}
                    append_row(out/'retrieval.jsonl',row); done[rid]=row
                summarize(out,expected)
                print(f'QUERY {case.id} pairs={len(pairs)} edges={dict(Counter(e["relation"] for e in edges))} cells={len(done)}/600',flush=True)
            if state_fingerprint(memory)!=before: raise ValueError('Source-only read changed frozen memory')
            encoder.cache.clear()
        if set(done)!=expected: raise ValueError('Incomplete original100')
        write_json(out/'semantic_usage.json',{'fresh_model_pairs_this_continuation':nli.actual_pairs,'QA_or_hosted_calls':0,
            'scope':'source classifier calls, not answer-model tokens or efficiency benchmark'})
        summarize(out,expected)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',default=OUTPUT)
    run(p.parse_args().output)
