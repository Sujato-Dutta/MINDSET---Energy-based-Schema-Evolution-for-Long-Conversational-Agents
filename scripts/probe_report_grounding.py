"""Exact development-100: report grounding vs matched read/write controls."""
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict
import argparse
import json
import os
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mindset.methods.report_grounding import PredicateEquivalence, ReportCatalog, inject_witnesses, select_with_witnesses
from mindset.methods.mdl_state import MDLEnergyMemory
from mindset.methods.compact import CompactConfig
from mindset.methods.base import MemoryHit
from mindset.acl.joint_iteration import protocol, groups_for
from mindset.acl.pipeline import local_stack, record_environment, source_fingerprint, write_csv, rank_metrics
from mindset.acl.io import read_json, write_json, digest_file, append_row, unique_rows
from mindset.acl.runtime import CachedEncoder
from mindset.acl.evidence import EvidenceCache
from mindset.acl.clients import load_tokenizer, messages, ANSWER_PROMPT_HASH
from mindset.retrieval.hybrid import reciprocal_rank_fusion, ranked_indices
from mindset.utils.hashing import stable_hash
from mindset.methods.joint import utc_time

ARMS=('energy','two_reports','chronological','pool_only','previous')
METRICS=('recall@8','precision@8','mrr@8','ndcg@8')


def summary(out,cohort=None):
    rows=list(unique_rows(out/'evidence.jsonl').values()); metrics=[]; paired=[]
    expected_questions=cohort['count'] if cohort else 100
    panels={'all':lambda r:True}
    if cohort and cohort.get('schema_version')=='mindset-native-knowledge-update-validation-v1':
        overlap=set(cohort['overlap_development100_ids']); exposed=set(cohort['known_v2_scored_exposure'])
        panels.update(development100_overlap=lambda r:r['question_id'] in overlap,
                      not_in_development100=lambda r:r['question_id'] not in overlap,
                      not_in_known_v2_scored_artifacts=lambda r:r['question_id'] not in exposed)
    for panel,predicate in panels.items():
      for dataset in ('locomo','longmemeval'):
        panel_rows=[r for r in rows if predicate(r)]
        primary={r['question_id']:r for r in panel_rows if r['dataset']==dataset and r['arm']=='energy'}
        for arm in ARMS:
            subset=[r for r in panel_rows if r['dataset']==dataset and r['arm']==arm]
            if not subset: continue
            metrics.append({'panel':panel,'dataset':dataset,'arm':arm,'questions':len(subset),
                **{m:float(np.mean([r['retrieval'][m] for r in subset])) for m in METRICS},
                'candidate_recall@48':float(np.mean([r['candidate_recall'] for r in subset])),
                'mean_evidence_tokens':float(np.mean([r['evidence_tokens'] for r in subset])),
                'queries_bound':sum(r['report_certificate'].get('bound_reports',0)>0 for r in subset),
                'queries_with_reserved_reports':sum(bool(r['report_certificate']['reserved_ids']) for r in subset),
                'queries_with_committed_edge':sum(r['report_certificate']['committed_report_edge'] for r in subset)})
            if cohort and cohort.get('schema_version')=='mindset-native-knowledge-update-validation-v1':
                metrics[-1].update(mean_gold_session_report_presence=float(np.mean([
                    r['grounding_diagnostic']['gold_session_report_presence'] for r in subset])),
                    mean_gold_session_bound_report_presence=float(np.mean([
                    r['grounding_diagnostic']['gold_session_bound_report_presence'] for r in subset])),
                    queries_with_generic_state_exclusion=sum(bool((r.get('state_certificate') or {}).get('excluded')) for r in subset))
            if arm!='energy':
                pairs=[(primary[r['question_id']],r) for r in subset if r['question_id'] in primary]
                paired.append({'panel':panel,'dataset':dataset,'control':arm,'pairs':len(pairs),
                    'identical_complete_prompts':sum(a['prompt_sha256']==b['prompt_sha256'] for a,b in pairs),
                    **{m+'_delta':float(np.mean([a['retrieval'][m]-b['retrieval'][m] for a,b in pairs])) for m in METRICS}})
    if metrics: write_csv(out/'retrieval.csv',metrics)
    if paired: write_csv(out/'causal_opportunity.csv',paired)
    write_json(out/'status.json',{'status':'complete_development' if len(rows)==expected_questions*len(ARMS) else 'partial_development',
        'completed_cells':len(rows),'expected_cells':expected_questions*len(ARMS),'qa_calls':0,'judge_calls':0,'full_runs_authorized':False,
        'scope':'repeated-look development retrieval; NOT answer accuracy or confirmatory superiority'})
    primary={r['question_id']:r for r in rows if r['arm']=='energy'}
    controls=[r for r in rows if r['arm']=='state_off']
    changed=[r['question_id'] for r in controls if r['question_id'] in primary
             and r['prompt_sha256']!=primary[r['question_id']]['prompt_sha256']]
    write_json(out/'answer_gate.json',{'paired_questions':len(controls),'changed_vs_state_off_ids':changed,
        'complete_retrieval':len(rows)==expected_questions*len(ARMS),'qa_calls':0,'judge_calls':0,
        'scope':'actual input difference only; not evidence of answer improvement',
        'next_action':'inspect changed packets before matched native answer/judge validation' if changed else 'no energy-vs-state-off answer-input difference'})


def run(a):
    global ARMS
    if a.knowledge_update:
        from mindset.acl.knowledge_update import protocol as chosen_protocol
        ARMS=('energy','two_reports','chronological','latest_report','pool_only','state_off','previous')
    else: chosen_protocol=protocol
    cfg,cohort,artifact=chosen_protocol(a.manifest,a.calibration)
    if a.typed:
        from mindset.methods.typed_grounding import TypedReportCatalog as Catalog
    elif a.broad:
        from mindset.methods.broad_grounding import BroadReportCatalog as Catalog
    else: Catalog=ReportCatalog
    out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    projection=Path(a.projections); nli=Path(a.nli)
    if read_json(projection/'status.json')['status']!='complete_projection': raise ValueError('Incomplete report inputs')
    if read_json(nli/'status.json')['status']!='complete_nli_cache': raise ValueError('Incomplete predicate binding')
    if read_json(nli/'audit.json')['projection_sha256']!=digest_file(projection/'reports.jsonl'): raise ValueError('NLI/input mismatch')
    if a.knowledge_update:
        frozen_nli=read_json(ROOT/'outputs/acl/report_nli_27346/audit.json')
        actual_nli=read_json(nli/'audit.json')
        if any(actual_nli[k]!=frozen_nli[k] for k in ('repository','revision','label_mapping','model_files')):
            raise ValueError('Frozen predicate model changed')
    sources=defaultdict(dict); queries=defaultdict(dict)
    for r in unique_rows(projection/'reports.jsonl').values():
        key=(r['dataset'],r['cluster'])
        if r['kind']=='episode': sources[key][r['source_id']]=r
        else: queries[key][r['question_id']]=r
    if {(d,r['question_id']) for (d,c),qs in queries.items() for r in qs.values()}!={(r['dataset'],r['id']) for r in cohort['records']}:
        raise ValueError('Frozen cohort mismatch')
    with (nli/'pairs.jsonl').open() as f: equivalence=PredicateEquivalence([json.loads(line) for line in f])
    prior=ROOT/('outputs/acl/knowledge_update_validation_27363/evidence.jsonl' if a.broad
                else 'outputs/acl/mindset_mdl_natural_27335/evidence.jsonl')
    previous={(r['dataset'],r['question_id']):r for r in unique_rows(prior).values() if r['arm']==('previous' if a.broad else 'mdl')}
    write_json(out/'audit.json',{'scope':'native_knowledge_update_frozen_validation' if a.knowledge_update else 'report_grounding_development100','source_fingerprint':source_fingerprint(),
        'worker_sha256':digest_file(__file__),'controller_sha256':digest_file(ROOT/'src/mindset/methods/report_grounding.py'),
        'manifest_sha256':digest_file(a.manifest),'calibration_sha256':digest_file(a.calibration),
        'projection_sha256':digest_file(projection/'reports.jsonl'),'parser_audit':read_json(projection/'audit.json'),
        'nli_audit':read_json(nli/'audit.json'),'nli_pairs_sha256':digest_file(nli/'pairs.jsonl'),
        'prior_evidence_sha256':digest_file(prior),'arms':ARMS,'candidate_budget':48,'episode_budget':8,
        'common_pool':'same flat RRF pool + union of all selectors raw witness IDs, bounded48, keep original first24',
        'previous_arm':'unchanged flat48; checks exact prior prompt identity',
        'report_truth':'report distributions only; past claims never certify current state',
        'answer_prompt_sha256':ANSWER_PROMPT_HASH,'new_fitted_scalars':0,'api_calls':0,
        'broad_grounding':a.broad,'typed_grounding':a.typed,
        'typed_catalog_sha256':digest_file(ROOT/'src/mindset/methods/typed_grounding.py') if a.typed else None,
        'candidate_catalog_sha256':digest_file(ROOT/'src/mindset/methods/broad_grounding.py') if a.broad else None,
        'exposure_scope':'all 78 KU now previously scored; redesigned grounding is development, not pristine validation' if a.broad else 'original frozen diagnostic',
        'full_runs_authorized':False,'score_reuse':'quality only, not latency'},immutable=True)
    record_environment(out); encoder,reranker=local_stack(cfg,cache=False); encoder=CachedEncoder(encoder,enabled=True)
    tokenizer=load_tokenizer(read_json(ROOT/'configs/checkpoints.lock.json')[cfg['evidence_tokenizer']])
    params={**{k:v for k,v in cfg['mindset'].items() if k!='profile'},**artifact['selected_parameters']}
    completed=0
    for dataset,cluster,cases in groups_for(cohort,cfg):
        key=(dataset,cluster); events=cases[0].events
        base=MDLEnergyMemory(CompactConfig(**params),encoder=encoder,local_reranker=reranker); base.ingest(events)
        catalogs={policy:Catalog(sources[key],events,equivalence,policy=policy) for policy in ('mdl','two_reports')}
        for policy,catalog in catalogs.items():
            if policy=='mdl' and any(d['chosen']['adjusted_energy']!=min(p['adjusted_energy'] for p in d['candidates']) for d in catalog.audit):
                raise ValueError('MDL did not choose the feasible energy minimum')
            write_json(out/'structure'/f'{dataset}-{cluster}-{policy}.json',{'report_groups':catalog.groups,'decisions':catalog.audit})
        index=EvidenceCache(events,encoder,tokenizer,cfg)
        for case in cases:
            qr=queries[key][case.id]
            if qr['text']!=case.question: raise ValueError('Query projection mismatch')
            qv=encoder.encode([case.question],normalize_embeddings=True,convert_to_numpy=True)[0]
            fused=reciprocal_rank_fusion([ranked_indices(base._episode_vectors@qv),ranked_indices(base._bm25.scores(case.question))])
            flat=sorted(fused,key=lambda j:(-fused[j],j))[:48]
            choices={}
            for arm in ARMS:
                cat=catalogs['two_reports' if arm=='two_reports' else 'mdl']
                selector='chronological' if arm=='chronological' else 'latest' if arm=='latest_report' else 'off' if arm in ('pool_only','state_off','previous') else 'energy'
                ids,certificate=cat.witnesses(qr,selector=selector)
                deadline=utc_time(case.question_date)
                if a.broad and deadline and any(utc_time(cat.by_id[eid].timestamp)>deadline for eid in ids):
                    raise ValueError('Future source report cannot be a question-time witness')
                choices[arm]=([base._index_by_id[eid] for eid in ids],certificate)
            anchors=list(dict.fromkeys(i for ids,c in choices.values() for i in ids))
            common=inject_witnesses(flat,anchors)
            union=sorted(set(common)|set(flat))
            scores=reranker.predict([(case.question,base._documents[i]) for i in union],batch_size=8,show_progress_bar=False)
            if len(scores)!=len(union) or not np.isfinite(scores).all(): raise ValueError('Invalid CE scores')
            scoremap=dict(zip(union,scores,strict=True))
            unit={e.event_id:e.session_id if case.retrieval_unit=='session' else e.event_id for e in events}
            source_ids={r['eid'] for g in catalogs['mdl'].groups for r in g['reports']}
            bound_group,binding=catalogs['mdl'].bind(qr)
            bound_ids={r['eid'] for r in bound_group['reports']} if bound_group else set()
            gold=set(case.gold)
            grounding={'extracted_report_episodes':len(source_ids),'bound_report_episodes':len(bound_ids),
                'binding':binding,'gold_session_report_presence':len({unit[eid] for eid in source_ids}&gold)/len(gold),
                'gold_session_bound_report_presence':len({unit[eid] for eid in bound_ids}&gold)/len(gold),
                'interpretation':'presence in annotated session ONLY; NOT correctness of extracted fact or transition',
                'transition_accuracy':None,'transition_accuracy_unavailable_reason':'no natural transition ground-truth labels',
                'generic_state_scope':asdict(base._scope(case.question,case.question_date))}
            if a.broad:
                grounding['broad_query_mode']=qr['report_query']['mode']
                grounding['bound_report_sequence']=bound_group['reports'] if bound_group else []
            for arm in ARMS:
                pool=flat if arm=='previous' else common; witnesses,certificate=choices[arm]
                state_certificate=None
                # Nonactivated questions retain the exact previous state readout.
                if arm=='state_off':
                    # Match the generic state's CE tie rule exactly; otherwise
                    # a tie-order artifact could look like a state benefit.
                    selected_indices=sorted(pool,key=lambda i:(-float(scoremap[i]),base._episode_order[i]))[:8]
                    hits=[MemoryHit(base._episode_order[i],base._documents[i],float(scoremap[i]),
                        (base._episode_order[i],),'raw_ce_state_off') for i in selected_indices]
                elif not witnesses:
                    hits=base.select_scored(case.question,pool,[scoremap[i] for i in pool],8,case.question_date)
                    state_certificate=base.last_readout_certificate.copy()
                else:
                    selected_indices=select_with_witnesses(pool,[scoremap[i] for i in pool],witnesses)
                    hits=[MemoryHit(base._episode_order[i],base._documents[i],float(scoremap[i]),
                        (base._episode_order[i],),'grounded_raw_report') for i in selected_indices]
                selected,packet=index.packet(hits,case.question); prompt=messages(case.question,packet,case.question_date)
                row={'id':f'{dataset}/{arm}/{case.id}','dataset':dataset,'cluster':cluster,'question_id':case.id,
                     'question':case.question,'question_date':case.question_date,'category':case.category,'answer':case.answer,
                     'arm':arm,'selected':selected,'evidence':packet,'messages':prompt,'prompt_sha256':stable_hash(prompt),
                     'evidence_tokens':len(tokenizer.encode(packet,add_special_tokens=False)),
                     'candidate_ids':[base._episode_order[i] for i in pool],
                     'candidate_recall':len({unit[base._episode_order[i]] for i in pool}&set(case.gold))/len(set(case.gold)),
                     'retrieval':rank_metrics(selected,case,8),'report_certificate':certificate,
                     'grounding_diagnostic':grounding,'state_certificate':state_certificate}
                if arm=='previous' and (dataset,case.id) in previous and row['prompt_sha256']!=previous[(dataset,case.id)]['prompt_sha256']:
                    raise ValueError('Previous MDL prompt reproduction failed')
                append_row(out/'evidence.jsonl',row); completed+=1
            summary(out,cohort); print(f'{completed}/{cohort["count"]*len(ARMS)} report-grounding cells',flush=True)
        encoder.cache.clear(); del base,index,catalogs
    summary(out,cohort)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default=str(ROOT/'data/manifests/mindset_v2_development100.json'))
    for name in ('calibration','projections','nli','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--knowledge-update',action='store_true')
    p.add_argument('--broad',action='store_true')
    p.add_argument('--typed',action='store_true')
    a=p.parse_args()
    if a.typed and not (a.broad and a.knowledge_update): p.error('Typed diagnostic requires --broad --knowledge-update')
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    run(a)
