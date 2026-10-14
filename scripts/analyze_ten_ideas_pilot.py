"""DGX Slurm analysis of all ten ideas, with exact common question panels."""
import argparse
import os
from pathlib import Path
import numpy as np
from mindset.acl.io import read_json, write_json, unique_rows, append_row, digest_file
from mindset.acl.pipeline import write_csv
from mindset.acl.ten_ideas_pilot import ARMS, METRICS

def analyze(root, merged):
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('DGX Slurm only')
    root=Path(root); merged=Path(merged); merged.mkdir(parents=True,exist_ok=True)
    rows={}; audits={}; structures=[]; cdm=[]
    names=sorted(p.name for p in root.glob('shard-*')) or ['group-a','group-b','group-c','group-d']
    for name in names:
        directory=root/name
        if not (directory/'audit.json').exists(): continue
        audits[name]=read_json(directory/'audit.json')
        if (directory/'cdm_calibration.json').exists():
            artifact=read_json(directory/'cdm_calibration.json')
            cdm.append({key:artifact[key] for key in ('records_sha256','alpha','quantile','seed','n_independent_histories')})
        for key,row in unique_rows(directory/'retrieval.jsonl').items():
            if key in rows: raise ValueError('Duplicate arm/question cell')
            rows[key]=row
        for file in sorted((directory/'structure').glob('*.json')):
            item=read_json(file); item['source']=str(file); structures.append(item)
    if not audits: raise ValueError('No actual outputs')
    if cdm and any(a!=cdm[0] for a in cdm): raise ValueError('CDM calibration differs between history shards')
    for field in ('manifest_sha256','calibration_sha256','parameters','checkpoints','configuration','source_adapter',
                  'controller_source_sha256','worker_sha256'):
        values=[a[field] for a in audits.values()]
        if any(v!=values[0] for v in values): raise ValueError('Incompatible panels: '+field)
    results=[]
    for dataset in ('locomo','longmemeval'):
        base={r['question_id']:r for r in rows.values() if r['dataset']==dataset and r['method']=='joint'}
        for arm in ARMS:
            panel={r['question_id']:r for r in rows.values() if r['dataset']==dataset and r['method']==arm}
            common=sorted(set(base)&set(panel))
            if not common: continue
            item={'dataset':dataset,'method':arm,'questions':len(panel),'paired_questions':len(common),
                'changed_raw_evidence_packets':sum(panel[q]['evidence']!=base[q]['evidence'] for q in common),
                'changed_ordered_episode_ids':sum([s['event_id'] for s in panel[q]['selected']]!=[s['event_id'] for s in base[q]['selected']] for q in common),
                'mean_evidence_tokens':float(np.mean([panel[q]['evidence_tokens'] for q in common])),
                'diagnostic_cache_enabled_query_seconds':float(np.mean([panel[q]['diagnostic_query_seconds'] for q in common])),
                'scope':'descriptive repeated-look development; no statistical superiority or novelty claim'}
            for metric in (*METRICS,'candidate_pool_recall@48'):
                values=[panel[q]['metrics'][metric] if metric in METRICS else panel[q][metric] for q in common]
                baselines=[base[q]['metrics'][metric] if metric in METRICS else base[q][metric] for q in common]
                item[metric]=float(np.mean(values)); item['delta_'+metric]=float(np.mean(np.subtract(values,baselines)))
            item['retrieval_score']=.35*item['recall@8']+.25*item['mrr@8']+.25*item['ndcg@8']+.15*item['precision@8']
            item['delta_retrieval_score']=.35*item['delta_recall@8']+.25*item['delta_mrr@8']+.25*item['delta_ndcg@8']+.15*item['delta_precision@8']
            results.append(item)
    write_csv(merged/'comparison.csv',results)
    write_json(merged/'analysis.json',{'actual_source_audits':audits,'comparisons':results,'structure_profiles':structures,
        'qa_or_judge_accuracy_claims':False,'full_study':False})
    complete=len(rows)==1100 and all(sum(r['dataset']==d and r['method']==m for r in rows.values())==50
        for d in ('locomo','longmemeval') for m in ARMS)
    if complete:
        from mindset.acl.pipeline import ROOT
        manifest=read_json(ROOT/'data/manifests/mindset_v2_development100.json')
        expected={f'{r["dataset"]}/{m}/{r["id"]}' for r in manifest['records'] for m in ARMS}
        if set(rows)!=expected: raise ValueError('Wrong original100 question identities')
    write_json(merged/'status.json',{'status':'complete_development' if complete else 'partial_development',
        'completed_cells':len(rows),'expected_cells':1100,'unique_questions':len({(r['dataset'],r['question_id']) for r in rows.values()}),
        'qa_calls':0,'judge_calls':0,'full_runs_authorized':False})
    write_json(merged/'audit.json',{'kind':'ten_ideas_merged_pilot','source_audits':audits,
        'source_files_sha256':{name:digest_file(root/name/'retrieval.jsonl') for name in audits},
        'source_adapter':None,'methods':list(ARMS)},immutable=complete)
    # Write only new completed immutable cells; exact repeat is resumable.
    done=unique_rows(merged/'retrieval.jsonl')
    for key,row in sorted(rows.items()):
        if key in done and done[key]!=row: raise ValueError('Changed resumed merged row')
        if key not in done: append_row(merged/'retrieval.jsonl',row)
    for r in results: print('RESULT',r,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='outputs/acl/ten_ideas_r02')
    p.add_argument('--output',default='outputs/acl/ten_ideas_r02/merged')
    a=p.parse_args(); analyze(a.root,a.output)
