"""Paired diagnostic inspection, including partial runs; DGX Slurm only."""
import os
from collections import Counter
from pathlib import Path
from mindset.acl.pipeline import ROOT
from mindset.acl.io import read_json,write_json,unique_rows,digest_file


def run():
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('DGX Slurm only')
    source=ROOT/'outputs/acl/query_state_agent_r02'; rows=unique_rows(source/'predictions.jsonl')
    plans=unique_rows(source/'plans.jsonl'); results={}
    for dataset in ('locomo','longmemeval'):
        a={r['question_id']:r for r in rows.values() if r['dataset']==dataset and r['method']=='joint'}
        b={r['question_id']:r for r in rows.values() if r['dataset']==dataset and r['method']=='state_agent'}
        paired=sorted(set(a)&set(b)); changed=[]
        for q in paired:
            row=b[q]
            if row['execution']['bound'] or a[q]['prompt_sha256']!=row['prompt_sha256']:
                changed.append({'question_id':q,'question':row['question'],'category':row['category'],
                    'reference_evaluator_only':row['answer'],'baseline_prediction':a[q]['prediction'],
                    'agent_prediction':row['prediction'],'native_f1_difference':row['native_f1']-a[q]['native_f1'] if dataset=='locomo' else None,
                    'token_f1_difference':row['token_f1']-a[q]['token_f1'],
                    'packet_changed':a[q]['prompt_sha256']!=row['prompt_sha256'],
                    'rank_differences':{m:row['metrics'][m]-a[q]['metrics'][m] for m in row['metrics']},
                    'plan':plans[f'{dataset}/{q}']['plan'],'execution':row['execution']})
        results[dataset]={'paired_questions':len(paired),'baseline_cells':len(a),'agent_cells':len(b),
            'mean_token_f1_difference':sum(b[q]['token_f1']-a[q]['token_f1'] for q in paired)/len(paired) if paired else None,
            'mean_native_f1_difference':sum(b[q]['native_f1']-a[q]['native_f1'] for q in paired)/len(paired) if paired and dataset=='locomo' else None,
            'reason_counts':dict(Counter(b[q]['execution']['reason'] for q in paired)),
            'actual_version_links':sum(b[q]['execution']['transition_link_used'] for q in paired),
            'actual_branch_links':sum(b[q]['execution']['branch_link_used'] for q in paired),
            'bound_or_changed_cases':changed}
    out=ROOT/'outputs/acl'/f'query_state_inspection_{os.environ["SLURM_JOB_ID"]}'
    write_json(out/'inspection.json',{'source_status':read_json(source/'status.json'),'paired_results':results,
        'scope':'paired descriptive audit, partial run if fewer than 50 pairs per dataset; no significance or novelty claim'})
    for dataset,result in results.items():
        print('PAIRED',dataset,result,flush=True)


if __name__=='__main__': run()
