"""DGX-only paired QA/retrieval feedback, without significance overclaims."""
import os
from pathlib import Path
import numpy as np
from mindset.acl.io import read_json,write_json,unique_rows
from mindset.acl.pipeline import write_csv
from mindset.acl.ten_ideas_pilot import ARMS

def main():
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('DGX Slurm only')
    root=Path('outputs/acl/ten_ideas_r02')
    answers=root/'answers'
    if read_json(answers/'status.json')['status']!='complete_development_answers': raise ValueError('Incomplete matched answers')
    predictions=list(unique_rows(answers/'predictions.jsonl').values())
    if len(predictions)!=1100: raise ValueError('Wrong100-question panel')
    retrieval=read_json(root/'merged/analysis.json')
    comparisons=[]; changed_answers=[]
    for dataset in ('locomo','longmemeval'):
        base={r['question_id']:r for r in predictions if r['dataset']==dataset and r['method']=='joint'}
        if len(base)!=50: raise ValueError('Wrong baseline subset')
        for arm in ARMS:
            panel={r['question_id']:r for r in predictions if r['dataset']==dataset and r['method']==arm}
            if set(panel)!=set(base): raise ValueError('Paired question subset mismatch')
            field='native_f1' if dataset=='locomo' else 'token_f1'
            delta=[panel[q][field]-base[q][field] for q in sorted(base)]
            item={'dataset':dataset,'method':arm,'questions':50,
                'native_locomo_f1':float(np.mean([r['native_f1'] for r in panel.values()])) if dataset=='locomo' else None,
                'secondary_lme_token_f1':float(np.mean([r['token_f1'] for r in panel.values()])) if dataset=='longmemeval' else None,
                'delta_answer_f1':float(np.mean(delta)),
                'answer_wins':sum(d>1e-12 for d in delta),'answer_ties':sum(abs(d)<=1e-12 for d in delta),
                'answer_losses':sum(d < -1e-12 for d in delta),
                'identical_prompts':sum(panel[q]['prompt_sha256']==base[q]['prompt_sha256'] for q in base),
                'logical_input_tokens':sum(r['input_tokens'] for r in panel.values()),
                'logical_output_tokens':sum(r['output_tokens'] for r in panel.values()),
                'incomplete_responses':sum(r['incomplete_response'] for r in panel.values()),
                'native_lme_accuracy':None,'judge_status':'deferred',
                'scope':'exposed development, exact matched reader; not confirmatory significance or efficiency'}
            r=next(r for r in retrieval['comparisons'] if r['dataset']==dataset and r['method']==arm)
            item.update({k:v for k,v in r.items() if k not in item})
            comparisons.append(item)
            for q in sorted(base):
                difference=panel[q][field]-base[q][field]
                if abs(difference)<=1e-12: continue
                detail={'dataset':dataset,'method':arm,'question_id':q,'question':panel[q]['question'],
                    'reference_evaluator_only':panel[q]['answer'],'baseline_prediction':base[q]['prediction'],
                    'new_prediction':panel[q]['prediction'],'delta_f1':difference,
                    'baseline_selected_ids':[s['event_id'] for s in base[q]['selected']],
                    'new_selected_ids':[s['event_id'] for s in panel[q]['selected']]}
                changed_answers.append(detail)
                print('ANSWER_CHANGE',detail,flush=True)
    write_csv(root/'final_comparison.csv',comparisons)
    write_json(root/'final_analysis.json',{'qa':comparisons,'changed_answers':changed_answers,'retrieval':retrieval,
        'answer_cache_usage':read_json(answers/'status.json'),'inference':read_json(answers/'audit.json'),
        'claimed_statistical_superiority':False,'claimed_novelty':False,'full_runs_authorized':False})
    for row in comparisons: print('FINAL',row,flush=True)

if __name__=='__main__': main()
