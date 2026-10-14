"""Full-panel descriptive paired audit, not held-out significance or SOTA."""
import argparse
from collections import Counter
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def paired(first,second):
    ids=sorted(set(first)&set(second))
    if not ids: raise ValueError('No shared questions')
    differences=[first[q]-second[q] for q in ids]
    return {'questions':len(ids),'mean_difference':sum(differences)/len(ids),
        'first_mean':sum(first[q] for q in ids)/len(ids),'second_mean':sum(second[q] for q in ids)/len(ids),
        'wins':sum(d>1e-12 for d in differences),'ties':sum(abs(d)<=1e-12 for d in differences),
        'losses':sum(d < -1e-12 for d in differences),'ids':ids}


def analyze(records,prepared,legacy):
    from mindset.evaluation.locomo import locomo_native_score
    from mindset.datasets.locomo import CATEGORY_NAMES
    expected={r['id'] for r in prepared}
    if len(expected)!=len(prepared) or len({r['id'] for r in records})!=len(records): raise ValueError('Duplicate cells')
    if {r['id'] for r in records}!=expected: raise ValueError('Requires complete declared panels, not favourable prefix')
    source={r['id']:r for r in prepared}; panels={}; summaries=[]
    metrics=('native_f1','recall@8','precision@8','mrr@8','ndcg@8')
    for r in records:
        if r['prompt_sha256'] is None or r['selected']!=source[r['id']]['selected']: raise ValueError('Source packet changed')
        category=next(int(k) for k,v in CATEGORY_NAMES.items() if v==r['category'])
        score=locomo_native_score(r['prediction'],r['answer'],category)
        if abs(score-r['native_f1'])>1e-12: raise ValueError('Native F1 changed')
        panels.setdefault(r['method'],{})[r['question_id']]=r
    for method,panel in panels.items():
        values=list(panel.values())
        summaries.append({'method':method,'questions':len(values),
            **{m:sum((r['native_f1'] if m=='native_f1' else r['retrieval'][m]) for r in values)/len(values) for m in metrics},
            'incomplete_responses':sum(r.get('incomplete_response',False) for r in values),
            'abstentions':sum('no information available' in r['prediction'].casefold() for r in values),
            'mean_answer_words':sum(len(r['prediction'].split()) for r in values)/len(values),
            'mean_reference_words':sum(len(str(r['answer']).split()) for r in values)/len(values),
            'logical_input_tokens':sum(r['input_tokens'] for r in values),'logical_output_tokens':sum(r['output_tokens'] for r in values)})
    comparisons=[]
    for first,second in (('retained_v1','joint'),('retained_v1','flat48'),('joint','flat48'),
                         ('historical_ids','joint'),('historical_ids','flat48'),('historical_ids','retained_v1')):
        a=panels[first]; b=panels[second]
        for metric in metrics:
            def scores(panel): return {q:(r['native_f1'] if metric=='native_f1' else r['retrieval'][metric]) for q,r in panel.items()}
            result=paired(scores(a),scores(b))
            same_prompt=sum(a[q]['prompt_sha256']==b[q]['prompt_sha256'] for q in result['ids'])
            comparisons.append({'first':first,'second':second,'metric':metric,**result,'identical_prompts':same_prompt})
    old={r['id']:r for r in legacy}; historical=[]
    for method,panel in panels.items():
        shared=sorted(set(panel)&set(old))
        for q in shared:
            if panel[q]['question']!=old[q]['question'] or panel[q]['answer']!=old[q]['answer']:
                raise ValueError('Historical targets differ')
        a={q:panel[q]['native_f1'] for q in shared}
        b={q:locomo_native_score(old[q]['prediction'],old[q]['answer'],int(old[q]['category'])) for q in shared}
        historical.append({'method':method,'metric':'native_f1',**paired(a,b),
            'limitation':'different reader/prompt/batching; this is NOT a method-only contrast'})
    reconstruction=[r for r in prepared if r['method']=='retained_v1' and 'matches_historical_selected_ids' in r]
    return {'scope':'exposed LC development; descriptive; no retuning, new significance or SOTA claim',
        'questions':len({r['question_id'] for r in records}),'cells':len(records),'metrics':summaries,
        'paired_same_reader':comparisons,'paired_against_historical_gpt_oss':historical,
        'v1_reconstruction':{'shared_questions':len(reconstruction),
            'exact_selected_id_order_matches':sum(r['matches_historical_selected_ids'] for r in reconstruction),
            'limitation':'retained defaults/current pinned encoder and CE; exact old source checkout unavailable'},
        'judge_calls':0,'full_runs_authorized':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--answers',required=True); p.add_argument('--prepared',required=True); p.add_argument('--output',required=True)
    a=p.parse_args()
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    from mindset.acl.io import unique_rows,read_rows,digest_file,write_json
    from mindset.acl.pipeline import write_csv
    answer_path=Path(a.answers)/'predictions.jsonl'; prepared_path=Path(a.prepared)/'retrieval.jsonl'
    old_path=ROOT/'outputs/acl/legacy_score_inputs_r01/predictions.jsonl'
    report=analyze(list(unique_rows(answer_path).values()),list(unique_rows(prepared_path).values()),read_rows(old_path))
    report.update(job_id=os.environ['SLURM_JOB_ID'],hashes={str(p):digest_file(p) for p in (answer_path,prepared_path,old_path)})
    out=Path(a.output); write_json(out/'report.json',report,immutable=True)
    write_csv(out/'metrics.csv',report['metrics'])
    write_csv(out/'paired.csv',[{k:v for k,v in r.items() if k!='ids'} for r in report['paired_same_reader']])
    write_csv(out/'historical_paired.csv',[{k:v for k,v in r.items() if k!='ids'} for r in report['paired_against_historical_gpt_oss']])
    print('METRICS',report['metrics'],flush=True)
    print('FIDELITY',report['v1_reconstruction'],flush=True)
    for row in report['paired_same_reader']:
        if row['metric']=='native_f1': print('PAIRED_F1',{k:v for k,v in row.items() if k!='ids'},flush=True)


if __name__=='__main__': main()
