"""Read-only score comparison of historical and bounded development artifacts.

Must run in a DGX Slurm allocation. No generation, retrieval, or judge calls.
Different checkpoints/prompts still preclude causal attribution of a gap.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def rows(path):
    result=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').split('\n') if line.strip()]
    if len({r['id'] for r in result})!=len(result): raise ValueError('Duplicate artifact IDs')
    return result


def summary(records):
    from mindset.evaluation.locomo import locomo_native_score
    scores=[locomo_native_score(r['prediction'],r['answer'],int(r['category'])) for r in records]
    return {'questions':len(records),'native_f1':sum(scores)/len(scores) if scores else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy',required=True)
    p.add_argument('--current',required=True)
    p.add_argument('--manifest',default=str(ROOT/'data/manifests/mindset_v2_development100.json'))
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    from mindset.acl.io import read_json,write_json,digest_file
    from mindset.datasets.locomo import CATEGORY_NAMES
    old_dir=Path(a.legacy); new_dir=Path(a.current)
    old=rows(old_dir/'predictions.jsonl'); current=rows(new_dir/'predictions.jsonl')
    old_by_id={r['id']:r for r in old}; old_metric=read_json(old_dir/'metrics.json')
    total=summary(old)
    if not math.isclose(total['native_f1'],old_metric['native_f1'],abs_tol=1e-12):
        raise ValueError('Original stored native F1 not reproduced')
    expected={r['id'] for r in read_json(a.manifest)['records'] if r['dataset']=='locomo'}
    report={'scope':'descriptive artifact audit; different model, prompt and method; NOT a causal regression test',
        'job_id':os.environ['SLURM_JOB_ID'],'legacy_model':read_json(old_dir/'run_config.json')['arguments']['model'],
        'current_checkpoint':read_json(new_dir/'audit.json')['checkpoint'],
        'legacy_all':total,'legacy_answerable_categories_1_4':summary([r for r in old if int(r['category'])!=5]),
        'legacy_by_category':{str(c):summary([r for r in old if int(r['category'])==c]) for c in range(1,6)},
        'current':{},'matched_shared_questions':{},
        'hashes':{str(path):digest_file(path) for path in (old_dir/'predictions.jsonl',old_dir/'metrics.json',
            old_dir/'run_config.json',new_dir/'predictions.jsonl',new_dir/'audit.json',Path(a.manifest))},
        'qa_calls':0,'judge_calls':0}
    for method in ('joint','flat48','lightmem'):
        selected=[dict(r) for r in current if r['dataset']=='locomo' and r['method']==method]
        for r in selected:
            r['category']=next(int(k) for k,v in CATEGORY_NAMES.items() if v==r['category'])
            fresh=summary([r])['native_f1']
            if not math.isclose(fresh,r['native_f1'],abs_tol=1e-12): raise ValueError('Current stored F1 not reproduced')
        ids={r['question_id'] for r in selected}
        if ids-expected: raise ValueError('Unexpected current question')
        report['current'][method]={**summary(selected),'expected':len(expected),'complete_panel':ids==expected,
            'by_category':{str(c):summary([r for r in selected if r['category']==c]) for c in range(1,5)}}
        shared=[r for r in selected if r['question_id'] in old_by_id]
        for r in shared:
            previous=old_by_id[r['question_id']]
            if any(r[k]!=previous[k] for k in ('question','answer','category')):
                raise ValueError('Matched question/reference/category changed')
        report['matched_shared_questions'][method]={'ids':[r['question_id'] for r in shared],
            'legacy':summary([old_by_id[r['question_id']] for r in shared]),'current':summary(shared),
            'limitation':'same IDs only; method, model and prompt differ; not a method-only comparison'}
    write_json(a.output,report,immutable=True)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
