"""Source-only LC50 protocol audit; no model, tuning, repaired labels or judging."""
import argparse
from collections import Counter
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def caption(turn):
    value=turn.get('blip_caption')
    return value if isinstance(value,str) and value.strip() else None


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True)
    a=p.parse_args()
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    from mindset.acl.io import read_json,unique_rows,write_json,digest_file
    from mindset.acl.pipeline import config,cases_for
    from mindset.acl.iteration import verify_iteration
    cfg=config(); manifest=ROOT/'data/manifests/mindset_v2_development100.json'
    cohort=read_json(manifest); verify_iteration(cohort,cfg)
    selected_ids={r['id'] for r in cohort['records'] if r['dataset']=='locomo'}
    cases={c.id:c for c in cases_for('locomo',cfg) if c.id in selected_ids}
    source=ROOT/cfg['benchmarks']['locomo']['path']
    samples={str(r.get('sample_id',i)):r for i,r in enumerate(read_json(source))}
    original={}; totals=Counter()
    for cluster in sorted({c.cluster for c in cases.values()}):
        turns={t['dia_id']:t for key,session in samples[cluster]['conversation'].items()
               if key.startswith('session_') and not key.endswith('_date_time') for t in session}
        original[cluster]=turns
        totals['source_turns']+=len(turns); totals['source_turns_with_native_caption']+=sum(caption(t) is not None for t in turns.values())
    answer_dir=ROOT/'outputs/acl/development100_answers_r01'; answers=list(unique_rows(answer_dir/'predictions.jsonl').values())
    details=[]
    for method in ('joint','flat48'):
        panel=[r for r in answers if r['dataset']=='locomo' and r['method']==method]
        if {r['question_id'] for r in panel}!=selected_ids: raise ValueError('Incomplete original LC50')
        for row in panel:
            case=cases[row['question_id']]; turns=original[case.cluster]
            if any(e.text!=turns[e.event_id]['text'] for e in case.events): raise ValueError('Native text unexpectedly changed')
            gold_captions=[{'event_id':i,'text':turns[i]['text'],'native_caption':caption(turns[i])}
                           for i in case.gold if caption(turns[i]) is not None]
            selected_captions=[{'event_id':s['event_id'],'native_caption':caption(turns[s['event_id']]),
                'caption_string_in_saved_passage':caption(turns[s['event_id']]) in s['text']}
                for s in row['selected'] if caption(turns[s['event_id']]) is not None]
            details.append({'method':method,'question_id':case.id,'category':case.category,
                'question':case.question,'native_f1':row['native_f1'],'gold_caption_turns':gold_captions,
                'selected_caption_turns':selected_captions,'selected_ids':[s['event_id'] for s in row['selected']],
                'native_source_speakers':sorted({e.speaker for e in case.events}),
                'metadata_speakers':[samples[case.cluster]['conversation'].get(k) for k in ('speaker_a','speaker_b')],
                'limitation':'caption presence is not proof it contains the answer; no label repair or alias inference'})
    summaries=[]
    for method in ('joint','flat48'):
        rows=[r for r in details if r['method']==method]
        summaries.append({'method':method,'questions':len(rows),
            'questions_with_native_caption_in_gold':sum(bool(r['gold_caption_turns']) for r in rows),
            'questions_selecting_native_caption_turns':sum(bool(r['selected_caption_turns']) for r in rows),
            'questions_with_selected_caption_string_omitted':sum(any(not s['caption_string_in_saved_passage'] for s in r['selected_caption_turns']) for r in rows),
            'zero_f1_questions_with_caption_in_gold':sum(r['native_f1']==0 and bool(r['gold_caption_turns']) for r in rows)})
    report={'scope':'source-only audit; no generation, parameter fitting or changed cohort/source artifacts',
        'job_id':os.environ['SLURM_JOB_ID'],'totals':dict(totals),'summaries':summaries,'cases':details,
        'hashes':{str(p):digest_file(p) for p in (source,manifest,answer_dir/'predictions.jsonl')},
        'official_protocol_reference':'https://github.com/snap-research/locomo/blob/main/task_eval/gpt_utils.py',
        'protocol_note':'Official raw-dialog formatting includes blip_caption when present; current frozen text-only adapter does not append it',
        'qa_calls':0,'judge_calls':0,'full_runs_authorized':False}
    write_json(a.output,report,immutable=True)
    print('TOTALS',report['totals'],flush=True); print('SUMMARY',summaries,flush=True)
    for row in details:
        if row['method']=='joint' and row['gold_caption_turns']: print('GOLD_CAPTION_CASE',row,flush=True)


if __name__=='__main__': main()
