"""Independent terminal artifact checks and source-access failure analysis."""
from collections import Counter
import os
import re
from mindset.acl.io import read_json,write_json,unique_rows,digest_file
from mindset.acl.joint_iteration import protocol,groups_for
from mindset.acl.pipeline import ROOT,rank_metrics
from mindset.methods.joint import utc_time
from mindset.methods.transition_packets import ARMS,slot_matches


def run():
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('DGX Slurm only')
    out=ROOT/'outputs/acl/transition_packets_r01'
    if read_json(out/'status.json')['status']!='complete_development': raise ValueError('Incomplete retrieval')
    cfg,cohort,_=protocol(ROOT/'data/manifests/mindset_v2_development100.json',
        ROOT/'outputs/acl/mindset_joint_v2_27058/calibration/calibration.json')
    ids={(r['dataset'],r['id']) for r in read_json(out/'subset.json')['records']}
    cases={(d,c.id):c for d,cluster,queries in groups_for(cohort,cfg) for c in queries if (d,c.id) in ids}
    rows=unique_rows(out/'retrieval.jsonl')
    expected={f'{d}/{arm}/{q}' for d,q in ids for arm in ARMS}
    if set(rows)!=expected: raise ValueError('Wrong terminal matrix')
    coverage=[]; bound=[]; catalogs={}
    for key,case in cases.items():
        d,q=key; baseline=rows[f'{d}/joint/{q}']; path=out/'catalogs'/f'{d}-{case.cluster}.json'
        if (d,case.cluster) not in catalogs: catalogs[(d,case.cluster)]=read_json(path)
        catalog=catalogs[(d,case.cluster)]; edges=catalog['edges']
        anchors={s['event_id'] for s in baseline['selected'][:4]}; text=case.question.casefold()
        anchored=[e for e in edges if anchors.intersection((e['left'],e['right']))]
        owner=[e for e in anchored if re.search(r'(?<!\w)'+re.escape(e['owner'])+r'(?!\w)',text)
            or (e['owner']=='user' and re.search(r'\b(?:i|my|me)\b',text))]
        scoped=[e for e in owner if slot_matches(e['slot'],text)]
        date=utc_time(case.question_date); original={e.event_id:e for e in case.events}
        timely=[e for e in scoped if not date or all(utc_time(original[eid].timestamp)<=date for eid in (e['left'],e['right']))]
        admitted=rows[f'{d}/compact_energy/{q}']['admitted_contrasts']
        coverage.append({'dataset':d,'question_id':q,'category':case.category,'question':case.question,
            'existing_contrasts':len(edges),'retrieval_anchored_contrasts':len(anchored),
            'owner_grounded_contrasts':len(owner),'attribute_scoped_contrasts':len(scoped),
            'time_eligible_contrasts':len(timely),'admitted_contrasts':len(admitted),
            'intent':baseline['intent']})
        for arm in ARMS:
            row=rows[f'{d}/{arm}/{q}']; actual=rank_metrics(row['selected'],case,8)
            if actual!=row['metrics']: raise ValueError('Saved ranking metrics mismatch')
            if row['evidence_tokens']>baseline['evidence_tokens']: raise ValueError('Unmatched token ceiling')
            if row['certificate']['bound']:
                details=[]
                for edge in admitted:
                    details.append({'edge':edge,'both_endpoints_already_in_baseline':
                        {edge['left'],edge['right']}.issubset({s['event_id'] for s in baseline['selected']}),
                        'source_left':original[edge['left']].text,'source_right':original[edge['right']].text})
                bound.append({'dataset':d,'method':arm,'question_id':q,'question':case.question,
                    'reference_evaluator_only':case.answer,'packet_changed':row['evidence']!=baseline['evidence'],
                    'certificate':row['certificate'],'source_contrasts':details})
        a=rows[f'{d}/compact_energy/{q}']['certificate'].get('proposals',[])
        b=rows[f'{d}/compact_no_energy/{q}']['certificate'].get('proposals',[])
        if a!=b: raise ValueError('Energy ablation proposal mismatch')
    summary={}
    for d in ('locomo','longmemeval'):
        selected=[r for r in coverage if r['dataset']==d]; cats=[c for (dataset,cl),c in catalogs.items() if dataset==d]
        kinds=Counter()
        for c in cats: kinds.update(c['contrast_types'])
        summary[d]={'questions':len(selected),'histories':len(cats),'source_reports':sum(c['source_reports'] for c in cats),
            'contrast_types':dict(kinds),'ambiguous_time_groups':sum(c['ambiguous_time_groups'] for c in cats),
            'category_counts':dict(Counter(r['category'] for r in selected)),
            **{field+'_questions':sum(r[field]>0 for r in selected) for field in
                ('existing_contrasts','retrieval_anchored_contrasts','owner_grounded_contrasts',
                 'attribute_scoped_contrasts','time_eligible_contrasts','admitted_contrasts')}}
    write_json(out/'coverage_audit.json',{'summary':summary,'per_question':coverage,'bound_cases':bound,
        'retrieval_sha256':digest_file(out/'retrieval.jsonl'),'subset_sha256':digest_file(out/'subset.json'),
        'checks':'exact question/arm matrix, recomputed ranking scores, matched token ceiling, same compact proposals',
        'qa_calls':0,'judge_calls':0,'full_runs_authorized':False})
    print(summary,flush=True)


if __name__=='__main__': run()
