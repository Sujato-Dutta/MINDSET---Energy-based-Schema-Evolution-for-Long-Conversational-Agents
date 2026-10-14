"""Input-only owner/argument/readout audit on the frozen development100."""
from pathlib import Path
from collections import defaultdict
import argparse
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from mindset.acl.joint_iteration import protocol,groups_for
from mindset.acl.io import read_json,write_json,digest_file,unique_rows
from mindset.methods.report_grounding import PredicateEquivalence
from mindset.methods.conflict_frontier import ConflictFrontierCatalog
from mindset.methods.participant_frontier import ParticipantFrontierCatalog


def main(a):
    cfg,cohort,_=protocol(a.manifest,a.calibration); p=Path(a.projections); n=Path(a.nli)
    if read_json(p/'status.json')['status']!='complete_projection': raise ValueError('Incomplete projection')
    if read_json(n/'status.json')['status']!='complete_nli_cache': raise ValueError('Incomplete matcher')
    if read_json(n/'audit.json')['projection_sha256']!=digest_file(p/'reports.jsonl'): raise ValueError('Cache mismatch')
    sources=defaultdict(dict); queries=defaultdict(dict)
    for row in unique_rows(p/'reports.jsonl').values():
        key=(row['dataset'],row['cluster'])
        if row['kind']=='episode': sources[key][row['source_id']]=row
        else: queries[key][row['question_id']]=row
    if {(d,r['question_id']) for (d,c),qs in queries.items() for r in qs.values()}!={(r['dataset'],r['id']) for r in cohort['records']}:
        raise ValueError('Original development100 changed')
    with (n/'pairs.jsonl').open() as f: eq=PredicateEquivalence([json.loads(line) for line in f])
    result=[]
    for dataset,cluster,cases in groups_for(cohort,cfg):
        events=cases[0].events; proposed=ParticipantFrontierCatalog(sources[(dataset,cluster)],events,eq)
        self_only=ConflictFrontierCatalog(sources[(dataset,cluster)],events,eq)
        for case in cases:
            qr=queries[(dataset,cluster)][case.id]
            if qr['text']!=case.question: raise ValueError('Input question mismatch')
            choices={}
            for name,kwargs in (('participant',{}),('no_opaque_tail',{'opaque':False}),('no_contrast',{'contrast':False})):
                ids,certificate=proposed.frontier(qr,**kwargs); choices[name]={'ids':ids,'certificate':certificate}
            ids,certificate=self_only.frontier(qr); choices['self_only']={'ids':ids,'certificate':certificate}
            result.append({'question_id':case.id,'dataset':dataset,'question':case.question,
                'owner_attribution':qr['report_query']['owner_attribution'],'contract':qr['report_query']['contract'],
                'operator':qr['report_query']['mode'],'choices':choices,
                'sources':[{'id':eid,'speaker':proposed.by_id[eid].speaker,'text':proposed.by_id[eid].text,
                            'timestamp':proposed.by_id[eid].timestamp} for eid in choices['participant']['ids']]})
    summary=[]
    for dataset in ('locomo','longmemeval'):
        rows=[r for r in result if r['dataset']==dataset]
        if len(rows)!=50: raise ValueError('Missing cohort panel')
        for name in ('participant','self_only','no_opaque_tail','no_contrast'):
            summary.append({'dataset':dataset,'arm':name,'questions':len(rows),
                'owner_resolved':sum(r['contract']['owner'] is not None for r in rows),
                'bound':sum(r['choices'][name]['certificate'].get('bound_reports',0)>0 for r in rows),
                'with_witnesses':sum(bool(r['choices'][name]['ids']) for r in rows),
                'with_opaque_tail':sum(bool(r['choices'][name]['certificate'].get('opaque_tail_ids')) for r in rows)})
    write_json(a.output,{'summary':summary,'questions':result,'manifest_sha256':digest_file(a.manifest),
        'projection_sha256':digest_file(p/'reports.jsonl'),'nli_audit':read_json(n/'audit.json'),
        'worker_sha256':digest_file(__file__),'controller_sha256':digest_file(ROOT/'src/mindset/methods/participant_frontier.py'),
        'scope':'input-only development100 audit; NO answers/gold/retrieval or QA performance claims',
        'energy_used':False,'api_calls':0,'new_fitted_scalars':0,'full_runs_authorized':False},immutable=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default=str(ROOT/'data/manifests/mindset_v2_development100.json'))
    for name in ('calibration','projections','nli','output'): p.add_argument('--'+name,required=True)
    a=p.parse_args()
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    main(a)
