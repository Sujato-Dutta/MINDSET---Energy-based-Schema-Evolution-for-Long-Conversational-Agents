"""DGX-only label-free opportunity audit of the energy-free frontier."""
from pathlib import Path
from collections import defaultdict
import argparse
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mindset.acl.knowledge_update import protocol
from mindset.acl.joint_iteration import groups_for
from mindset.acl.io import read_json,write_json,unique_rows,digest_file
from mindset.methods.report_grounding import PredicateEquivalence
from mindset.methods.typed_grounding import TypedReportCatalog
from mindset.methods.conflict_frontier import ConflictFrontierCatalog


def main(a):
    cfg,cohort,_=protocol(a.manifest,a.calibration)
    projection=Path(a.projections); nli=Path(a.nli)
    if read_json(projection/'status.json')['status']!='complete_projection': raise ValueError('Incomplete projection')
    if read_json(nli/'status.json')['status']!='complete_nli_cache': raise ValueError('Incomplete predicate cache')
    if read_json(nli/'audit.json')['projection_sha256']!=digest_file(projection/'reports.jsonl'): raise ValueError('NLI/input mismatch')
    sources=defaultdict(dict); queries=defaultdict(dict)
    for row in unique_rows(projection/'reports.jsonl').values():
        key=(row['dataset'],row['cluster'])
        if row['kind']=='episode': sources[key][row['source_id']]=row
        else: queries[key][row['question_id']]=row
    actual={(d,r['question_id']) for (d,c),qs in queries.items() for r in qs.values()}
    if actual!={(r['dataset'],r['id']) for r in cohort['records']}: raise ValueError('Cohort changed')
    with (nli/'pairs.jsonl').open() as f: eq=PredicateEquivalence([json.loads(line) for line in f])
    results=[]
    for dataset,cluster,cases in groups_for(cohort,cfg):
        events=cases[0].events; source=sources[(dataset,cluster)]
        baseline=TypedReportCatalog(source,events,eq)
        proposed=ConflictFrontierCatalog(source,events,eq)
        for case in cases:
            qr=queries[(dataset,cluster)][case.id]
            if qr['text']!=case.question: raise ValueError('Question changed')
            choices={}
            for name in ('energy','latest_report'):
                ids,certificate=baseline.witnesses(qr,selector='energy' if name=='energy' else 'latest')
                choices[name]={'ids':ids,'certificate':certificate}
            for name,kwargs in (('frontier',{}),('no_opaque_tail',{'opaque':False}),('no_contrast',{'contrast':False})):
                ids,certificate=proposed.frontier(qr,**kwargs)
                choices[name]={'ids':ids,'certificate':certificate}
            results.append({'question_id':case.id,'question':case.question,'operator':qr['report_query']['mode'],
                'query_contract':qr['report_query']['contract'],'choices':choices,
                'frontier_sources':[{'id':eid,'timestamp':proposed.by_id[eid].timestamp,'text':proposed.by_id[eid].text}
                                    for eid in choices['frontier']['ids']]})
    summary={name:{'questions':len(results),'with_witnesses':sum(bool(r['choices'][name]['ids']) for r in results),
        'with_opaque_tail':sum(bool(r['choices'][name]['certificate'].get('opaque_tail_ids')) for r in results),
        'mean_witness_count':sum(len(r['choices'][name]['ids']) for r in results)/len(results)}
        for name in ('energy','latest_report','frontier','no_opaque_tail','no_contrast')}
    different={name:[r['question_id'] for r in results if r['choices'][name]['ids']!=r['choices']['frontier']['ids']]
               for name in ('energy','latest_report','no_opaque_tail','no_contrast')}
    write_json(a.output,{'scope':'input-only repeated-look development; no answers/gold/QA performance claims',
        'summary':summary,'different_witness_ids':different,'questions':results,
        'manifest_sha256':digest_file(a.manifest),'projection_sha256':digest_file(projection/'reports.jsonl'),
        'nli_sha256':digest_file(nli/'pairs.jsonl'),'worker_sha256':digest_file(__file__),
        'controller_sha256':digest_file(ROOT/'src/mindset/methods/conflict_frontier.py'),
        'api_calls':0,'new_fitted_scalars':0,'full_runs_authorized':False},immutable=True)
    print(json.dumps({'summary':summary,'different_witness_ids':different},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default=str(ROOT/'data/manifests/mindset_v2_knowledge_update_validation.json'))
    for name in ('calibration','projections','nli','output'): p.add_argument('--'+name,required=True)
    a=p.parse_args()
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    main(a)
