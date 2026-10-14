"""Input-only binding/causal-opportunity audit on frozen development100."""
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mindset.acl.joint_iteration import protocol, groups_for
from mindset.acl.io import read_json, write_json, unique_rows, digest_file
from mindset.methods.report_grounding import PredicateEquivalence, ReportCatalog
from mindset.methods.joint import utc_time


def main(a):
    if a.knowledge_update:
        from mindset.acl.knowledge_update import protocol as chosen_protocol
    else: chosen_protocol=protocol
    cfg,cohort,artifact=chosen_protocol(a.manifest,a.calibration)
    projection=Path(a.projections); nli=Path(a.nli)
    if read_json(projection/'status.json')['status']!='complete_projection': raise ValueError('Incomplete projection')
    if read_json(nli/'audit.json')['projection_sha256']!=digest_file(projection/'reports.jsonl'): raise ValueError('Cache mismatch')
    sources=defaultdict(dict); queries=defaultdict(dict)
    for r in unique_rows(projection/'reports.jsonl').values():
        key=(r['dataset'],r['cluster'])
        if r['kind']=='episode': sources[key][r['source_id']]=r
        else: queries[key][r['question_id']]=r
    with (nli/'pairs.jsonl').open() as f: eq=PredicateEquivalence([json.loads(line) for line in f])
    result=[]
    for dataset,cluster,cases in groups_for(cohort,cfg):
        catalogs={p:ReportCatalog(sources[(dataset,cluster)],cases[0].events,eq,policy=p) for p in ('mdl','two_reports')}
        for case in cases:
            qr=queries[(dataset,cluster)][case.id]; choices={}
            for arm in ('energy','two_reports','chronological','latest_report'):
                cat=catalogs['two_reports' if arm=='two_reports' else 'mdl']
                selector='chronological' if arm=='chronological' else 'latest' if arm=='latest_report' else 'energy'
                ids,certificate=cat.witnesses(qr,selector=selector)
                deadline=utc_time(case.question_date)
                future=[eid for eid in ids if deadline and utc_time(cat.by_id[eid].timestamp)>deadline]
                if future: raise ValueError(('Future witness',case.id,future))
                choices[arm]={'source_ids':ids,'certificate':certificate}
            result.append({'dataset':dataset,'cluster':cluster,'question_id':case.id,'question':case.question,
                           'operator':qr['report_query']['mode'],'choices':choices})
    if len(result)!=cohort['count']: raise ValueError('Cohort count mismatch')
    summaries=[]
    for dataset in ('locomo','longmemeval'):
        subset=[r for r in result if r['dataset']==dataset]
        for arm in ('energy','two_reports','chronological','latest_report'):
            summaries.append({'dataset':dataset,'arm':arm,'questions':len(subset),
                'operators':dict(Counter(r['operator'] for r in subset)),
                'bound':sum(r['choices'][arm]['certificate'].get('bound_reports',0)>0 for r in subset),
                'reserved':sum(bool(r['choices'][arm]['source_ids']) for r in subset),
                'committed_edges':sum(r['choices'][arm]['certificate']['committed_report_edge'] for r in subset)})
    output={'scope':'inputs only, no answers/gold/retrieval metrics; report order not latent truth',
        'manifest_sha256':digest_file(a.manifest),'projection_sha256':digest_file(projection/'reports.jsonl'),
        'nli_pairs_sha256':digest_file(nli/'pairs.jsonl'),'summary':summaries,'query_bindings':result,'api_calls':0}
    write_json(a.output,output,immutable=True); print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default=str(ROOT/'data/manifests/mindset_v2_development100.json'))
    for name in ('calibration','projections','nli','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--knowledge-update',action='store_true')
    a=p.parse_args()
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    main(a)
