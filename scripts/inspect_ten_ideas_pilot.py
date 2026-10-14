"""Source-audited diagnostic details; run in a DGX Slurm allocation only."""
import os
from pathlib import Path
from collections import Counter,defaultdict
from mindset.acl.io import read_json,write_json,unique_rows
from mindset.acl.joint_iteration import protocol,groups_for
from mindset.acl.pipeline import ROOT
from mindset.acl.ten_ideas_pilot import ARMS

if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('DGX Slurm only')
root=ROOT/'outputs/acl/ten_ideas_r02'
cfg,cohort,_=protocol(ROOT/'data/manifests/mindset_v2_development100.json',ROOT/'outputs/acl/mindset_joint_v2_27058/calibration/calibration.json')
cases={(d,c.id):c for d,cl,group in groups_for(cohort,cfg) for c in group}
rows={}; structures={}; queries={}
for shard in sorted(root.glob('shard-*')):
    rows.update(unique_rows(shard/'retrieval.jsonl'))
    for path in (shard/'structure').glob('*.json'): structures[path.name]=read_json(path)
    for path in (shard/'query_structure').glob('*.json'): queries[path.name]=read_json(path)
changed=[]
for row in rows.values():
    baseline=rows.get(f'{row["dataset"]}/joint/{row["question_id"]}')
    if not baseline or row['method']=='joint' or row['evidence']==baseline['evidence']: continue
    case=cases[(row['dataset'],row['question_id'])]
    delta={k:row['metrics'][k]-baseline['metrics'][k] for k in row['metrics']}
    item={'dataset':row['dataset'],'method':row['method'],'question_id':row['question_id'],
        'question':case.question,'reference_evaluator_only':case.answer,'gold_evaluator_only':case.gold,
        'metric_deltas':delta,'baseline_source_ids':[s['event_id'] for s in baseline['selected']],
        'new_source_ids':[s['event_id'] for s in row['selected']],'query_certificate':row['idea_certificate']}
    changed.append(item)
    if any(delta[k] for k in ('recall@8','mrr@8','ndcg@8')): print('RANKING_CHANGE',item,flush=True)
profiles=[]
for dataset in ('locomo','longmemeval'):
    for arm in ARMS:
        subset=[(name,r) for name,r in structures.items() if name.startswith(dataset+'-') and name.endswith('-'+arm+'.json')]
        if not subset: continue
        item={'dataset':dataset,'method':arm,'built_histories':len(subset),'episodes':sum(r['episodes'] for name,r in subset),
            'controller_write_seconds':sum(r['controller_write_seconds'] for name,r in subset),'operation_counts':Counter(),
            'bound_queries':0,'supported_episodes':0,'window_reconsiderations':0,'window_branches':0,'spectral_reassignments':0,
            'query_certificate_counts':Counter()}
        for name,r in subset:
            t=r['telemetry']; item['operation_counts'].update(t.get('operation_counts',{}))
            item['supported_episodes']+=t.get('idea_supported_episodes',0)
            rw=t.get('reversible_window',{}); item['window_reconsiderations']+=rw.get('reconsidered_transition_count',0)
            item['window_branches']+=rw.get('expanded_branches',0)
            item['spectral_reassignments']+=t.get('spectral_summary',{}).get('graph_partition_reassignments',0)
            post=queries.get(name,{}).get('telemetry_after_queries',{})
            item['bound_queries']+=post.get('idea_bound_queries',0)
        for row in rows.values():
            if row['dataset']!=dataset or row['method']!=arm: continue
            cert=row['idea_certificate']
            item['query_certificate_counts'][cert.get('compile_reason','not_applicable')]+=1
            if cert.get('triggered'): item['query_certificate_counts']['repair_triggered']+=1
            if cert.get('selected_repair'): item['query_certificate_counts']['repair_accepted']+=1
            if cert.get('witness_ids'): item['query_certificate_counts']['compiler_witnesses']+=1
            if cert.get('anti_support_ids'): item['query_certificate_counts']['anti_memory_active']+=1
        profiles.append(item); print('MECHANISM',item,flush=True)
write_json(root/'mechanism_snapshot.json',{'completed_cells':len(rows),'changed_cases':changed,'profiles':profiles,
    'complete_panel':len(rows)==1100,'timing_scope':'single-pass cached quality screening; NOT paper efficiency'})
