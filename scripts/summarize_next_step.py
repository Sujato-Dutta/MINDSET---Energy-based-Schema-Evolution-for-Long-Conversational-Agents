"""Summarize completed causal and latency diagnostics on allocated DGX CPU."""
from pathlib import Path
import argparse
import os
import sys
import csv
from collections import defaultdict
from statistics import median
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--causal',required=True); p.add_argument('--latency',required=True); p.add_argument('--output',required=True)
    args=p.parse_args()
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    sys.path.insert(0,str(ROOT/'src'))
    from mindset.acl.io import read_json,write_json,unique_rows,digest_file
    from mindset.acl.pipeline import write_csv,record_environment
    causal=Path(args.causal); latency=Path(args.latency); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    if read_json(causal/'status.json')['status']!='complete_development': raise ValueError('Causal panel incomplete')
    records=list(unique_rows(causal/'evidence.jsonl').values())
    if len(records)!=900 or any(r['candidate_count']!=48 or len(r['selected'])>8 for r in records):
        raise ValueError('Question count, 48-candidate or top-eight budget mismatch')
    with (latency/'stage_profiles.csv').open() as f: profiles=list(csv.DictReader(f))
    grouped=defaultdict(list)
    for r in profiles: grouped[(r['dataset'],r['method'],r['question_id'])].append(r)
    stages=[]
    for (dataset,method,id),rows in sorted(grouped.items()):
        if len(rows)!=3: raise ValueError('Tail repetitions incomplete')
        native=median(float(r['native_retrieval_ms']) for r in rows)
        cold=median(float(r['cold_packet_ms']) for r in rows)
        warm=median(float(r['warm_static_packet_ms']) for r in rows)
        stages.append({'dataset':dataset,'method':method,'question_id':id,
            'native_retrieval_median_ms':native,'cold_packet_median_ms':cold,'warm_static_packet_median_ms':warm,
            'cold_packet_fraction_of_median_component_sum':cold/(native+cold) if native+cold else 0,
            'static_packet_reuse_speedup':cold/warm if warm else None,
            'all_packets_identical':all(r['identical_packet']=='True' for r in rows),
            'scope':'12 selected tail cases, not full-panel or end-to-end metrics'})
    write_csv(out/'latency_attribution.csv',stages)
    with (causal/'causal_opportunity.csv').open() as f: comparisons=list(csv.DictReader(f))
    structure=defaultdict(list)
    for path in (causal/'structure').glob('*.json'):
        dataset='longmemeval' if path.name.startswith('longmemeval-') else 'locomo'
        arm=next(a for a in read_json(causal/'audit.json')['arms'] if path.name.endswith('-'+a+'.json'))
        data=read_json(path); structure[(dataset,arm)].append(data)
    structural=[]
    for (dataset,arm),rows in sorted(structure.items()):
        structural.append({'dataset':dataset,'arm':arm,'histories':len(rows),'schemas':sum(r['schema_count'] for r in rows),
            'historical_schemas':sum(r['historical_schemas'] for r in rows),
            'grounded_episodes':sum(r['invariants']['grounded_episode_count'] for r in rows),
            'native_state_error_accuracy':None,'state_error_labels':'unavailable_not_fabricated'})
    write_csv(out/'structure.csv',structural); record_environment(out)
    write_json(out/'summary.json',{'causal_cells':900,'causal_opportunity':comparisons,
        'latency_tail_cases':len(stages),'server_config_probe':read_json(latency/'server_config_probe.json'),
        'next_decision':'Inspect complete-prompt changes and answer deltas before claiming controller necessity; attribute latency before optimization',
        'full_runs_authorized':False,'answer_quality_available':False,
        'worker_sha256':digest_file(Path(__file__))})
    print('Completed 900-cell causal audit and',len(stages),'latency tail profiles',flush=True)

if __name__=='__main__': main()
