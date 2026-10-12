"""Same-affinity CPU thread experiment; no answer/parameter tuning."""
from pathlib import Path
from time import perf_counter
import argparse
import os
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--calibration',required=True); p.add_argument('--output',required=True)
    args=p.parse_args()
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    if int(os.environ.get('SLURM_CPUS_PER_TASK','0'))<16: p.error('Same-affinity probe requires allocation of 16 CPUs')
    sys.path.insert(0,str(ROOT/'src'))
    import torch
    from mindset.acl.joint_iteration import protocol,groups_for,build,get_hits
    from mindset.acl.pipeline import local_stack,record_environment,write_csv
    from mindset.acl.io import read_json,write_json,digest_file
    from mindset.acl.evidence import EvidenceCache
    from mindset.acl.clients import load_tokenizer
    manifest=ROOT/'data/manifests/mindset_v2_development100.json'
    cfg,cohort,artifact=protocol(manifest,args.calibration); out=Path(args.output)
    if out.exists() and any(out.iterdir()): raise ValueError('Fresh thread-probe directory required')
    out.mkdir(parents=True); record_environment(out)
    # Existing development IDs only; selected to reproduce observed tails,
    # not because of their answers, retrieval scores or method gains.
    targets={'locomo':{'conv-43:q0057','conv-49:q0048'},'longmemeval':{'352ab8bd'}}
    encoder,reranker=local_stack(cfg,cache=False)
    tokenizer=load_tokenizer(read_json(ROOT/'configs/checkpoints.lock.json')[cfg['evidence_tokenizer']]); rows=[]
    references={}; found=set()
    for dataset,cluster,queries in groups_for(cohort,cfg):
        chosen=[q for q in queries if q.id in targets[dataset]]
        if not chosen: continue
        for method in ('joint','compact'):
            memory=build(method,queries[0].events,cfg,artifact['selected_parameters'],encoder,reranker)
            for count in (16,8,4,1):
                torch.set_num_threads(count)
                get_hits(method,memory,'Conversation facts and events',cfg)
                for case in chosen:
                    found.add((dataset,case.id)); key=(dataset,method,case.id)
                    for rep in range(3):
                        start=perf_counter(); hits=get_hits(method,memory,case.question,cfg); middle=perf_counter()
                        selected,packet=EvidenceCache(case.events,encoder,tokenizer,cfg).packet(hits,case.question); end=perf_counter()
                        signature=([r['event_id'] for r in selected],packet)
                        references.setdefault(key,signature)
                        rows.append({'dataset':dataset,'method':method,'question_id':case.id,'torch_threads':count,'repetition':rep,
                            'native_retrieval_ms':(middle-start)*1000,'packet_ms':(end-middle)*1000,'retrieval_total_ms':(end-start)*1000,
                            'same_evidence_as_reference':signature==references[key],
                            'blas_threads':'fixed by Slurm environment, not changed with torch thread count',
                            'scope':'selected-case CPU diagnostic, not full-panel or end-to-end percentiles'})
                    write_csv(out/'thread_profiles.csv',rows)
            del memory
    if found!={(d,i) for d,ids in targets.items() for i in ids}: raise ValueError('Selected diagnostic IDs are not all in frozen cohort')
    summary=[]
    for count in (16,8,4,1):
        subset=[r for r in rows if r['torch_threads']==count]
        summary.append({'torch_threads':count,'observations':len(subset),
            'retrieval_p50_ms':float(np.median([r['retrieval_total_ms'] for r in subset])),
            'retrieval_p95_ms':float(np.quantile([r['retrieval_total_ms'] for r in subset],.95)),
            'all_evidence_identical':all(r['same_evidence_as_reference'] for r in subset)})
    write_csv(out/'summary.csv',summary)
    write_json(out/'audit.json',{'manifest_sha256':digest_file(manifest),'worker_sha256':digest_file(Path(__file__)),
        'selected_ids':{d:sorted(ids) for d,ids in targets.items()},'cpu_affinity':'same Slurm process/allocation for all settings',
        'caches':'no query-score/embedding/packet cache','answer_calls':0,'hyperparameter_changes':False,
        'scope':'operational thread diagnostic; selected cases and fixed descending order cannot establish full-panel gains'})

if __name__=='__main__': main()
