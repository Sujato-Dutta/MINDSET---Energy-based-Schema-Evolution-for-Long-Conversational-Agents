"""DGX-only latency stage attribution and static-index reuse diagnostic."""
from pathlib import Path
from collections import defaultdict
from time import perf_counter
import argparse
import os
import sys
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input',required=True); p.add_argument('--output',required=True)
    args=p.parse_args()
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only, not laptop or login node')
    sys.path.insert(0,str(ROOT/'src'))
    from mindset.acl.joint_iteration import protocol,groups_for,build,get_hits
    from mindset.acl.pipeline import local_stack,record_environment,write_csv
    from mindset.acl.io import read_json,write_json,unique_rows,digest_file
    from mindset.acl.evidence import EvidenceCache
    from mindset.acl.clients import load_tokenizer
    source=Path(args.input); out=Path(args.output)
    if out.exists() and any(out.iterdir()): raise ValueError('Fresh diagnostics directory required')
    out.mkdir(parents=True); record_environment(out)
    calibration=source/'calibration/calibration.json'; manifest=ROOT/'data/manifests/mindset_v2_development100.json'
    cfg,cohort,artifact=protocol(manifest,calibration)
    measures=[r for path in sorted((source/'latency_cpu').glob('rep-*/queries.jsonl')) for r in unique_rows(path).values()]
    by_case=defaultdict(list)
    for r in measures: by_case[(r['dataset'],r['method'],r['question_id'])].append(r['retrieval_seconds'])
    tail=sorted(by_case,key=lambda key:(-float(np.median(by_case[key])),key))[:12]
    quality=list(unique_rows(source/'quality/retrieval.jsonl').values())
    by_quality={(r['dataset'],r['method'],r['question_id']):r for r in quality}
    tails=[]
    for key in tail:
        row=by_quality[key]
        tails.append({'dataset':key[0],'method':key[1],'question_id':key[2],
            'original_retrieval_median_ms':float(np.median(by_case[key]))*1000,
            'evidence_tokens':row['evidence_tokens'],'selected_sources':len(row['selected']),
            'selected_token_spans':sum(s['token_end']-s['token_start'] for s in row['selected'])})
    write_csv(out/'tail_queries.csv',tails)
    # Check the actual future server's config loader now, without GPU inference.
    gpu_py=ROOT/'.venv-gpu-prototype/bin/python'
    env=dict(os.environ,HF_HOME=str(ROOT/'.cache/gpu_prototype/huggingface'),CUDA_VISIBLE_DEVICES='')
    if gpu_py.exists():
        spec=read_json(ROOT/'configs/checkpoints.lock.json')['qwen35_9b']
        code='from vllm.transformers_utils.config import get_config; c=get_config('+repr(spec['checkpoint'])+',trust_remote_code=False,revision='+repr(spec['revision'])+'); print(type(c).__name__,c.model_type)'
        result=subprocess.run([str(gpu_py),'-c',code],env=env,capture_output=True,text=True,timeout=120)
        write_json(out/'server_config_probe.json',{'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
            'scope':'CPU config compatibility probe; not GPU inference or proof of server startup'})
    encoder,reranker=local_stack(cfg,cache=False)
    tokenizer=load_tokenizer(read_json(ROOT/'configs/checkpoints.lock.json')[cfg['evidence_tokenizer']]); report=[]
    for dataset,cluster,queries in groups_for(cohort,cfg):
        selected=[q for q in queries if any(d==dataset and i==q.id for d,m,i in tail)]
        if not selected: continue
        methods=sorted({m for d,m,i in tail if d==dataset and i in {q.id for q in selected}})
        for method in methods:
            m=build(method,queries[0].events,cfg,artifact['selected_parameters'],encoder,reranker)
            get_hits(method,m,'Conversation facts and events',cfg)
            for case in selected:
                if (dataset,method,case.id) not in tail: continue
                original={e.event_id:e for e in case.events}
                for repetition in range(3):
                    index=EvidenceCache(case.events,encoder,tokenizer,cfg)
                    start=perf_counter(); hits=get_hits(method,m,case.question,cfg); retrieved=perf_counter()
                    sources,packet=index.packet(hits,case.question); cold_end=perf_counter()
                    warm_start=perf_counter(); warm_sources,warm_packet=index.packet(hits,case.question); warm_end=perf_counter()
                    if warm_packet!=packet or warm_sources!=sources: raise ValueError('Static index reuse changed evidence')
                    report.append({'dataset':dataset,'method':method,'question_id':case.id,'repetition':repetition,
                        'native_retrieval_ms':(retrieved-start)*1000,'cold_packet_ms':(cold_end-retrieved)*1000,
                        'warm_static_packet_ms':(warm_end-warm_start)*1000,
                        'cold_total_ms':(cold_end-start)*1000,
                        'native_plus_warm_packet_ms':(retrieved-start+warm_end-warm_start)*1000,
                        'packet_tokenizations':index.tokenization_calls,'packet_chunk_embedding_calls':index.chunk_embedding_calls,
                        'selected_raw_characters':sum(len(original[s['event_id']].text) for s in sources),
                        'long_selected_events':sum(len(index.event_chunks(original[s['event_id']]))>1 for s in sources),
                        'identical_packet':True,'scope':'selected-tail_diagnostic_not_full-panel_efficiency'})
                write_csv(out/'stage_profiles.csv',report)
                print(f'Profiled tail: {dataset}/{method}/{case.id}',flush=True)
    write_json(out/'audit.json',{'manifest_sha256':digest_file(manifest),'source_input':str(source),'worker_sha256':digest_file(Path(__file__)),
        'selection':'12 slowest median query-method pairs in completed development100 timing artifacts',
        'static_cache_policy':'warm source tokenization/chunk embeddings only; no query-score or answer caching',
        'important':'warming cost is measured by cold_packet_ms, not free ingestion; timings are diagnostics, not paper percentiles',
        'answer_calls':0,'full_runs_authorized':False})

if __name__=='__main__': main()
