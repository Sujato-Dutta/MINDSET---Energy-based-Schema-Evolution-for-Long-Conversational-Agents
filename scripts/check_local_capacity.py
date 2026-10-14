"""DGX-only pinned BF16 server smoke; never a benchmark/efficiency result."""
from pathlib import Path
from dataclasses import asdict
import argparse
import os
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from mindset.acl.pipeline import config,server_client,request_json,record_environment
from mindset.acl.clients import load_tokenizer,messages,prompt_tokens
from mindset.acl.io import read_json,write_json,digest_file,append_row


def server_identifier(info,checkpoint,cache_home):
    # vLLM 0.17 resolves offline repository IDs into exact snapshot paths. That
    # API alias is allowed only for this locked repository AND revision.
    repo=checkpoint['checkpoint']; revision=checkpoint['revision']
    snapshot=Path(cache_home)/'hub'/('models--'+repo.replace('/','--'))/'snapshots'/revision
    matches=[r['id'] for r in info['data'] if r['id'] in {repo,str(snapshot)}]
    if len(matches)!=1: raise ValueError('Server model/revision alias mismatch')
    return matches[0]


def main(a):
    out=Path(a.output); cfg=config(); lock=read_json(ROOT/'configs/checkpoints.lock.json'); model=cfg['primary_generator']
    if model!='qwen35_9b': raise ValueError('Capacity smoke is specifically pinned primary9b')
    info=request_json(os.environ['MINDSET_LOCAL_BASE_URL'].rstrip('/')+'/models')
    served=server_identifier(info,lock[model],os.environ['HF_HOME'])
    tokenizer=load_tokenizer(lock[model]); client=server_client(model,cfg)
    client.model=served
    evidence='[1] 2025-01-01 | Speaker: user\nMy car is red.\n\n[2] 2025-02-01 | Speaker: user\nI changed my car color to blue.'
    fixtures=[('short','What color is my car now?',evidence,'2025-02-02'),
        ('historical','What color was my car in January?',evidence,'2025-02-02'),
        ('long','What color is my car now?',('[2024-01-01] | Speaker: user\nI enjoy gardening and walking.\n'*500)+evidence,'2025-02-02')]
    record_environment(out)
    write_json(out/'audit.json',{'model':model,'checkpoint':lock[model],'checkpoint_lock_sha256':digest_file(ROOT/'configs/checkpoints.lock.json'),
        'worker_sha256':digest_file(__file__),'server_info':info,'served_api_identifier':served,
        'scope':'synthetic serving-capacity smoke; NOT native QA or efficiency',
        'context_cap':a.context_cap,'dtype':'bfloat16','offload_mode':a.offload_mode,
        'cpu_weight_offload_gib':4 if a.offload_mode=='uva' else None,
        'prefetch_config':{'group_size':8,'num_in_group':2,'prefetch_step':1} if a.offload_mode=='prefetch' else None,
        'quantization':None,'full_runs_authorized':False},immutable=True)
    for name,question,packet,date in fixtures:
        prompt=messages(question,packet,date); count=prompt_tokens(tokenizer,prompt,thinking=False)
        if count+cfg['generation']['max_output_tokens']>a.context_cap: raise ValueError('Smoke prompt exceeds cap; no truncation')
        result=client.complete(prompt,max_tokens=cfg['generation']['max_output_tokens'],seed=cfg['seed'])
        if result.cache_hit: raise ValueError('Capacity smoke cannot use cached answers')
        if result.input_tokens!=count:
            raise ValueError(('Client/server prompt-token counts differ',name,count,result.input_tokens))
        append_row(out/'smoke_answers.jsonl',{'id':name,'question':question,'request_prompt_tokens':count,**asdict(result)})
        print(f'{name}: {count} input tokens; {result.output_tokens} output; {result.text}',flush=True)
    write_json(out/'status.json',{'status':'passed_three_finite_completions','synthetic_calls':3,'benchmark_qa_calls':0,
        'judge_calls':0,'scope':'model serving/fit only; not correctness/generalization/efficiency','full_runs_authorized':False},immutable=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True); p.add_argument('--context-cap',type=int,default=16384)
    p.add_argument('--offload-mode',choices=('uva','prefetch'),default='prefetch')
    a=p.parse_args()
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    main(a)
