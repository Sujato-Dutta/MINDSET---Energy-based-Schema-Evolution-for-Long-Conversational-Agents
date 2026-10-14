"""Four synthetic planner sanity cases, not benchmark performance evidence."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from mindset.acl.pipeline import ROOT,config,server_client
from mindset.acl.io import write_json,digest_file
from mindset.methods.query_state_agent import planner_messages,planner_schema


def fixtures():
    def row(sid,value,status='active',start='2025-01-01',end=None,pred=None,children=None,slot='location',contexts=None):
        return {'schema_id':sid,'entity':'sam','attributes':{slot:[value]},'contexts':contexts or [],
            'status':status,'valid_from':start,'valid_until':end,'predecessor':pred,'successors':children or [],
            'branches':[],'medoid_excerpt':f'I live in {value}.' if slot=='location' else f'I prefer {value}.'}
    return [
        ('current','Where does Sam live now?',[row('s1','Paris')],'CURRENT'),
        ('previous','Where did Sam live before?',[
            row('s1','Paris','historical',end='2025-02-01',children=['s2']),
            row('s2','Rome',start='2025-02-01',pred='s1')],'PREDECESSOR'),
        ('context','What does Sam prefer at work?',[
            row('s1','tea',slot='preference',contexts=['at home']),
            row('s2','coffee',slot='preference',contexts=['at work'])],'CONTEXT'),
        ('unsupported','What is Sam\'s blood type?',[],'SEARCH')]


def run(output):
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('DGX Slurm only')
    cfg=config(); client=server_client(cfg['primary_generator'],cfg); client.allow_incomplete=True
    results=[]
    for name,question,rows,expected in fixtures():
        client.options['response_format']={'type':'json_schema','json_schema':{
            'name':'memory_access_plan','strict':True,'schema':planner_schema(rows)}}
        completion=client.complete(planner_messages(question,rows),max_tokens=256,seed=cfg['seed'])
        try: plan=json.loads(completion.text)
        except json.JSONDecodeError: plan=None
        results.append({'fixture':name,'question':question,'inventory':rows,'expected_operator_synthetic_only':expected,
            'plan':plan,'completion':asdict(completion),'correct_operator':bool(plan and plan.get('operation')==expected)})
        print(f'PLANNER_SANITY {name} expected={expected} actual={plan}',flush=True)
    write_json(Path(output)/'planner_sanity.json',{'scope':'synthetic software diagnostic, not natural QA or novelty',
        'worker_sha256':digest_file(__file__),'rows':results,'requests':len(results),
        'input_tokens':sum(r['completion']['input_tokens'] for r in results),
        'output_tokens':sum(r['completion']['output_tokens'] for r in results)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True)
    run(p.parse_args().output)
