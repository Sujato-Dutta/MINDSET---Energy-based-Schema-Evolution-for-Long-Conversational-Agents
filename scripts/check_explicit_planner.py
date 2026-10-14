"""Same synthetic fixtures with an explicit output-field contract, DGX only."""
import importlib.util
import json
import os
from dataclasses import asdict
from mindset.acl.pipeline import ROOT,config,server_client
from mindset.acl.io import write_json,digest_file
from mindset.acl.planner_contract import explicit_messages
from mindset.methods.query_state_agent import planner_schema


def run():
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('DGX Slurm only')
    cfg=config(); client=server_client(cfg['primary_generator'],cfg); client.allow_incomplete=True
    spec=importlib.util.spec_from_file_location('planner_fixture_builder',ROOT/'scripts/check_planner_semantics.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    out=ROOT/'outputs/acl'/f'explicit_planner_{os.environ["SLURM_JOB_ID"]}'; out.mkdir(parents=True,exist_ok=True)
    results=[]
    for name,question,rows,expected in mod.fixtures():
        client.options['response_format']={'type':'json_schema','json_schema':{
            'name':'memory_access_plan','strict':True,'schema':planner_schema(rows)}}
        prompt=explicit_messages(question,rows)
        completion=client.complete(prompt,max_tokens=256,seed=cfg['seed'])
        try: plan=json.loads(completion.text)
        except json.JSONDecodeError: plan=None
        results.append({'fixture':name,'expected_operator_synthetic_only':expected,'messages':prompt,
            'plan':plan,'completion':asdict(completion),'correct_operator':bool(isinstance(plan,dict) and plan.get('operation')==expected)})
        write_json(out/'results.json',{'scope':'synthetic prompt-contract regression; NOT novelty or natural QA',
            'worker_sha256':digest_file(__file__),'contract_sha256':digest_file(ROOT/'src/mindset/acl/planner_contract.py'),
            'results':results})
        print('EXPLICIT_CONTRACT',name,'expected',expected,'actual',plan,flush=True)


if __name__=='__main__': run()
