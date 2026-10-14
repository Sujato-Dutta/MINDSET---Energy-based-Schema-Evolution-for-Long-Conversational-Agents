"""Matched prompt decoding diagnosis on four synthetic fixtures, DGX only."""
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
from mindset.acl.pipeline import ROOT,config,server_client,request_json
from mindset.acl.io import write_json,digest_file
from mindset.methods.query_state_agent import planner_messages,planner_schema


def run():
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('DGX Slurm only')
    cfg=config(); client=server_client(cfg['primary_generator'],cfg); client.allow_incomplete=True
    info=request_json(os.environ['MINDSET_LOCAL_BASE_URL'].rstrip('/')+'/models')
    checkpoint=json.loads((ROOT/'configs/checkpoints.lock.json').read_text())[cfg['primary_generator']]
    if {r['id'] for r in info['data']}!={checkpoint['checkpoint']}: raise ValueError('Wrong model')
    spec=importlib.util.spec_from_file_location('planner_fixture_builder',ROOT/'scripts/check_planner_semantics.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    results=[]
    out=ROOT/'outputs/acl'/f'planner_decoding_{os.environ["SLURM_JOB_ID"]}'; out.mkdir(parents=True,exist_ok=True)
    for name,question,inventory,expected in mod.fixtures():
        prompt=planner_messages(question,inventory)
        exact=planner_schema(inventory)
        loose={**exact,'properties':{k:{'type':'string'} for k in exact['properties']}}
        formats={'free_json':None,'json_object':{'type':'json_object'},
            'exact_schema':{'type':'json_schema','json_schema':{'name':'memory_access_plan','strict':True,'schema':exact}},
            'loose_schema':{'type':'json_schema','json_schema':{'name':'memory_access_plan','strict':True,'schema':loose}}}
        for variant,response_format in formats.items():
            client.options.pop('response_format',None)
            if response_format is not None: client.options['response_format']=response_format
            completion=client.complete(prompt,max_tokens=256,seed=cfg['seed'])
            try: parsed=json.loads(completion.text)
            except json.JSONDecodeError: parsed=None
            result={'fixture':name,'decoder':variant,'expected_operator_synthetic_only':expected,
                'plan':parsed,'correct_operator':bool(isinstance(parsed,dict) and parsed.get('operation')==expected),
                'completion':asdict(completion)}
            results.append(result)
            write_json(out/'results.json',{'scope':'synthetic software diagnosis, not natural QA or method innovation',
                'checkpoint':checkpoint,'worker_sha256':digest_file(__file__),'results':results})
            print('DECODE',name,variant,'expected',expected,'actual',parsed,flush=True)
    print('COMPLETE',str(out),flush=True)


if __name__=='__main__': run()
