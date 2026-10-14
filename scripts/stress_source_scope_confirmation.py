"""Frozen NLI comparison on the same lexically disjoint synthetic checks."""
import os
from mindset.acl.pipeline import ROOT,record_environment
from mindset.acl.local_semantics import SourceNLI
from mindset.acl.io import write_json,digest_file
from mindset.utils.hashing import stable_hash
from source_scope_confirmation_fixtures import confirmation_fixtures
from stress_source_semantics import summarize

if __name__=='__main__':
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('DGX Slurm only')
    cases=confirmation_fixtures(); out=ROOT/'outputs/acl'/f'source_scope_nli_confirmation_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True,exist_ok=True); record_environment(out)
    nli=SourceNLI(ROOT,out/'pairs.sqlite')
    labels=nli.predict([p for r in cases for p in ((r['left'],r['right']),(r['right'],r['left']))])
    rows=[{**r,'forward_label':labels[2*i],'reverse_label':labels[2*i+1],
           'proxy_contradiction':labels[2*i]==labels[2*i+1]=='contradiction'} for i,r in enumerate(cases)]
    report={'fixture_sha256':stable_hash(cases),'worker_sha256':digest_file(__file__),
        'model_identity':nli.identity,'summary':summarize(rows),'cases':rows,
        'scope':'same-family lexical confirmation only, NOT OOD or benchmark accuracy; no retuning',
        'full_runs_authorized':False}
    write_json(out/'analysis.json',report); print('CONFIRMATION',report['summary'],flush=True)
