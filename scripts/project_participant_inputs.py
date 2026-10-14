"""DGX source-only participant projection on the original development100."""
from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import importlib.util
import json
import os
ROOT=Path(__file__).resolve().parents[1]


def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'src/mindset/memory'/file)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


broad=module('broad','broad_frames.py'); contracts=module('contracts','query_contracts.py')
participants=module('participants','participant_contracts.py')


def query(nlp,text,speakers):
    attribution=participants.canonical_owner(nlp(text),text,speakers)
    canonical=attribution['canonical_text']; doc=nlp(canonical)
    q=contracts.project_query(doc,canonical,broad.project_report_query(doc,canonical))
    q['contract']['owner']=attribution['owner']; q['owner_attribution']=attribution
    if attribution['owner'] is None: q.update(mode='ordinary',probes=[],self_reference=False)
    return q


def functional_checks(nlp):
    fixtures=[('How many workers does Rae lead now?',['Rae','Avery'],'rae'),
        ("What is Rae's current supplier?",['Rae','Avery'],'rae'),
        ('Where does Rae Kim work now?',['Rae Kim','Avery'],'rae kim'),
        ('How many workers do I lead?',['user','assistant'],'user'),
        ('How many workers do I lead?',['Rae','Avery'],None),
        ('Where do Rae and Avery work?',['Rae','Avery'],None),
        ("What does Rae's friend own?",['Rae','Avery'],None),
        ('Where does Morgan work?',['Rae','Avery'],None),
        ('Where does R work?',['Rae','Avery'],None),
        ('Where do I and Rae work?',['user','Rae'],None)]
    result=[]
    for text,speakers,owner in fixtures:
        q=query(nlp,text,speakers); assert q['contract']['owner']==owner,(text,q)
        for rewrite in q['owner_attribution']['rewrites']:
            assert text[slice(*rewrite['source_span'])]==rewrite['source_text']
        result.append({'question':text,'speakers':speakers,'expected_owner':owner,'passed':True})
    for text,source,speaker in [('How many workers does Rae lead now?','I lead three workers.','Rae'),
             ("What is Rae's current supplier?",'My supplier is Cedar.','Rae')]:
        q=query(nlp,text,['Rae','Avery']); doc=nlp(source)
        frames=[contracts.report_contract(f,nlp(f['value']),source) for f in broad.project_reports(doc,source,speaker)]
        assert any(contracts.compatible(q['contract'],f,source) and
            any(p['unit_terms']==f['unit_terms'] and p['probe']==f['probe'] for p in q['probes']) for f in frames),(text,q,frames)
    return {'owner_checks':result,'query_source_role_checks':2}


def main(a):
    import spacy
    nlp=spacy.load('en_core_web_sm')
    if spacy.__version__!='3.8.7' or nlp.meta['version']!='3.8.0': raise ValueError('Parser changed')
    out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    save=lambda name,value:(out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    save('functional_tests.json',functional_checks(nlp))
    with Path(a.input).open(encoding='utf8') as f: rows=[json.loads(line) for line in f if line.strip()]
    speakers=defaultdict(set)
    for row in rows:
        if row['kind']=='episode': speakers[(row['dataset'],row['cluster'])].add(row['speaker'])
    total=count=0
    with (out/'reports.jsonl').open('w',encoding='utf8') as f:
        for doc,row in nlp.pipe(((r['text'],r) for r in rows),as_tuples=True,batch_size=64,n_process=1):
            if row['kind']=='episode':
                raw=broad.project_reports(doc,row['text'],row['speaker'])+contracts.frequency_frames(doc,row['text'],row['speaker'])
                frames=[contracts.report_contract(r,nlp(r['value']),row['text']) for r in raw]
                frames=[r for r in frames if r['family']!='object' or r['contract']['argument_head']]
                extra={'reports':frames}; count+=len(frames)
            else: extra={'report_query':query(nlp,row['text'],speakers[(row['dataset'],row['cluster'])])}
            f.write(json.dumps({**row,**extra},sort_keys=True,ensure_ascii=False)+'\n'); total+=1
            if total%1000==0: f.flush(); print(f'{total}/{len(rows)} participant inputs; {count} reports',flush=True)
    save('audit.json',{'input_sha256':sha(a.input),'worker_sha256':sha(__file__),
        'module_sha256':{file:sha(ROOT/'src/mindset/memory'/file) for file in
                        ('participant_contracts.py','query_contracts.py','broad_frames.py','report_frames.py')},
        'parser_bytes_sha256':hashlib.sha256(nlp.to_bytes()).hexdigest(),'spacy':spacy.__version__,'parser_version':nlp.meta['version'],
        'scope':'exact metadata speaker in direct argument only; no aliases, gold, answers or coefficient fit',
        'new_fitted_scalars':0,'qa_calls':0})
    save('status.json',{'status':'complete_projection','rows':total,'expected_rows':len(rows),'reports':count,'qa_calls':0})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input',required=True); p.add_argument('--output',required=True)
    a=p.parse_args()
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    main(a)
