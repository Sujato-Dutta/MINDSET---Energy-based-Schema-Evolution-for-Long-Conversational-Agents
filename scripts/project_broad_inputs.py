"""Pinned spaCy source/query projection, Slurm only; no answers or gold inputs."""
from pathlib import Path
import argparse
import importlib.util
import hashlib
import json
import os
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('broad_projection',ROOT/'src/mindset/memory/broad_frames.py')
projection=importlib.util.module_from_spec(spec); spec.loader.exec_module(projection)


def main(a):
    import spacy
    nlp=spacy.load('en_core_web_sm')
    if spacy.__version__!='3.8.7' or nlp.meta['version']!='3.8.0': raise ValueError('Parser changed')
    out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    def save(name,value): (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    fixtures=[('My supplier is Cedar.','property'),('My supplier was Birch.','property'),
              ('I changed my supplier to Cedar.','property'),('I work at Cedar.','relation'),
              ('I worked at Birch.','relation'),('I prefer tea.','object'),
              ('My supplier might be Cedar.',None),('"My supplier is Cedar."',None),
              ('Is my supplier Cedar?',None),('My supplier is not Cedar.',None),
              ('If my supplier is Cedar, I will leave.',None),("My friend's supplier is Cedar.",None),
              ('I think my supplier is Cedar.',None)]
    tests=[]
    for text,family in fixtures:
        frames=projection.project_reports(nlp(text),text,'user')
        if family is None: assert not frames,(text,frames)
        else: assert family in {f['family'] for f in frames},(text,frames)
        for f in frames: assert text[slice(*f['value_span'])]==f['value']
        tests.append({'text':text,'expected_family':family,'reports':frames,'passed':True})
    for text,mode in [('What is my current supplier?','state_current'),('Where do I work now?','state_current'),
                      ('Where did I work before?','historical'),('When did I work at Cedar?','event')]:
        q=projection.project_report_query(nlp(text),text); assert q['mode']==mode,(text,q)
    save('functional_tests.json',{'source_checks':tests,'query_checks':4})
    save('audit.json',{'input_sha256':sha(a.input),'worker_sha256':sha(__file__),
        'projection_sha256':sha(ROOT/'src/mindset/memory/broad_frames.py'),
        'numeric_projection_sha256':sha(ROOT/'src/mindset/memory/report_frames.py'),
        'parser_bytes_sha256':hashlib.sha256(nlp.to_bytes()).hexdigest(),'parser':nlp.meta,'spacy':spacy.__version__,
        'scope':'input-only bounded report projection; not current-state truth or semantic completeness','qa_calls':0})
    with Path(a.input).open(encoding='utf8') as stream: rows=[json.loads(line) for line in stream if line.strip()]
    total=reports=0
    with (out/'reports.jsonl').open('w',encoding='utf8') as stream:
        for doc,row in nlp.pipe(((r['text'],r) for r in rows),as_tuples=True,batch_size=64,n_process=1):
            if row['kind']=='episode': extra={'reports':projection.project_reports(doc,row['text'],row['speaker'])}; reports+=len(extra['reports'])
            else: extra={'report_query':projection.project_report_query(doc,row['text'])}
            stream.write(json.dumps({**row,**extra},ensure_ascii=False,sort_keys=True)+'\n'); stream.flush(); total+=1
            if total%1000==0: print(f'{total}/{len(rows)} broad inputs; {reports} reports',flush=True)
    save('status.json',{'status':'complete_projection','rows':total,'expected_rows':len(rows),'reports':reports,'qa_calls':0})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    main(a)
