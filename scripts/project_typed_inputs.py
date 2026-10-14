"""Refine existing source-only projections with pinned dependency contracts."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('contracts',ROOT/'src/mindset/memory/query_contracts.py')
contracts=importlib.util.module_from_spec(spec); spec.loader.exec_module(contracts)
spec=importlib.util.spec_from_file_location('broad',ROOT/'src/mindset/memory/broad_frames.py')
broad=importlib.util.module_from_spec(spec); spec.loader.exec_module(broad)


def functional_checks(nlp):
    fixtures=[('Do I have a spare wrench?', 'I have a spare wrench.',True),
              ('Do I have a spare wrench?', 'I have help finding a backup service.',False),
              ('How often do I see Dr Morgan?', 'I see Dr Morgan twice a week.',True),
              ('How often do I see Dr Morgan?', 'I see Dr Morgan.',False),
              ('How often do I see Dr Morgan?', 'I see Dr Avery twice a week.',False),
              ('How many workers do I lead now?', 'I lead three workers.',True),
              ('How many issues have I read?', 'I read a series.',False),
              ('What is my current supplier?', 'My supplier is Cedar.',True),
              ('What is my high score in Chess?', 'My high score in golf is 24.',False),
              ('Where do I work now?', 'I work at Cedar.',True)]
    results=[]
    for question,text,expected in fixtures:
        qdoc=nlp(question); q=contracts.project_query(qdoc,question,broad.project_report_query(qdoc,question))
        doc=nlp(text); raw=broad.project_reports(doc,text,'user')+contracts.frequency_frames(doc,text,'user')
        frames=[contracts.report_contract(f,nlp(f['value']),text) for f in raw]
        ok=any(contracts.compatible(q['contract'],f,text) and
               any(p['unit_terms']==f['unit_terms'] and p['probe']==f['probe'] for p in q['probes']) for f in frames)
        assert ok==expected,(question,text,q,frames)
        for f in frames: assert text[slice(*f['value_span'])]==f['value']
        results.append({'question':question,'source':text,'expected_binding':expected,'passed':True})
    for text in ('I might see Dr Morgan weekly.','I do not see Dr Morgan weekly.',
                 '"I see Dr Morgan weekly."','Do I see Dr Morgan weekly?',
                 'If I see Dr Morgan weekly, I will feel better.','I think I see Dr Morgan weekly.'):
        assert not contracts.frequency_frames(nlp(text),text,'user'),text
    text='What type of lens did I purchase most recently?'; doc=nlp(text)
    q=contracts.project_query(doc,text,broad.project_report_query(doc,text))
    assert q['mode']=='latest_event',(text,q)
    return {'pair_checks':results,'frequency_rejection_checks':6,'latest_event_check':True}


def main(a):
    import spacy
    nlp=spacy.load('en_core_web_sm')
    if spacy.__version__!='3.8.7' or nlp.meta['version']!='3.8.0': raise ValueError('Parser changed')
    out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    save=lambda name,value:(out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    save('functional_tests.json',functional_checks(nlp))
    with Path(a.input).open(encoding='utf8') as f: rows=[json.loads(line) for line in f if line.strip()]
    total=count=0
    with (out/'reports.jsonl').open('w',encoding='utf8') as f:
        for row in rows:
            if row['kind']=='episode':
                frames=[]
                for frame in row['reports']:
                    frame=contracts.report_contract(frame,nlp(frame['value']),row['text'])
                    if frame['family']=='object' and not frame['contract']['argument_head']: continue
                    frames.append(frame)
                # Parse whole text only when the explicit frequency pattern is
                # present; ordinary rows reuse the audited source-only frames.
                if contracts.FREQUENCY.search(row['text']):
                    for frame in contracts.frequency_frames(nlp(row['text']),row['text'],row['speaker']):
                        frames.append(contracts.report_contract(frame,nlp(frame['value']),row['text']))
                extra={'reports':frames}; count+=len(frames)
            else:
                extra={'report_query':contracts.project_query(nlp(row['text']),row['text'],row['report_query'])}
            f.write(json.dumps({**row,**extra},ensure_ascii=False,sort_keys=True)+'\n'); total+=1
            if total%1000==0: f.flush(); print(f'{total}/{len(rows)} typed inputs; {count} reports',flush=True)
    save('audit.json',{'input_sha256':sha(a.input),'worker_sha256':sha(__file__),
        'contract_sha256':sha(ROOT/'src/mindset/memory/query_contracts.py'),
        'parser_bytes_sha256':hashlib.sha256(nlp.to_bytes()).hexdigest(),'parser':nlp.meta,'spacy':spacy.__version__,
        'scope':'input-only type/argument constraints; not semantic completeness or world truth',
        'new_fitted_scalars':0,'qa_calls':0})
    save('status.json',{'status':'complete_projection','rows':total,'expected_rows':len(rows),'reports':count,'qa_calls':0})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input',required=True); p.add_argument('--output',required=True)
    a=p.parse_args()
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    main(a)
