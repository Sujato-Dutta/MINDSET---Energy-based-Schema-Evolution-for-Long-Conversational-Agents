"""Independent synthetic scope/time/modality checks for the fixed NLI proxy.

Not benchmark labels, tuning, world truth inference or a method performance
claim. The model sees literal sentence pairs only; fixture expectations are
used after classification. Run only through DGX Slurm.
"""
import os
from pathlib import Path
from collections import Counter
from mindset.acl.pipeline import ROOT,record_environment
from mindset.acl.local_semantics import SourceNLI
from mindset.acl.io import write_json,digest_file
from mindset.utils.hashing import stable_hash


def fixtures():
    groups={
        'same_scope_negation':[
            ('I own a car.','I do not own a car.',True),
            ('I am a teacher.','I am not a teacher.',True),
            ('I like jazz.','I do not like jazz.',True),
            ('I have exactly two cats right now.','I have exactly three cats right now.',True)],
        'coexisting_attributes':[
            ('I like jazz.','I like rock music.',False),
            ('I work as a teacher.','I volunteer at a community center.',False),
            ('My budget is 100 dollars.','My weight is 70 kilograms.',False),
            ('I teach English.','I do not teach French.',False)],
        'different_contexts':[
            ('I prefer tea at home.','I prefer coffee at work.',False),
            ('I run in the morning on Mondays.','I run in the evening on Tuesdays.',False),
            ('I live in Paris during summer.','I live in Rome during winter.',False),
            ('My budget for food is 100 dollars.','My budget for travel is 200 dollars.',False)],
        'different_times':[
            ('I lived in Paris in 2020.','I live in Rome in 2025.',False),
            ('I owned two cats in 2020.','I own three cats in 2025.',False),
            ('My budget was 100 dollars last year.','My budget is 200 dollars this year.',False),
            ('I was a teacher before 2020.','I am a doctor after 2020.',False)],
        'plans_not_current_state':[
            ('I plan to move to Paris.','I currently live in Rome.',False),
            ('I plan to contact Alex.','I plan to invite Ben and Casey.',False),
            ('I hope to become a doctor.','I am currently a student.',False),
            ('I may buy a car next year.','I do not own a car now.',False)],
        'different_holders_or_objects':[
            ('My brother lives in Paris.','My sister lives in Rome.',False),
            ('My old car is blue.','My new car is red.',False),
            ('I own a blue bicycle.','I own a red car.',False),
            ('My dog is two years old.','My cat is three years old.',False)]}
    return [{'id':f'{group}:{i}','family':group,'left':a,'right':b,'expected_contradiction':label}
            for group,pairs in groups.items() for i,(a,b,label) in enumerate(pairs)]


def summarize(records):
    groups={}
    for family in sorted({r['family'] for r in records}):
        subset=[r for r in records if r['family']==family]
        groups[family]={'cases':len(subset),'expected_contradictions':sum(r['expected_contradiction'] for r in subset),
            'bidirectional_contradictions':sum(r['proxy_contradiction'] for r in subset),
            'false_contradictions':sum(r['proxy_contradiction'] and not r['expected_contradiction'] for r in subset),
            'missed_contradictions':sum(not r['proxy_contradiction'] and r['expected_contradiction'] for r in subset)}
    return {'families':groups,'cases':len(records),
        'false_contradictions':sum(r['proxy_contradiction'] and not r['expected_contradiction'] for r in records),
        'missed_contradictions':sum(not r['proxy_contradiction'] and r['expected_contradiction'] for r in records),
        'scope':'independent synthetic semantic scope stress only; NOT benchmark accuracy, retuning or confirmed world truth'}


def run():
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('DGX Slurm only')
    rows=fixtures(); out=ROOT/'outputs/acl'/f'source_semantics_stress_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True,exist_ok=True); record_environment(out)
    nli=SourceNLI(ROOT,out/'pairs.sqlite')
    # Fixture expectations are not passed to the model.
    labels=nli.predict([p for r in rows for p in ((r['left'],r['right']),(r['right'],r['left']))])
    records=[{**r,'forward_label':labels[2*i],'reverse_label':labels[2*i+1],
              'proxy_contradiction':labels[2*i]==labels[2*i+1]=='contradiction'} for i,r in enumerate(rows)]
    report={'fixture_sha256':stable_hash(rows),'worker_sha256':digest_file(__file__),
        'model_identity':nli.identity,'summary':summarize(records),'cases':records,
        'no_benchmark_inputs_or_fitted_thresholds':True,'fresh_model_pairs':nli.actual_pairs,
        'full_runs_authorized':False}
    write_json(out/'analysis.json',report); print('STRESS',report['summary'],flush=True)


if __name__=='__main__': run()
