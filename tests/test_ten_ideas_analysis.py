"""Arithmetic/provenance checks for the bounded pilot report; DGX tests only."""
import importlib.util
from pathlib import Path
import pytest
from mindset.acl.io import write_json,append_row,read_json

def analyzer():
    path=Path(__file__).resolve().parents[1]/'scripts/analyze_ten_ideas_pilot.py'
    spec=importlib.util.spec_from_file_location('ten_idea_analysis',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def audit():
    return dict(manifest_sha256='manifest',calibration_sha256='calibration',parameters={'x':.1},
        checkpoints={'encoder':'locked'},configuration={'top_k':8},source_adapter='text',
        controller_source_sha256={'a':'fixed'},worker_sha256='worker')

def row(method,q,score):
    return {'id':f'locomo/{method}/{q}','dataset':'locomo','method':method,'question_id':q,
        'metrics':{k:score for k in ('recall@8','precision@8','mrr@8','ndcg@8')},
        'candidate_pool_recall@48':score,'selected':[{'event_id':q}],
        'evidence':q,'evidence_tokens':10,'diagnostic_query_seconds':.01}

def test_report_deltas_use_same_pair_subset(tmp_path,monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID','test-fixture')
    root=tmp_path/'input'; shard=root/'shard-0'; shard.mkdir(parents=True)
    write_json(shard/'audit.json',audit())
    for r in (row('joint','a',1),row('joint','b',.2),row('idea01_rwse','b',.5)):
        append_row(shard/'retrieval.jsonl',r)
    out=tmp_path/'output'; analyzer().analyze(root,out)
    report=read_json(out/'analysis.json')['comparisons']
    proposed=next(r for r in report if r['method']=='idea01_rwse')
    assert proposed['paired_questions']==1
    assert proposed['delta_recall@8']==pytest.approx(.3)  # not .5 - (.2+1)/2
    assert read_json(out/'status.json')['unique_questions']==2
    assert read_json(out/'status.json')['status']=='partial_development'

def test_analysis_rejects_mixed_controller_revisions(tmp_path,monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID','test-fixture')
    root=tmp_path/'input'
    for i in range(2):
        shard=root/f'shard-{i}'; shard.mkdir(parents=True)
        a=audit(); a['controller_source_sha256']={'a':str(i)}
        write_json(shard/'audit.json',a)
        append_row(shard/'retrieval.jsonl',row('joint',str(i),.5))
    with pytest.raises(ValueError,match='controller_source_sha256'):
        analyzer().analyze(root,tmp_path/'output')
