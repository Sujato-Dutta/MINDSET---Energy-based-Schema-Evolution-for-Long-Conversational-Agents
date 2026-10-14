import importlib.util
from pathlib import Path
import csv
import pytest
from mindset.acl.io import append_row,read_json
spec=importlib.util.spec_from_file_location('source_seed_worker',
    Path(__file__).resolve().parents[1]/'scripts/probe_raw_source_seeds.py')
worker=importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def panel(out,drop=None):
    expected=set()
    for dataset in ('locomo','longmemeval'):
        for q in ('a','b'):
            for arm in worker.ARMS:
                rid=f'{dataset}/{arm}/{q}'; expected.add(rid)
                if rid==drop: continue
                append_row(out/'retrieval.jsonl',{'id':rid,'dataset':dataset,'method':arm,'question_id':q,
                    'evidence':'changed' if arm=='hybrid_seed' and q=='a' else 'original',
                    'metrics':{m:.5 for m in worker.METRICS},'evidence_tokens':5,
                    'logical_ce_pairs':48 if arm in ('joint_ref','hybrid_seed') else 96})
    return expected


def test_source_seed_comparison_and_multi_stage_compute_are_explicit(tmp_path):
    worker.summarize(tmp_path,panel(tmp_path))
    assert read_json(tmp_path/'status.json')['completed_cells']==16
    rows=list(csv.DictReader((tmp_path/'comparison.csv').open()))
    hybrid=next(r for r in rows if r['dataset']=='locomo' and r['method']=='hybrid_seed')
    assert int(hybrid['packet_changes_from_expanded_joint'])==1
    assert float(hybrid['mean_logical_ce_pairs'])==48
    flat=next(r for r in rows if r['dataset']=='locomo' and r['method']=='flat_ce_seed')
    assert float(flat['mean_logical_ce_pairs'])==96
    assert read_json(tmp_path/'status.json')['qa_calls']==0


def test_partial_and_unexpected_source_seed_panels(tmp_path):
    expected=panel(tmp_path,'longmemeval/hybrid_seed/b'); worker.summarize(tmp_path,expected)
    assert read_json(tmp_path/'status.json')['status']=='partial_development'
    append_row(tmp_path/'retrieval.jsonl',{'id':'unknown'})
    with pytest.raises(ValueError): worker.summarize(tmp_path,expected)
