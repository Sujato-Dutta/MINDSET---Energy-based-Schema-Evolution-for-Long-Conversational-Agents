import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('semantics_stress',
    Path(__file__).resolve().parents[1]/'scripts/stress_source_semantics.py')
worker=importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def test_independent_fixture_ids_groups_and_boolean_expectations():
    rows=worker.fixtures()
    assert len(rows)==24 and len({r['id'] for r in rows})==24
    assert sum(r['expected_contradiction'] for r in rows)==4
    assert len({r['family'] for r in rows})==6
    assert all(isinstance(r['expected_contradiction'],bool) for r in rows)


def test_stress_counts_do_not_treat_compatible_reports_as_errors_of_memory():
    rows=[{**r,'proxy_contradiction':r['expected_contradiction']} for r in worker.fixtures()]
    result=worker.summarize(rows)
    assert result['cases']==24 and result['false_contradictions']==result['missed_contradictions']==0
    rows[-1]['proxy_contradiction']=True; rows[0]['proxy_contradiction']=False
    result=worker.summarize(rows)
    assert result['false_contradictions']==result['missed_contradictions']==1
