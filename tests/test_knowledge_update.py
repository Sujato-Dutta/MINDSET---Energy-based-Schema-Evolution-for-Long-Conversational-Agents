import pytest
from mindset.acl.knowledge_update import select_records
from mindset.methods.report_grounding import ReportCatalog, PredicateEquivalence
from test_report_grounding import fixture, query


def test_native_category_selection_complete_source_order_and_overlap_only():
    native={'records':[{'id':'z','category':'knowledge-update'},{'id':'a','category':'temporal-reasoning'},
                       {'id':'b','category':'knowledge-update'}]}
    dev={'records':[{'dataset':'longmemeval','id':'b'},{'dataset':'locomo','id':'z'}]}
    rows=select_records(native,dev)
    assert [r['id'] for r in rows]==['z','b']
    assert [r['in_development100'] for r in rows]==[False,True]
    assert all(r['dataset']=='longmemeval' for r in rows)


def test_duplicate_or_missing_native_category_rejected():
    with pytest.raises(ValueError): select_records({'records':[]},{'records':[]})
    with pytest.raises(ValueError): select_records({'records':[{'id':'x','category':'knowledge-update'}]*2},{'records':[]})


def test_latest_control_returns_one_actual_source_without_claiming_transition():
    events,rows=fixture([1,2]); c=ReportCatalog(rows,events,PredicateEquivalence())
    ids,cert=c.witnesses(query(),selector='latest')
    assert ids==['e2'] and not cert['committed_report_edge'] and not cert['world_truth_certified']


def test_latest_control_does_not_override_unsupported_query_binding():
    events,rows=fixture([1,2]); c=ReportCatalog(rows,events,PredicateEquivalence())
    assert c.witnesses(query(mode='ordinary'),selector='latest')[0]==[]


def test_quality_summary_uses_native_count_and_separate_exposure_panels(tmp_path):
    import importlib.util
    from pathlib import Path
    from mindset.acl.io import append_row,read_json
    path=Path(__file__).resolve().parents[1]/'scripts/probe_report_grounding.py'
    spec=importlib.util.spec_from_file_location('quality_ku_test',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.ARMS=('energy','previous')
    for qid in ('seen','disjoint'):
        for arm in module.ARMS:
            row={'id':f'{arm}/{qid}','question_id':qid,'dataset':'longmemeval','arm':arm,
                 'retrieval':{m:.5 for m in module.METRICS},'candidate_recall':1.,'evidence_tokens':10,
                 'report_certificate':{'reserved_ids':[],'committed_report_edge':False},
                 'grounding_diagnostic':{'gold_session_report_presence':.5,'gold_session_bound_report_presence':0.},
                 'prompt_sha256':qid,'state_certificate':None}
            append_row(tmp_path/'evidence.jsonl',row)
    cohort={'count':2,'schema_version':'mindset-native-knowledge-update-validation-v1',
            'overlap_development100_ids':['seen'],'known_v2_scored_exposure':{'seen':['old.jsonl']}}
    module.summary(tmp_path,cohort)
    status=read_json(tmp_path/'status.json')
    assert status['completed_cells']==status['expected_cells']==4
    assert status['status']=='complete_development'
    import csv
    with (tmp_path/'retrieval.csv').open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==8
    assert all(int(r['questions'])==(2 if r['panel']=='all' else 1) for r in rows)
