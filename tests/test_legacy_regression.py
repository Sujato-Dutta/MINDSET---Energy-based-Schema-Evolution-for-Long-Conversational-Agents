from types import SimpleNamespace as N
import importlib.util
from pathlib import Path
import pytest
from mindset.acl.development_answers import regression_plan,REGRESSION_METHODS
from mindset.acl.io import write_json,append_row,digest_file

spec=importlib.util.spec_from_file_location('legacy_regression',Path(__file__).resolve().parents[1]/'scripts/prepare_legacy_regression.py')
worker=importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def test_historical_sources_use_selected_not_gold_ids_and_keep_order():
    events=[N(event_id=i,timestamp='date',speaker='Rae',text=i) for i in ('selected1','selected2','gold')]
    case=N(id='q',question='Where?',answer='reference',category='single_hop',events=events,gold=['gold'])
    record={'id':'q','question':'Where?','answer':'reference','category':4,'retrieved_ids':['selected2','selected1'],
            'gold_evidence_ids':['gold']}
    hits=worker.historical_hits(case,record)
    assert [h.source_event_ids[0] for h in hits]==['selected2','selected1']
    assert all('reference' not in h.text for h in hits)
    record['retrieved_ids']=['selected1','selected1']
    with pytest.raises(ValueError): worker.historical_hits(case,record)
    record['retrieved_ids']=['not-in-source']
    with pytest.raises(ValueError): worker.historical_hits(case,record)


def panel(tmp_path,monkeypatch):
    import mindset.acl.development_answers as answers
    monkeypatch.setattr(answers,'ROOT',tmp_path)
    source=tmp_path/'outputs/acl/legacy_score_inputs_r01/predictions.jsonl'
    for i in range(35): append_row(source,{'id':f'q{i}'})
    prepared=tmp_path/'prepared'; write_json(prepared/'audit.json',{
        'kind':'retained_v1_regression_diagnosis','legacy_predictions_sha256':digest_file(source)})
    cases={}; rows=[]
    for i in range(50):
        e=N(event_id='e',timestamp='date',speaker='Rae')
        cases[('locomo',f'q{i}')]=N(events=[e],question='Where?',answer='EVALUATOR_ONLY',question_date='',cluster='c',category='single_hop')
        for method in REGRESSION_METHODS:
            if method=='historical_ids' and i>=35: continue
            selected=[{'event_id':'e','timestamp':'date','speaker':'Rae','text':'raw source'}]
            rows.append({'dataset':'locomo','method':method,'question_id':f'q{i}','selected':selected,
                'evidence':'[1] raw source','metrics':{}})
    return rows,cases,prepared


def test_regression_panel_requires_all_original50_and_every_available_shared_id(tmp_path,monkeypatch):
    rows,cases,prepared=panel(tmp_path,monkeypatch)
    planned=regression_plan(rows,cases,prepared)
    assert len(planned)==185
    assert all('EVALUATOR_ONLY' not in str(p['messages']) for p in planned)
    rows=[r for r in rows if not (r['method']=='historical_ids' and r['question_id']=='q34')]
    with pytest.raises(ValueError,match='Incomplete'): regression_plan(rows,cases,prepared)


def test_regression_panel_rejects_extra_or_unknown_arms(tmp_path,monkeypatch):
    rows,cases,prepared=panel(tmp_path,monkeypatch)
    rows[0]['method']='winning_cherry_pick'
    with pytest.raises(ValueError,match='arms'): regression_plan(rows,cases,prepared)
