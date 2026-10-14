"""Independent paired/component arithmetic and certificate audit fixtures."""
import importlib.util
from pathlib import Path
from copy import deepcopy
import pytest
from mindset.methods.semantic_witness import ARMS
spec=importlib.util.spec_from_file_location('witness_audit',
    Path(__file__).resolve().parents[1]/'scripts/audit_semantic_witness.py')
worker=importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def rows():
    result=[]
    for dataset in ('locomo','longmemeval'):
        for arm in ARMS:
            cert={'mode':'ordinary','proposals':[{'cost':[0,0,-1,1],'selected_ids':['e1']}],
                  'selected_proposal':0,'target_edges':[]}
            result.append({'id':f'{dataset}/{arm}/q','dataset':dataset,'method':arm,'question_id':'q',
                'selected':[{'event_id':'e1'}],'evidence':'source','evidence_tokens':1,
                'metrics':{m:.5 for m in worker.METRICS},'candidate_recall@48':1.,
                'certificate':deepcopy(cert),'semantic_edges':[]})
    return result


def test_equal_panels_no_invented_energy_effect():
    reports=worker.paired_components(rows())
    for dataset in reports.values():
        assert dataset['semantic_cover minus no_order']['membership_changes']==0
        assert dataset['semantic_pool minus flat48']['recall@8']==0
        assert dataset['objective_diagnostic']['selected_nondefault_proposals']==0


def test_incomplete_or_duplicate_panels_rejected():
    with pytest.raises(ValueError): worker.paired_components(rows()[:-1])
    with pytest.raises(ValueError): worker.paired_components(rows()+[rows()[0]])


def test_nonminimal_finite_proposal_detected():
    data=rows()
    for r in data:
        if r['method']=='no_order':
            r['certificate']['proposals'].append({'cost':[0,0,-2,1],'selected_ids':['e1']})
    with pytest.raises(ValueError,match='minimum'): worker.paired_components(data)


def test_ordering_cannot_change_objective():
    data=rows()
    for r in data:
        if r['method']=='semantic_cover': r['certificate']['mode']='current'
    with pytest.raises(ValueError,match='Ordering ablation'): worker.paired_components(data)


def test_active_objective_counts_only_actual_improvements():
    data=rows()
    for r in data:
        if r['method'] in ('no_order','semantic_cover'):
            r['certificate']['proposals']=[{'cost':[1,0,-1,1],'selected_ids':['e2']},
                                          {'cost':[0,0,-.5,1],'selected_ids':['e1']}]
            r['certificate']['selected_proposal']=1
    reports=worker.paired_components(data)
    assert reports['locomo']['objective_diagnostic']['queries_reducing_missing_witnesses']==1
