from types import SimpleNamespace as N
import pytest
from mindset.acl.development_answers import request_plan
from mindset.acl.clients import ANSWER_PROMPT


def fixture():
    e=N(event_id='e1',speaker='Rae',timestamp='2025-01-01',text='I like tea.')
    case=N(question='What does Rae like?',question_date='2025-01-02',events=[e],cluster='c',category='single-hop',answer='SECRET_REFERENCE')
    selected={'event_id':'e1','speaker':'Rae','timestamp':'2025-01-01','text':'[2025-01-01] | Speaker: Rae\nI like tea.'}
    row={'dataset':'locomo','question_id':'q','method':'flat48','selected':[selected],
         'evidence':'[1] '+selected['text'],'metrics':{'recall@8':1}}
    return [row],{('locomo','q'):case}


def test_reference_not_inserted_into_answer_request():
    rows,cases=fixture(); item=request_plan(rows,cases,methods=('flat48',))[0]
    assert item['messages'][0]['content']==ANSWER_PROMPT
    assert 'SECRET_REFERENCE' not in str(item['messages'])
    assert item['answer']=='SECRET_REFERENCE'


def test_mismatched_source_attribution_and_serialization_rejected():
    rows,cases=fixture(); rows[0]['selected'][0]['speaker']='Someone else'
    with pytest.raises(ValueError): request_plan(rows,cases,methods=('flat48',))
    rows,cases=fixture(); rows[0]['evidence']='rewritten summary'
    with pytest.raises(ValueError): request_plan(rows,cases,methods=('flat48',))


def test_incomplete_duplicate_or_unknown_cells_rejected():
    rows,cases=fixture()
    with pytest.raises(ValueError): request_plan(rows,cases,methods=('flat48','joint'))
    with pytest.raises(ValueError): request_plan(rows+rows,cases,methods=('flat48',))
    rows[0]['question_id']='not-frozen'
    with pytest.raises(ValueError): request_plan(rows,cases,methods=('flat48',))
