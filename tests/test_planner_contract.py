import json
from mindset.acl.planner_contract import explicit_messages
from mindset.methods.query_state_agent import planner_schema


def test_schema_visible_to_model_and_context_not_operation():
    rows=[{'entity':'sam','attributes':{'location':['Paris']},'contexts':[]}]
    messages=explicit_messages('Where does Sam live now?',rows)
    payload=json.loads(messages[1]['content'])
    assert payload['output_schema']==planner_schema(rows)
    assert 'Put CURRENT here, not in context' in messages[0]['content']
    assert 'never null' in messages[0]['content']
    assert set(payload)=={'question','question_date','observed_inventory','output_schema'}


def test_no_natural_answer_or_gold_field():
    payload=json.loads(explicit_messages('test',[])[1]['content'])
    assert 'answer' not in payload and 'gold' not in payload
