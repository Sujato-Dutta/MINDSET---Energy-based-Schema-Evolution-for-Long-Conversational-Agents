"""Read-only state execution contracts; run on DGX, never the laptop."""
import json
from dataclasses import replace
import pytest
from mindset.methods.compact import CompactConfig
from mindset.methods.joint import JointMindsetMemory
from mindset.methods.query_state_agent import (inventory, execute, planner_schema,
    planner_messages, retrieve, state_fingerprint, fallback)
from test_compact import Encoder, Reranker, event


def memory(events=None):
    m=JointMindsetMemory(CompactConfig(complexity_penalty=.1,switch_margin=.05),
        encoder=Encoder(),local_reranker=Reranker())
    m.ingest(events or [event(1,'I live in Paris.'),event(2,'I moved to Rome.')])
    return m


def plan(op='CURRENT',**kwargs):
    result={'intent':'current','operation':op,'entity':'sam','attribute':'location','context':'','time':''}
    result.update(kwargs); return result


def run(m,query,p):
    return execute(m,query,p,inventory(m,query))


def test_inventory_is_observed_and_read_only():
    m=memory(); before=state_fingerprint(m); rows=inventory(m,'Where does Sam live now?')
    assert rows and {r['entity'] for r in rows}=={'sam'}
    assert all(r['schema_id'] in m.schemas for r in rows)
    assert state_fingerprint(m)==before
    s=planner_schema(rows)
    assert s['additionalProperties'] is False and 'location' in s['properties']['attribute']['enum']


def test_actual_current_and_previous_and_historical_provenance():
    m=memory()
    current=run(m,'Where does Sam live now?',plan())
    assert current['bound'] and current['provenance_ids']==['e2']
    previous=run(m,'Where did Sam live before?',plan('PREDECESSOR',intent='historical'))
    assert previous['bound'] and previous['transition_link_used'] and set(previous['provenance_ids'])=={'e1','e2'}
    historical=run(m,'Where did Sam live?',plan('HISTORY',intent='historical'))
    assert historical['bound'] and set(historical['provenance_ids'])=={'e1','e2'}
    assert not historical['transition_link_used']


def test_trace_requires_actual_edges_not_raw_date_order():
    m=memory([event(1,'I live in Paris.')])
    result=run(m,'How has where Sam lives changed?',plan('TRACE_CHANGE',intent='temporal_change'))
    assert not result['bound'] and result['reason']=='no_actual_version_edges'
    m=memory()
    assert run(m,'How has where Sam lives changed?',plan('TRACE_CHANGE',intent='temporal_change'))['transition_link_used']


def test_when_question_not_silently_bound_to_change_history():
    m=memory()
    result=run(m,'When does Sam plan on traveling?',plan('TRACE_CHANGE',intent='temporal_change'))
    assert not result['bound'] and result['reason']=='not_explicit_change_question'


def test_at_time_half_open_validity_and_no_invented_dates():
    m=memory()
    result=run(m,'Where did Sam live on 2025-01-01?',plan('AT_TIME',intent='historical',time='2025-01-01'))
    assert result['bound'] and result['provenance_ids']==['e1']
    assert run(m,'Where did Sam live on 2025-01-02?',plan('AT_TIME',intent='historical',time='2025-01-02'))['provenance_ids']==['e2']
    assert not run(m,'Where did Sam live last January?',plan('AT_TIME',time='2025-01-01'))['bound']
    assert not run(m,'Where did Sam live on 2025-02-31?',plan('AT_TIME',time='2025-02-31'))['bound']


def test_exception_cannot_be_overwritten_by_query():
    m=memory([event(1,'I live in Paris.'),event(2,'I live in Rome.')])
    result=run(m,'Where does Sam live now?',plan())
    assert not result['bound'] and result['reason']=='unresolved_attribute_exceptions'


def test_context_executes_existing_split_not_latest_raw_fact():
    m=memory([event(1,'I prefer tea at home.'),event(2,'I prefer coffee at work.')])
    result=run(m,'What does Sam prefer at home?',plan('CONTEXT',intent='contextual',attribute='preference',context='at home'))
    assert result['bound'] and result['branch_link_used'] and result['provenance_ids']==['e1']
    assert not run(m,'What does Sam prefer?',plan('CONTEXT',attribute='preference',context='at home'))['bound']


def test_broad_preference_not_single_valued_current():
    m=memory([event(1,'I prefer tea.'),event(2,'I prefer coffee.')])
    assert not run(m,'What does Sam prefer now?',plan(attribute='preference'))['bound']
    assert run(m,'What does Sam prefer?',plan('HISTORY',attribute='preference'))['bound']


@pytest.mark.parametrize('bad',[
    {'operation':'CURRENT'},plan(entity='not observed'),plan(attribute='music_preference'),
    plan(confidence=.99),plan(entity=17),plan('SEARCH'),plan(intent='multihop')])
def test_unsupported_or_invented_plans_fall_back(bad):
    assert not run(memory(),'Where does Sam live now?',bad)['bound']


def test_single_user_first_person_binding_but_not_relative_property():
    m=memory([replace(event(1,'I live in Paris.'),speaker='user')])
    assert run(m,'Where do I live now?',plan(entity='user'))['bound']
    assert not run(m,'Where does my sister live now?',plan(entity='user'))['bound']
    assert not run(m,'Where does my sister live now?',plan('HISTORY',entity='user'))['bound']


def test_fallback_identical_baseline_and_preserves_state():
    m=memory(); before=state_fingerprint(m)
    a=m.retrieve('Where did Sam live?',8)
    b=retrieve(m,'Where did Sam live?',fallback('unknown'),8)
    assert a==b and before==state_fingerprint(m)


def test_executed_witnesses_are_scored_source_ids_inside_budget():
    m=memory(); before=state_fingerprint(m)
    result=run(m,'Where does Sam live now?',plan())
    hits=retrieve(m,'Where does Sam live now?',result,8)
    assert hits[0].memory_id=='e2'
    assert all(h.source_event_ids==(h.memory_id,) for h in hits)
    assert len(m.last_candidate_pool)<=48 and m.local_reranker.calls[-1]==len(m.last_candidate_pool)
    assert before==state_fingerprint(m)


def test_planner_receives_question_and_observed_inventory_only():
    rows=inventory(memory(),'Where does Sam live?')
    payload=json.loads(planner_messages('Where does Sam live?',rows,'2025-01-04')[1]['content'])
    assert set(payload)=={'question','question_date','observed_inventory'}
    assert 'answer' not in payload and 'gold' not in payload
