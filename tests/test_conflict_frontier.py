from dataclasses import replace
import pytest
from mindset.methods.conflict_frontier import ConflictFrontierCatalog,select_frontier,LeanEpisodeIndex
from mindset.methods.report_grounding import PredicateEquivalence
from test_typed_grounding import typed_fixture
from test_compact import event
from test_compact import Encoder


def add_opaque(events,rows,text,*,day=12,speaker='user'):
    e=replace(event(day,text),speaker=speaker); events.append(e)
    rows[e.event_id]={'text':text,'speaker':speaker,'reports':[]}
    return e.event_id


def test_no_energy_or_committed_transition_is_required():
    events,rows,q=typed_fixture([1]*8+[2])
    c=ConflictFrontierCatalog(rows,events,PredicateEquivalence())
    ids,cert=c.frontier(q)
    assert ids==['e9','e8'] and not cert['energy_used'] and not cert['committed_report_edge']
    assert all(not g['closed'] for g in c.groups)
    assert all('chosen' not in item for item in c.audit)


def test_later_opaque_target_is_not_silently_ignored():
    events,rows,q=typed_fixture([1]*8+[2])
    eid=add_opaque(events,rows,'The worker roster changed again; I will explain later.')
    c=ConflictFrontierCatalog(rows,events,PredicateEquivalence())
    ids,cert=c.frontier(q)
    assert ids==[eid,'e9'] and cert['opaque_tail_ids']==[eid]
    assert cert['observed_latest_value']=='2' and not cert['world_truth_certified']
    assert c.frontier(q,opaque=False)[0]==['e9','e8']


def test_irrelevant_later_text_does_not_displace_witness():
    events,rows,q=typed_fixture([1,2]); add_opaque(events,rows,'I bought a new lens.')
    assert ConflictFrontierCatalog(rows,events,PredicateEquivalence()).frontier(q)[0]==['e2','e1']


def test_future_opaque_source_is_not_a_question_time_witness():
    events,rows,q=typed_fixture([1,2]); add_opaque(events,rows,'The worker roster changed.')
    q['question_date']='2025-01-03'
    assert ConflictFrontierCatalog(rows,events,PredicateEquivalence()).frontier(q)[0]==['e2','e1']


def test_assistant_assertion_not_promoted_as_user_update():
    events,rows,q=typed_fixture([1,2]); add_opaque(events,rows,'You lead three workers now.',speaker='assistant')
    assert ConflictFrontierCatalog(rows,events,PredicateEquivalence()).frontier(q)[0]==['e2','e1']


def test_unknown_timestamps_do_not_create_a_later_report():
    events,rows,q=typed_fixture([1,2]); add_opaque(events,rows,'The worker roster changed.')
    events[-1]=replace(events[-1],timestamp=None)
    assert ConflictFrontierCatalog(rows,events,PredicateEquivalence()).frontier(q)[0]==['e2','e1']


def test_no_target_contract_no_forced_frontier():
    events,rows,q=typed_fixture([1,2]); del q['report_query']['contract']
    assert ConflictFrontierCatalog(rows,events,PredicateEquivalence()).frontier(q)[0]==[]


def test_one_value_needs_only_one_witness():
    events,rows,q=typed_fixture([1,1])
    assert ConflictFrontierCatalog(rows,events,PredicateEquivalence()).frontier(q)[0]==['e2']


def test_selection_reserves_membership_without_bad_forced_rank():
    assert select_frontier([0,1,2,3],[8,7,6,1],[3],top_k=3)==[0,1,3]
    assert select_frontier([0,1],[2,1],[1],top_k=0)==[]
    assert select_frontier([0,1],[2,1],[])==[0,1]
    with pytest.raises(ValueError): select_frontier([0],[float('nan')],[])


def test_lean_index_preserves_rendering_order_and_ids():
    events=[event(1,'I live in Paris.'),event(2,'I prefer tea.')]
    index=LeanEpisodeIndex(events,Encoder())
    assert index.ids==['e1','e2'] and index.by_id=={'e1':0,'e2':1}
    assert index.documents[0]=='[2025-01-01] | Speaker: Sam\nI live in Paris.'
    assert index.candidates('Paris',Encoder())[0]==0
    with pytest.raises(ValueError): LeanEpisodeIndex([events[0],events[0]],Encoder())
