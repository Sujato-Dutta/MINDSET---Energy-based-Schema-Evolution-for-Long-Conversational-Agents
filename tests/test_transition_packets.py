"""Bounded transition-packet tests; execute exclusively in DGX Slurm."""
from dataclasses import replace
import pytest
from mindset.methods.compact import CompactConfig
from mindset.methods.joint import JointMindsetMemory
from mindset.methods.query_state_agent import state_fingerprint
from mindset.methods.transition_packets import (Contrast,TransitionCatalog,inject,intent,slot_matches,
    required_ids,select_packet)
from test_compact import Encoder,Reranker,event


def memory(events=None):
    m=JointMindsetMemory(CompactConfig(.1,.05),encoder=Encoder(),local_reranker=Reranker())
    m.ingest(events or [event(1,'I live in Paris.'),event(2,'I moved to Rome.')]); return m


class Pack:
    def __init__(self,k=2,visible=None): self.k=k; self.visible=visible
    def __call__(self,ids):
        selected=[{'event_id':i,'text':i} for i in list(dict.fromkeys(ids))[:self.k]]
        return selected,' '.join(s['text'] for s in selected)
    def token_count(self,text): return len(text.split())
    def witness_ids(self,selected):
        return {s['event_id'] for s in selected if self.visible is None or s['event_id'] in self.visible}


def test_catalog_read_only_and_actual_edges_separate():
    m=memory(); before=state_fingerprint(m); c=TransitionCatalog(m)
    assert any(e.kind=='schema_supersession' and e.schema_edge for e in c.edges)
    assert any(e.kind=='change_cue_report_contrast' and not e.schema_edge for e in c.edges)
    assert before==state_fingerprint(m)


def test_no_change_assertion_from_unqualified_multivalued_likes():
    c=TransitionCatalog(memory([event(1,'I like tea.'),event(2,'I like coffee.')]))
    assert not c.edges


def test_same_time_conflicts_do_not_create_arbitrary_temporal_edges():
    m=memory([event(1,'I live in Paris.'),replace(event(2,'I live in Rome.'),timestamp='2025-01-01'),
        event(3,'I live in Oslo.')])
    c=TransitionCatalog(m)
    assert c.ambiguous_time_groups==1
    assert not any(e.kind!='schema_supersession' for e in c.edges)


def test_role_slot_and_retrieval_anchors_required():
    m=memory(); c=TransitionCatalog(m)
    assert c.bind('Where does Sam live now?',['e2'],m)[0]
    for query in ('Where does Rae live now?',"Where does Sam's sister live now?",
                  'What does Sam eat now?','When did Sam plan to travel?'):
        assert not c.bind(query,['e2'],m)[0]
    assert not c.bind('Where does Sam live now?',['other'],m)[0]


def test_first_person_user_binding_without_role_inference():
    m=memory([replace(event(1,'I live in Paris.'),speaker='user'),replace(event(2,'I moved to Rome.'),speaker='user')])
    assert TransitionCatalog(m).bind('Where do I live now?',['e2'],m)[0]
    assert not TransitionCatalog(memory()).bind('Where do I live now?',['e2'],memory())[0]


def test_deadline_applies_to_binding_and_naive_sequence():
    m=memory([event(1,'I live in Paris.'),event(2,'I moved to Rome.'),event(3,'I moved to Oslo.')])
    c=TransitionCatalog(m)
    edges,_=c.bind('Where does Sam live now?',['e2'],m,'2025-01-02')
    assert edges and all('e3' not in (e.left,e.right) for e in edges)
    assert 'e3' not in c.chain(edges[0],'2025-01-02')
    assert not c.chain(edges[0],'2024-01-01')
    assert not c.bind('Where does Sam live now?',['e2'],m,'2024-01-01')[0]


def test_context_alone_does_not_bind_unrelated_attribute():
    m=memory([event(1,'I prefer tea at home.'),event(2,'I prefer coffee at work.')]); c=TransitionCatalog(m)
    assert c.bind('What does Sam prefer at work?',['e2'],m)[0]
    assert not c.bind('Where does Sam live at work?',['e2'],m)[0]


def test_budgeted_injection_preserves_direct24_and_uniqueness():
    original=[f'e{i}' for i in range(48)]; edge=Contrast('x','y','sam','location','observed_report_contrast')
    new=inject(original,[edge]); assert len(new)==48 and new[:24]==original[:24]
    assert {'x','y'}.issubset(new) and len(set(new))==48
    with pytest.raises(ValueError): inject(['x','x'],[])


def test_energy_ablation_same_proposals_and_only_objective_differs():
    m=memory(); c=TransitionCatalog(m); edge=Contrast('e1','e2','sam','location','observed_report_contrast')
    ranking=['other1','other2','e2','e1']; pack=Pack()
    a=select_packet('compact_energy',ranking,[edge],'change','How has Sam changed location?',c,pack)
    b=select_packet('compact_no_energy',ranking,[edge],'change','How has Sam changed location?',c,pack)
    assert a[2]['proposals']==b[2]['proposals']
    assert {s['event_id'] for s in a[0]}=={'e1','e2'}
    assert {s['event_id'] for s in b[0]}=={'other1','other2'}
    assert a[2]['energy_used'] and not b[2]['energy_used']


def test_selected_episode_without_value_is_not_a_passage_witness():
    edge=Contrast('e1','e2','sam','location','observed_report_contrast')
    result=select_packet('compact_energy',['e1','e2'],[edge],'change','change',TransitionCatalog(memory()),
        Pack(visible={'e2'}))
    assert result[2]['witnesses_survived']==['e2']
    assert result[2]['proposals'][result[2]['selected_proposal']]['energy'][0]==1


@pytest.mark.parametrize('question,mode',[
    ('When does Sam travel?','ordinary'),('Where does Sam live now?','current'),
    ('Where did Sam live before?','historical'),('How did Sam change location?','change')])
def test_intent_does_not_treat_every_when_question_as_change(question,mode):
    assert intent(question)==mode


def projection(m,eid,value,start,end):
    e=m.episodes[eid]
    return {'text':e.text,'speaker':e.speaker,'timestamp':e.timestamp,'reports':[{
        'family':'property','holder_terms':[],'tense':'nonpast','value':value,'normalized_value':value,
        'value_span':[start,end],'owner':'sam','predicate':'budget','unit_head':'budget','contexts':[]}]}


def test_property_projection_values_are_source_spans_not_generated():
    m=memory([event(1,'My budget is 10.'),event(2,'My budget is 20.')])
    rows={'e1':projection(m,'e1','10',13,15),'e2':projection(m,'e2','20',13,15)}
    c=TransitionCatalog(m,rows)
    assert c.bind("What is Sam's budget now?",['e2'],m)[0]
    rows['e1']['reports'][0]['value']='invented'
    with pytest.raises(ValueError,match='Generated report'): TransitionCatalog(m,rows)


def test_bad_source_attribution_rejected():
    m=memory(); row={'text':'wrong','speaker':'Sam','timestamp':'2025-01-01','reports':[]}
    with pytest.raises(ValueError,match='Projection/source mismatch'): TransitionCatalog(m,{'e1':row})


def test_property_match_requires_all_scope_words():
    assert slot_matches('property::budget::budget','what is my budget now?')
    assert not slot_matches('property::camera lens::camera lens','what is my camera now?')
    assert slot_matches('quantity::lead::worker','how many workers do I lead now?')


def test_historical_and_context_obligations_different():
    e=Contrast('a','b','sam','location','observed_report_contrast')
    assert required_ids(e,'historical','before')==('a',)
    assert required_ids(e,'current','now')==('a','b')
    e=Contrast('a','b','sam','preference','observed_context_contrast',('at home',),('at work',))
    assert required_ids(e,'ordinary','at work')==('b',)
