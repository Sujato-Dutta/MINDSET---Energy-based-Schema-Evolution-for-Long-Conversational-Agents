"""Small source-only contracts; execute exclusively in DGX Slurm test jobs."""
from dataclasses import replace
import numpy as np
import pytest

from mindset.methods.base import ConversationEvent
from mindset.methods.compact import CompactConfig
from mindset.methods.ideas_signed_temporal_spectral_repair import (
    ObservedFact, TemporalProgram, SpectralConfig, observed_facts,
    certified_edges, compile_temporal, execute_temporal, signed_selection,
    spectral_partition, source_consistency, SignedRetrievalMemory,
    TemporalCompilerMemory, SpectralBifurcationMemory, CounterfactualRepairMemory,
    replay_source_greedy,
)
from mindset.methods.joint import JointMindsetMemory
from test_compact import Encoder, Reranker, event


def memory(cls=JointMindsetMemory, **kwargs):
    return cls(CompactConfig(complexity_penalty=.1,switch_margin=.2),
        encoder=Encoder(), local_reranker=Reranker(), **kwargs)


def fact(value, eid, stamp, change=False, context=()):
    return ObservedFact('sam','location',value,context,float(stamp),eid,change)


def test_signed_constraints_change_membership_without_deleting_history():
    assert signed_selection(['old','new','other'],[9,8,7],{'old'},top_k=2)==['new','other']
    assert signed_selection(['old','new'],[9,8],set(),top_k=2)==['old','new']
    assert signed_selection(['old'],[9],{'old'},top_k=8)==['old']


def test_invalid_signed_scores_rejected():
    with pytest.raises(ValueError): signed_selection(['a'],[np.nan],set())
    with pytest.raises(ValueError): signed_selection(['a','a'],[1,2],set())


def test_edge_requires_explicit_change_same_owner_context_and_strict_time():
    old=fact('paris','a',1); new=fact('rome','b',2,True)
    assert certified_edges([old,new])==[(old,new)]
    assert not certified_edges([old,replace(new,explicit_change=False)])
    assert not certified_edges([old,replace(new,timestamp=1)])
    assert not certified_edges([old,replace(new,owner='rae')])
    assert not certified_edges([old,replace(new,context=('at work',))])


@pytest.mark.parametrize('query,operator',[
    ('Where does Sam live now?','current'),
    ('Where did Sam live previously?','previous'),
    ('Where did Sam live on 2025-01-01?','value_at'),
    ('What changes occurred in where Sam lives?','changes'),
    ('Where did Sam live during 2025-01-01 and 2025-02-01?','valid_during'),
    ('Where does Sam live at home?','context_value'),
])
def test_temporal_compiler_six_operators(query,operator):
    program, reason=compile_temporal(query,{'sam','rae'})
    assert reason=='compiled' and program.operator==operator


@pytest.mark.parametrize('query',[
    'Where do they live now?', 'Where do Sam and Rae live now?',
    'What is Sam doing now?', 'Where does Samantha live now?',
    "Where does Sam's sister live now?",
    'Where did Sam live during 2025-01-01 and 2024-01-01?',
    'Where did Sam live on 2025-02-31?',
])
def test_temporal_compiler_honest_unknown(query):
    assert compile_temporal(query,{'sam','rae'})[0] is None


def test_temporal_execution_current_previous_changes_and_ambiguous_fallback():
    facts=[fact('paris','a',1),fact('rome','b',2,True),fact('rome','c',3)]
    assert execute_temporal(TemporalProgram('current','sam','location'),facts)[:2]==(['c'],'rome')
    assert execute_temporal(TemporalProgram('previous','sam','location'),facts)[:2]==(['a','b'],'paris')
    assert execute_temporal(TemporalProgram('changes','sam','location'),facts)[:2]==(['a','b'],'paris -> rome')
    assert execute_temporal(TemporalProgram('current','sam','location'),
        [facts[0],replace(facts[1],explicit_change=False)])[1] is None
    assert execute_temporal(TemporalProgram('current','sam','location'),
        [facts[0],replace(facts[1],timestamp=1)])[2]=='simultaneous_conflicting_reports'


def test_temporal_intervals_are_half_open_and_require_observed_start():
    facts=[fact('paris','a',1),fact('rome','b',2,True)]
    assert execute_temporal(TemporalProgram('value_at','sam','location',time=1.5),facts)[:2]==(['a'],'paris')
    assert execute_temporal(TemporalProgram('valid_during','sam','location',time=1,end=2),facts)[1]=='paris'
    assert execute_temporal(TemporalProgram('valid_during','sam','location',time=1,end=3),facts)[1] is None
    assert execute_temporal(TemporalProgram('value_at','sam','location',time=.5),facts)[1] is None


def test_context_is_not_collapsed_into_unqualified_current_value():
    facts=[fact('paris','a',1,context=('at home',)),fact('rome','b',2,context=('at work',))]
    assert execute_temporal(TemporalProgram('current','sam','location'),facts)[1] is None
    assert execute_temporal(TemporalProgram('context_value','sam','location',context='at work'),facts)[1]=='rome'


def test_source_facts_exclude_quotes_negations_plans_and_multi_values():
    m=memory(); m.ingest([event(1,'I live in Paris.'),event(2,'I moved to Rome.'),
        event(3,'I do not live in London.'),event(4,'I will live in Oslo.'),
        event(5,'"I live in Madrid," Rae said.'),event(6,'I live in Paris and Rome.')])
    facts=observed_facts(m.episodes.values())
    assert [f.episode_id for f in facts]==['e1','e2']
    assert len(certified_edges(facts))==1


def test_signed_historical_query_retains_unsigned_common_ranking():
    m=memory(SignedRetrievalMemory); m.ingest([event(1,'I live in Paris.'),event(2,'I moved to Rome.')])
    m.retrieve('Where does Sam live now?',1)
    assert m.last_idea_certificate['anti_support_ids']==['e1']
    old=m.retrieve('Where did Sam live previously?',8)
    assert not m.last_idea_certificate['anti_support_ids']
    assert {h.memory_id for h in old}=={'e1','e2'}
    assert m.validate_invariants()['grounded_episode_count']==2


def test_compiler_primary_path_still_reranks_raw_evidence_and_exposes_lookup_separately():
    m=memory(TemporalCompilerMemory); m.ingest([event(1,'I live in Paris.'),event(2,'I moved to Rome.')])
    hits=m.retrieve('Where does Sam live now?',8)
    assert m.deterministic_answer=='rome'
    assert m.local_reranker.calls==[2]
    assert all(h.source_event_ids==(h.memory_id,) for h in hits)
    assert m.last_idea_certificate['lookup_is_separate_diagnostic']
    m.retrieve('What does Sam enjoy?',8)
    assert m.deterministic_answer is None


def test_reserved_multiple_witnesses_survive_small_budget():
    m=memory(TemporalCompilerMemory); m.ingest([event(1,'I live in Paris.'),event(2,'I moved to Rome.'),event(3,'Tea coffee garden')])
    hits=m.retrieve('Where did Sam live previously?',2)
    assert {h.memory_id for h in hits}=={'e1','e2'}


def test_spectral_uses_normalized_laplacian_and_reproducible_disconnected_components():
    vectors=np.asarray([[1,0],[1,.01],[0,1],[.01,1]])
    result=spectral_partition(vectors)
    assert result['bifurcation'] and result['labels']==[0,0,1,1]
    assert result['eigenvalues'][1]==pytest.approx(0,abs=1e-10)
    assert result['eigengap']>1
    assert not result['semantic_validity_certified']


def test_coherent_and_single_disconnected_spectral_observations():
    assert not spectral_partition([[1,0],[1,.01],[1,.02],[1,.03]])['bifurcation']
    result=spectral_partition([[1,0],[1,.01],[0,1]])
    assert result['newest_disconnected'] and not result['bifurcation']
    with pytest.raises(ValueError): spectral_partition(np.ones((33,2)))
    with pytest.raises(ValueError): spectral_partition([[np.nan,1]])


def test_spectral_memory_preserves_source_invariants_and_declares_geometric_not_semantic_partition():
    m=memory(SpectralBifurcationMemory); m.ingest([event(1,'I prefer tea.'),event(2,'I prefer coffee.')])
    assert m.validate_invariants()['grounded_episode_count']==2
    assert m.spectral_audit and not m.last_idea_certificate['semantic_validity_certified']


def test_source_loss_is_based_on_actual_source_values_not_answer_labels():
    m=memory(); m.ingest([event(1,'I live in Paris.'),event(2,'I live in Rome.')])
    assert source_consistency(m)==1
    assert source_consistency(m,{'e1'})==0
    m.episodes['e1']=replace(m.episodes['e1'],metadata={'answer':'secret reference','f1':0})
    assert source_consistency(m)==1


def test_repair_nontrigger_does_not_need_shared_replay_tools():
    m=memory(CounterfactualRepairMemory); m.ingest([event(1,'I live in Paris.'),event(2,'I live in Paris.')])
    before={sid:tuple(s.all_provenance()) for sid,s in m.schemas.items()}
    assert m.retrieve('Where does Sam live?',8)
    assert not m.last_idea_certificate['triggered']
    assert before=={sid:tuple(s.all_provenance()) for sid,s in m.schemas.items()}


def test_repair_budgets_must_be_positive():
    with pytest.raises(ValueError): memory(CounterfactualRepairMemory,repair_window=0)


def test_multischema_child_tail_replay_cannot_mutate_parent_or_sibling():
    """Tail writes to an untouched shared employer schema must detach it."""
    from mindset.methods.idea_transition_tools import feasible_plans, state_for_plan
    from mindset.methods.joint import typed_features
    from mindset.memory.energy import EnergyEpisode, unit_vector
    parent=memory(); parent.ingest([event(1,'I live in Paris.'),event(2,'I work at Acme.')])
    before={sid:tuple(schema.support_episode_ids) for sid,schema in parent.schemas.items()}
    sums={sid:value.copy() for sid,value in parent._sums.items()}
    original_order=list(parent._episode_order)
    original_log=parent.operation_log.all()
    def episode(i,text):
        raw=event(i,text)
        vector=parent._encode([f'[{raw.timestamp}] {raw.speaker}: {raw.text}'])[0]
        return EnergyEpisode(raw.event_id,raw.conversation_id,raw.session_id,raw.timestamp,
            raw.speaker,raw.text,unit_vector(vector),typed_features(raw.speaker,raw.text),i-1)
    origin=episode(3,'I moved to Rome.')
    plans=feasible_plans(parent,origin)
    chosen=next(plan for plan in plans if plan['action']=='supersede')
    child=state_for_plan(parent,origin,chosen)
    sibling=state_for_plan(parent,origin,chosen)
    tail=episode(4,'I work at Acme.')
    replay_source_greedy(child,[tail])
    assert list(child._episode_order)==['e1','e2','e3','e4']
    assert list(sibling._episode_order)==['e1','e2','e3']
    assert list(parent._episode_order)==original_order
    assert parent.operation_log.all()==original_log
    assert before=={sid:tuple(schema.support_episode_ids) for sid,schema in parent.schemas.items()}
    for sid,values in sums.items(): np.testing.assert_array_equal(parent._sums[sid],values)
    employer_sid=next(sid for sid,schema in parent.schemas.items() if schema.topic.endswith('::employer'))
    assert child.schemas[employer_sid].support_episode_ids==['e2','e4']
    assert parent.schemas[employer_sid].support_episode_ids==['e2']
    assert sibling.schemas[employer_sid].support_episode_ids==['e2']
