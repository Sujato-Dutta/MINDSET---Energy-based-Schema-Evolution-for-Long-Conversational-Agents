"""No models/downloads; execute in the root-coordinated DGX test job."""
import copy
import pytest

from mindset.methods.ideas_window import search_window, ReversibleWindowMemory
from mindset.methods.compact import CompactConfig
from test_compact import Encoder, Reranker, event


def plan(action, cost):
    return {'action': action, 'mutation': action, 'schema_id': 's', 'cost': cost}


def independent_branch(state, episode, chosen):
    return state + [(episode,chosen['action'])]


def test_window_can_reverse_earlier_greedy_choice_when_later_source_changes_objective():
    def feasible(state, episode):
        if not state: return [plan('reinforce',0),plan('split',1)]
        return [plan('reinforce',10 if state[0][1]=='reinforce' else 0)]
    prefix=[]
    winner,audit=search_window(prefix,['a','b'],feasible,independent_branch,beam_width=4)
    assert winner['state']==[('a','split'),('b','reinforce')]
    assert winner['cost']==1 and audit['greedy_cost']==10
    assert prefix==[] and not audit['global_optimality_claimed']


def test_greedy_incumbent_survives_beam_pruning():
    # Greedy's locally good branch is pruned after accumulated costs, but later
    # becomes best. The retained independent complete path is mandatory.
    def feasible(state, episode):
        if episode=='a': return [plan('reinforce',0),plan('split',.5)]
        if episode=='b':
            return ([plan('reinforce',1)] if state[0][1]=='reinforce'
                else [plan('reinforce',0),plan('new',.1)])
        return [plan('reinforce',0 if state[0][1]=='reinforce' else 20)]
    winner,audit=search_window([],['a','b','c'],feasible,independent_branch,beam_width=2)
    assert winner['cost']==audit['greedy_cost']==1
    assert audit['greedy_incumbent_retained']


def test_window_ties_are_deterministic_and_prefix_frozen():
    prefix=[('frozen','new')]
    frozen=copy.deepcopy(prefix)
    feasible=lambda state,episode:[plan('split',1),plan('reinforce',1)]
    winner,_=search_window(prefix,['a'],feasible,independent_branch)
    assert winner['plans'][0]['action']=='reinforce' and prefix==frozen


def test_invalid_window_configuration_scores_and_empty_feasible_set():
    with pytest.raises(ValueError): search_window([],[],lambda s,e:[],independent_branch,beam_width=0)
    with pytest.raises(ValueError): search_window([],['a'],lambda s,e:[],independent_branch)
    with pytest.raises(ValueError): search_window([],['a'],lambda s,e:[plan('new',float('nan'))],independent_branch)


def memory(**kwargs):
    return ReversibleWindowMemory(CompactConfig(complexity_penalty=.1,switch_margin=.2),
        encoder=Encoder(),local_reranker=Reranker(),**kwargs)


def test_real_window_preserves_immutable_sources_models_and_transition_invariants():
    m=memory(); encoder=m.encoder; reranker=m.local_reranker
    events=[event(1,'I live in Paris.'),event(2,'I live in Rome.'),event(3,'I moved to Rome.'),
        event(4,'I live in Rome.'),event(5,'I prefer tea at home.')]
    m.ingest(events)
    assert m.encoder is encoder and m.local_reranker is reranker
    assert m.validate_invariants()['grounded_episode_count']==5
    assert m._episode_order==[e.event_id for e in events]
    assert all(m.episodes[e.event_id].text==e.text for e in events)
    assert len(m.last_idea_certificate['window_ids'])==3
    assert m.last_idea_certificate['frozen_prefix_transition_count']==2
    assert all(r['selected_cost']<=r['greedy_cost']+1e-12 for r in m.window_audit)
    assert m.retrieve('Where does Sam live?',8)


def test_frozen_prefix_does_not_change_after_leaving_window():
    m=memory(window_size=2); m.ingest([event(1,'I live in Paris.'),event(2,'I moved to Rome.'),event(3,'I live in Rome.')])
    first=m.operation_log.all()[0]
    m.ingest([event(4,'I moved to Paris.')])
    assert m.operation_log.all()[0]==first


def test_invalid_memory_window_budget():
    with pytest.raises(ValueError): memory(window_size=0)
    with pytest.raises(ValueError): memory(beam_width=0)
