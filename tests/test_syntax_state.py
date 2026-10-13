from dataclasses import replace
import pytest
from mindset.methods.syntax_state import SyntaxMindsetMemory, frame_features
from mindset.methods.compact import CompactConfig
from test_compact import Encoder, Reranker, event


def projected(text, value, changed=False):
    return {'text': text, 'speaker': 'user', 'frames': [{'slot': 'syntax_property_budget',
        'match_terms': ['budget'], 'value': value, 'value_span': [text.index(value), text.index(value) + len(value)],
        'contexts': [], 'explicit_change': changed}]}


def memory(**kwargs):
    texts = ['My budget is 120 dollars.', 'My budget is 80 dollars now.']
    query = 'How much is my current budget?'
    m = SyntaxMindsetMemory(CompactConfig(complexity_penalty=.1, switch_margin=.05),
        encoder=Encoder(), local_reranker=Reranker(),
        source_projections={f'e{i}': projected(t, v, i == 2) for i, (t, v) in enumerate(zip(texts, ['120 dollars', '80 dollars']), 1)},
        query_projections={query: {'terms': ['budget'], 'direct_self': False, 'possessive_self': True, 'present_tense': True}}, **kwargs)
    m.ingest([replace(event(i, t), speaker='user') for i, t in enumerate(texts, 1)])
    return m, query


def test_dependency_binding_and_predicate_make_raw_state_packet_effective():
    m, q = memory()
    scope = m._scope(q)
    assert scope.owners == ('user',) and 'syntax_property_budget' in scope.slots and scope.mode == 'current'
    assert [h.memory_id for h in m.select_scored(q, [0, 1], [10., 1.], 8)] == ['e2']
    assert m.last_readout_certificate['excluded']['e1'] == ['historical_schema']
    assert m.validate_invariants()['episode_count'] == 2


def test_state_off_is_same_sources_but_no_constraints():
    m, q = memory(state_readout=False)
    assert [h.memory_id for h in m.select_scored(q, [0, 1], [10., 1.], 8)] == ['e1', 'e2']


def test_no_compatible_witness_does_not_hide_a_source():
    m, q = memory()
    assert [h.memory_id for h in m.select_scored(q, [0], [10.], 8)] == ['e1']


def test_invented_value_or_stale_source_fails_closed():
    with pytest.raises(ValueError, match='grounded'):
        frame_features('user', 'My budget is 120 dollars.', [{'slot': 'syntax_property_budget',
            'match_terms': ['budget'], 'value': 'invented', 'value_span': [13, 20], 'contexts': [], 'explicit_change': False}])
    m, _ = memory()
    with pytest.raises(ValueError, match='stale'):
        m.ingest([replace(event(3, 'My budget is 10 dollars.'), speaker='user')])


def test_unknown_query_cannot_silently_use_unfrozen_parse():
    m, _ = memory()
    with pytest.raises(ValueError, match='query projection'): m._scope('An unseen question')


def test_missing_source_fails_before_any_raw_memory_mutation():
    m, _ = memory()
    old = m.validate_invariants()
    with pytest.raises(ValueError, match='stale'):
        m.ingest([replace(event(3, 'My budget is 10 dollars.'), speaker='user')])
    assert m.validate_invariants() == old


def test_change_query_exposes_real_version_edge_not_current_only_filter():
    m, _ = memory()
    q = 'How did my budget change?'
    m.query_projections[q] = {'terms': ['budget', 'change'], 'direct_self': False,
                             'possessive_self': True, 'present_tense': False}
    assert m._scope(q).mode == 'transition'
    hits = m.select_scored(q, [0, 1], [10., 1.], 8)
    assert {h.memory_id for h in hits} == {'e1', 'e2'}
    assert m.last_readout_certificate['transition_pair_reserved']
    assert not m.last_readout_certificate['excluded']
    assert not m._transition_pair(m._scope(q), '2025-01-01')
