from itertools import product
from math import log
from dataclasses import replace
import numpy as np
import pytest
from mindset.methods.mdl_state import (
    categorical_code, transition_candidates, choose_transition, ObservedMDLEnergyMemory)
from mindset.methods.compact import CompactConfig
from mindset.methods.state_readout import QueryScope
from mindset.methods.joint import utc_time
from test_compact import Encoder, Reranker, event


def memory(values, *, sessions=None, contexts=None, **kwargs):
    events = [replace(event(i, f'I live in {v} {(contexts or {}).get(i, "at home")}.'),
                      session_id=(sessions or {}).get(i, f's{i}')) for i, v in enumerate(values, 1)]
    observations = {e.event_id: {'speaker': e.speaker, 'text': e.text, 'slot': 'location', 'value': v,
        'contexts': [(contexts or {}).get(i, 'at home')], 'explicit_change': False}
        for i, (e, v) in enumerate(zip(events, values), 1)}
    scopes = {'current': QueryScope('current', ('sam',), ('location',), ('at home',))}
    scopes.update({f'history{i}': QueryScope('as_of', ('sam',), ('location',), ('at home',),
        utc_time(e.timestamp), utc_time(e.timestamp) + 86400) for i, e in enumerate(events, 1)})
    m = ObservedMDLEnergyMemory(CompactConfig(complexity_penalty=.1, switch_margin=.05),
        observations=observations, scopes=scopes, encoder=Encoder(), local_reranker=Reranker(), **kwargs)
    return m, events


def selected(m, query):
    return [h.memory_id for h in m.select_scored(query, list(range(len(m._episode_order))),
        list(range(len(m._episode_order), 0, -1)), 8)]


def test_jeffreys_integrated_code_known_small_cases_and_no_unused_mixture():
    assert categorical_code([1], 1) == pytest.approx(0)
    assert categorical_code([1], 2) == pytest.approx(log(2))
    assert categorical_code([2], 2) == pytest.approx(-log(3 / 8))
    assert categorical_code([1, 1], 2) == pytest.approx(-log(1 / 8))
    with pytest.raises(ValueError): categorical_code([1, 1], 1)
    with pytest.raises(ValueError): transition_candidates([])


def test_minimum_is_exact_over_all_admissible_single_cuts_and_terms_sum():
    for length in range(1, 8):
        for values in product(('a', 'b'), repeat=length):
            plans = transition_candidates(values)
            assert choose_transition(plans)['adjusted_energy'] == min(p['adjusted_energy'] for p in plans)
            for p in plans:
                t = p['terms']
                assert t['total'] == pytest.approx(t['likelihood_code_nats'] + t['segment_code_nats'] + t['boundary_code_nats'])
                if p['cut'] is not None:
                    c = p['cut']; alphabet = len(set(values))
                    from collections import Counter
                    expected = (categorical_code(Counter(values[:c]).values(), alphabet)
                              + categorical_code(Counter(values[c:]).values(), alphabet)
                              + 2 * log(2) + log(length - 1))
                    assert p['adjusted_energy'] == pytest.approx(expected)


def test_noise_does_not_create_version_and_repeated_update_eventually_does():
    m, events = memory(['Cedar'] * 8 + ['Willow'] * 3)
    m.ingest(events[:9])
    assert m.operation_log.all()[-1].operation == 'reinforce'
    assert m.operation_log.all()[-1].stored_as_exception
    m.ingest(events[9:])
    assert any(r.operation == 'supersede' for r in m.operation_log.all())
    old = next(s for s in m.schemas.values() if s.status == 'historical')
    new = m.schemas[old.successor_schema_ids[0]]
    assert old.valid_until == new.valid_from == events[8].timestamp
    assert selected(m, 'history9') == ['e9']
    assert selected(m, 'current') == ['e9', 'e10', 'e11']
    assert selected(m, 'history1') == ['e1']
    assert m.validate_invariants()['grounded_episode_count'] == 11
    np.testing.assert_allclose(m._sums[new.schema_id], sum(m.episodes[e].vector for e in new.support_episode_ids))
    assert all(c.get('optimality_gap', 0) < 1e-9 for c in m.observation_audit())


def test_same_session_spam_is_not_confirmation_and_source_is_immutable():
    values = ['Cedar'] * 8 + ['Willow'] * 3
    m, events = memory(values, sessions={9: 'one', 10: 'one', 11: 'one'})
    m.ingest(events)
    assert not any(r.operation == 'supersede' for r in m.operation_log.all())
    assert m.observation_audit()[-1]['effective_session_observations'] == 9
    assert [m.episodes[e.event_id].text for e in events] == [e.text for e in events]
    m.validate_invariants()


def test_contexts_coexist_without_temporal_overwrite():
    m, events = memory(['Cedar', 'Willow', 'Cedar', 'Willow'], contexts={2: 'at work', 4: 'at work'})
    m.ingest(events)
    assert [r.operation for r in m.operation_log.all()] == ['new', 'split', 'reinforce', 'reinforce']
    assert selected(m, 'current') == ['e1', 'e3']
    assert all(s.status == 'active' for s in m.schemas.values())
    m.validate_invariants()


def test_no_likelihood_cannot_justify_additional_boundaries():
    plans = transition_candidates(['a'] * 8 + ['b'] * 3, likelihood=False)
    assert choose_transition(plans)['action'] == 'reinforce'


def test_existing_two_scalars_are_not_silently_used_by_new_write_energy():
    values = ['Cedar'] * 8 + ['Willow'] * 3
    a, events = memory(values); b, others = memory(values)
    b.config = CompactConfig(complexity_penalty=1, switch_margin=1)
    a.ingest(events); b.ingest(others)
    assert a.observation_audit() == b.observation_audit()
    assert a.telemetry()['decision_scalars_used'] == []


def test_unknown_timestamp_prevents_a_temporal_boundary():
    m, events = memory(['Cedar'] * 8 + ['Willow'] * 3)
    events[8] = replace(events[8], timestamp='unknown')
    m.ingest(events)
    assert not any(r.operation == 'supersede' for r in m.operation_log.all())
    assert not m.observation_audit()[-1]['temporal_order_known']
