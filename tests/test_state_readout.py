import numpy as np
import pytest
from mindset.methods.compact import CompactConfig
from mindset.methods.state_readout import StateAwareMindsetMemory, explicit_period, query_scope
from test_compact import Encoder, Reranker, event


def memory(events, **kwargs):
    m = StateAwareMindsetMemory(CompactConfig(complexity_penalty=.1, switch_margin=.05),
                                encoder=Encoder(), local_reranker=Reranker(), **kwargs)
    m.ingest(events)
    return m


def select(m, query, **kwargs):
    return [h.memory_id for h in m.select_scored(query, list(range(len(m._episode_order))),
                                              np.arange(len(m._episode_order), 0, -1), **kwargs)]


def test_current_state_suppresses_stale_support_not_raw_history():
    m = memory([event(1, 'I live in Paris.'), event(2, 'I moved to Rome.')])
    assert select(m, 'Where does Sam live now?') == ['e2']
    assert 'historical_schema' in m.last_readout_certificate['excluded']['e1']
    assert select(m, 'When did Sam move?') == ['e1', 'e2']
    assert m.validate_invariants()['grounded_episode_count'] == 2


def test_as_of_restores_predecessor_and_blocks_future_observations():
    m = memory([event(1, 'I live in Paris.'), event(2, 'I moved to Rome.')])
    assert select(m, 'Where did Sam live on 2025-01-01?') == ['e1']
    assert select(m, 'Where did Sam live on 2025-01-02?') == ['e2']
    # A whole-month request can legitimately overlap both regimes.
    assert select(m, 'Where did Sam live in January 2025?') == ['e1', 'e2']


def test_context_selection_preserves_coexisting_branches():
    m = memory([event(1, 'I prefer tea at home.'), event(2, 'I prefer coffee at work.')])
    assert select(m, 'What does Sam prefer at work?') == ['e2']
    assert select(m, 'What does Sam prefer at home?') == ['e1']
    assert select(m, 'What does Sam prefer?') == ['e1', 'e2']


def test_noise_does_not_override_support_but_unwitnessed_pool_is_not_empty():
    m = memory([event(1, 'I live in Paris.'), event(2, 'I live in Rome.')])
    assert select(m, 'Where does Sam live?') == ['e1']
    hits = m.select_scored('Where does Sam live?', [1], [100.])
    assert [h.memory_id for h in hits] == ['e2']
    assert not m.last_readout_certificate['excluded']


def test_state_off_unknown_owner_predicate_and_past_fall_back_exactly():
    events = [event(1, 'I live in Paris.'), event(2, 'I moved to Rome.')]
    m = memory(events, state_readout=False)
    assert select(m, 'Where does Sam live now?') == ['e1', 'e2']
    m = memory(events)
    for q in ('Where does Alex live?', 'What does Sam enjoy?', 'Where did Sam live previously?',
              'Where did Sam live in 2024 and 2025?'):
        assert select(m, q) == ['e1', 'e2']


def test_period_precision_invalid_dates_and_event_routing():
    start, end = explicit_period('in 2024')
    assert end - start == 366 * 86400
    assert explicit_period('on 2025-02-31') == (0., 0.)
    assert explicit_period('2024 and 2025') == (0., 0.)
    assert query_scope('When did Sam move in 2025?', ['Sam']).mode == 'event'
    assert query_scope('Where does Sam live?', ['Sam'], '2025-01-01').period_start > 0


def test_single_ce_pass_raw_provenance_and_no_read_fitted_weights():
    m = memory([event(i, f'Tea item {i}') for i in range(1, 61)])
    hits = m.retrieve('Tea facts', 8)
    assert m.local_reranker.calls == [48]
    assert len(hits) == 8 and len(m.last_candidate_pool) == 48
    assert all(h.source_event_ids == (h.memory_id,) for h in hits)
    assert m.telemetry()['new_fitted_read_scalars'] == 0


def test_score_validation():
    m = memory([event(1, 'I live in Paris.')])
    with pytest.raises(ValueError): m.select_scored('x', [0], [float('nan')])
    with pytest.raises(ValueError): m.select_scored('x', [0, 0], [1., 1.])
