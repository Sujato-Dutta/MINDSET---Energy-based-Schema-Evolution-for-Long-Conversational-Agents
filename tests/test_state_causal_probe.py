from dataclasses import replace
import numpy as np
import pytest
from mindset.acl.state_worlds import worlds
from mindset.methods.state_causal_probe import (
    ObservedStateMemory, ConfirmationReplayMemory, LatestJournalMemory)
from mindset.methods.state_readout import QueryScope
from mindset.methods.compact import CompactConfig
from mindset.methods.joint import utc_time
from test_compact import Encoder, Reranker, event


def fixture(values, *, cls=ConfirmationReplayMemory, cues=(), sessions=None,
            contexts=None, **kwargs):
    events = [replace(event(i, f'I {"moved to" if i in cues else "live in"} {value} {(contexts or {}).get(i, "at home") }.'),
                      session_id=(sessions or {}).get(i, f's{i}'))
              for i, value in enumerate(values, 1)]
    observations = {e.event_id: {'speaker': e.speaker, 'text': e.text,
        'slot': 'location', 'value': value, 'contexts': [(contexts or {}).get(i, 'at home')],
        'explicit_change': i in cues} for i, (e, value) in enumerate(zip(events, values), 1)}
    scopes = {'current': QueryScope('current', ('sam',), ('location',), ('at home',))}
    scopes.update({f'history{i}': QueryScope('as_of', ('sam',), ('location',), ('at home',),
                  utc_time(e.timestamp), utc_time(e.timestamp) + 86400)
                  for i, e in enumerate(events, 1)})
    m = cls(CompactConfig(complexity_penalty=.1, switch_margin=.05),
            observations=observations, scopes=scopes, encoder=Encoder(), local_reranker=Reranker(), **kwargs)
    return m, events


def selected(m, query):
    return [h.memory_id for h in m.select_scored(query, list(range(len(m._episode_order))),
            list(range(len(m._episode_order), 0, -1)), 8)]


def similar_values():
    prefix = ' '.join(f'sector{i}' for i in range(16))
    return prefix + ' Cedar', prefix + ' Willow'


def test_deferred_cost_changes_choice_and_replays_first_support():
    a, b = similar_values()
    m, events = fixture([a, b, b])
    m.ingest(events[:2])
    assert m.operation_log.all()[-1].stored_as_exception
    m.ingest(events[2:])
    assert m.operation_log.all()[-1].operation == 'supersede'
    old = next(s for s in m.schemas.values() if s.status == 'historical')
    new = m.schemas[old.successor_schema_ids[0]]
    assert new.valid_from == old.valid_until == events[1].timestamp
    assert set(new.support_episode_ids) == {'e2', 'e3'}
    assert selected(m, 'history2') == ['e2']
    assert selected(m, 'current') == ['e2', 'e3']
    certificate = m.observation_audit()[-1]
    version = next(p for p in certificate['feasible_candidates'] if p['action'] == 'supersede')
    assert version['terms']['contradiction'] < 0  # signed potential delta, not probability
    assert version['terms']['distortion'] == pytest.approx(new.mean_distortion, abs=1e-6)
    assert certificate['optimality_gap'] == 0
    assert certificate['confirmation_replay']['decision_time'] == events[2].timestamp
    np.testing.assert_allclose(m._sums[new.schema_id], sum(m.episodes[e].vector for e in new.support_episode_ids))
    assert m.validate_invariants()['grounded_episode_count'] == 3
    assert [m.episodes[e.event_id].text for e in events] == [e.text for e in events]


@pytest.mark.parametrize('cls,kwargs', [(ObservedStateMemory, {}),
                                      (ConfirmationReplayMemory, {'contradiction_debt': False})])
def test_recurrence_eligibility_alone_does_not_remove_weak_contradiction_deadlock(cls, kwargs):
    a, b = similar_values()
    m, events = fixture([a, b, b], cls=cls, **kwargs)
    m.ingest(events)
    assert m.operation_log.all()[-1].signals['recurrence']
    assert m.operation_log.all()[-1].stored_as_exception
    assert all(s.status == 'active' for s in m.schemas.values())


def test_confirmation_replay_without_cost_change_can_fix_history_for_rule_control():
    m, events = fixture(['Cedar', 'Willow', 'Willow'], write_policy='rule')
    m.ingest(events)
    assert m.operation_log.all()[-1].operation == 'supersede'
    assert selected(m, 'history2') == ['e2']
    m.validate_invariants()


def test_old_noise_before_reset_support_cannot_confirm_later_report():
    m, events = fixture(['Cedar', 'Willow', 'Cedar', 'Willow'])
    m.ingest(events)
    assert m.operation_log.all()[-1].signals['recurrence'] == 0
    assert m.operation_log.all()[-1].stored_as_exception
    assert len(m.schemas) == 1


def test_same_session_repetition_is_not_independent_confirmation():
    m, events = fixture(['Cedar', 'Willow', 'Willow'], sessions={2: 'same', 3: 'same'})
    m.ingest(events)
    assert m.operation_log.all()[-1].signals['recurrence'] == 0
    assert m.operation_log.all()[-1].stored_as_exception


def test_contextual_coexistence_splits_instead_of_replaying_across_contexts():
    m, events = fixture(['Cedar', 'Willow', 'Willow'], contexts={2: 'at work', 3: 'at work'})
    m.ingest(events)
    assert [r.operation for r in m.operation_log.all()] == ['new', 'split', 'reinforce']
    assert selected(m, 'current') == ['e1']
    assert all(s.status == 'active' for s in m.schemas.values())
    m.validate_invariants()


def test_latest_journal_is_history_preserving_not_latest_text_only():
    m, events = fixture(['Cedar', 'Willow', 'Willow'], cls=LatestJournalMemory)
    m.ingest(events)
    assert selected(m, 'history1') == ['e1']
    assert selected(m, 'current') == ['e2', 'e3']
    assert selected(m, 'history2') == ['e2']  # later same-valued source is still future


def test_unknown_incoming_time_does_not_confirm_or_backdate():
    m, events = fixture(['Cedar', 'Willow', 'Willow'])
    events[2] = replace(events[2], timestamp='unknown')
    m.ingest(events)
    assert m.operation_log.all()[-1].signals['temporal_potential'] == 0
    assert m.operation_log.all()[-1].stored_as_exception
    assert len(m.schemas) == 1


def test_pending_future_report_is_not_replayed_into_an_earlier_confirmation():
    m, events = fixture(['Cedar', 'Willow', 'Willow'])
    events[1] = replace(events[1], timestamp='2025-02-01')
    m.ingest(events)
    assert m.operation_log.all()[-1].signals['recurrence'] == 0
    assert m.operation_log.all()[-1].stored_as_exception
    m.validate_invariants()


def test_source_trace_cannot_identify_true_change_versus_correlated_noise():
    corpus = worlds()
    for words in (0, 4, 16):
        for distractors in (0, 64):
            pair = [next(h for h in corpus if h['family'] == family and h['shared_words'] == words
                         and h['distractors'] == distractors) for family in ('confirmation', 'correlated_noise')]
            trace = lambda h: [(e['conversation_id'], e['timestamp'], e['speaker'], e['text'], e['session_id'], e['observation']) for e in h['events']]
            assert trace(pair[0]) == trace(pair[1])
            assert pair[0]['truth_timeline'][-1] != pair[1]['truth_timeline'][-1]
    a, b = similar_values()
    first, events = fixture([a, b, b])
    second, other = fixture([a, b, b])
    first.ingest(events); second.ingest(other)
    assert first.observation_audit() == second.observation_audit()
    assert selected(first, 'current') == selected(second, 'current')


def test_unverifiable_controlled_annotations_are_rejected():
    m, events = fixture(['Cedar'])
    m.observations['e1']['value'] = 'Not in source'
    with pytest.raises(ValueError, match='source-grounded'): m.ingest(events)
