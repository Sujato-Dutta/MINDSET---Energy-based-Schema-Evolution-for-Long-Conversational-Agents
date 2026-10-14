from types import SimpleNamespace
import pytest

from mindset.acl.source_relation_candidates import measured_pairs
from mindset.acl.source_relation_validation import verify_prepared_pair
from mindset.methods.semantic_witness import Span


def sources():
    events = [SimpleNamespace(event_id=eid, text=text, speaker='User', timestamp=stamp)
        for eid, text, stamp in (('a', 'I own a violin.', '2024-01-01'),
            ('b', 'I do not own a violin.', '2024-03-01'), ('c', 'I also own a cello.', '2024-02-01'))]
    spans = [Span(e.event_id, 0, len(e.text), e.text, 'user') for e in events]
    return events, spans


def test_one_closest_pair_per_seed_and_lossless_negation():
    events, spans = sources()
    result = measured_pairs([(0, 1), (0, 2), (2, 1)], spans, events)
    assert len(result) == 2 and result[0]['reports'][1]['text'] == 'I do not own a violin.'
    assert result[0]['source_spans'][1]['episode_id'] == 'b'
    assert not any(k in result[0] for k in ('answer', 'gold', 'category', 'question'))
    assert len(measured_pairs([(0, 1), (2, 1)], spans, events, anchor_limit=1)) == 1


def test_future_and_corrupted_source_spans_rejected():
    events, spans = sources()
    with pytest.raises(ValueError, match='future'):
        measured_pairs([(0, 1)], spans, events, question_date='2024-02-01')
    spans[1] = Span('b', 0, len(events[1].text), 'I own a violin.', 'user')
    with pytest.raises(ValueError, match='literal'):
        measured_pairs([(0, 1)], spans, events)


def test_author_and_duplicate_episode_checks():
    events, spans = sources()
    with pytest.raises(ValueError, match='Duplicate'):
        measured_pairs([(0, 1)], spans, [*events, events[0]])
    events[1].speaker = 'Other'
    with pytest.raises(ValueError, match='author'):
        measured_pairs([(0, 1)], spans, events)


def test_no_admissible_pair_is_retained_as_empty_not_invented():
    events, spans = sources()
    assert measured_pairs([], spans, events) == []
    with pytest.raises(ValueError):
        measured_pairs([], spans, events, anchor_limit=0)


def test_prepared_pair_revalidated_against_source_not_just_self_hash():
    from copy import deepcopy
    events, spans = sources(); pair = measured_pairs([(0, 1)], spans, events)[0]
    assert verify_prepared_pair(pair, events)
    changed = deepcopy(pair); changed['reports'][1]['observation_time'] = '2030-01-01'
    with pytest.raises(ValueError, match='metadata'):
        verify_prepared_pair(changed, events)
    changed = deepcopy(pair); changed['source_spans'][0]['char_start'] = True
    with pytest.raises(ValueError, match='Integer'):
        verify_prepared_pair(changed, events)
