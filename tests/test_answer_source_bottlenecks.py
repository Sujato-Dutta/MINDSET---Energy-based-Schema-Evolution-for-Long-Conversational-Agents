import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

spec = importlib.util.spec_from_file_location('answer_source_bottlenecks',
    Path(__file__).resolve().parents[1] / 'scripts/audit_answer_source_bottlenecks.py')
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text)
    def decode(self, tokens):
        return ''.join(tokens)


def event(eid):
    return SimpleNamespace(event_id=eid, session_id='s', timestamp='2024-01-01', speaker='User', text='I like jazz.')


def selected(e, length=None):
    tokens = Tokenizer().encode(worker.render_event(e)); end = len(tokens) if length is None else length
    return {'event_id': e.event_id, 'token_start': 0, 'token_end': end, 'text': ''.join(tokens[:end])}


def test_interval_gaps_not_complete():
    assert worker.complete_coverage([(0, 3), (3, 5)], 5)
    assert not worker.complete_coverage([(0, 2), (3, 5)], 5)
    with pytest.raises(ValueError):
        worker.complete_coverage([(0, 6)], 5)


def test_coarse_session_coverage_not_full_raw_content():
    events = [event('a'), event('b')]
    report = worker.source_coverage(events, [selected(events[0])], ['s'], 'session', Tokenizer())
    assert report['all_gold_units_retrieved'] is True
    assert report['all_annotated_raw_event_text_visible'] is False
    assert report['gold_raw_event_count'] == 2


def test_full_episode_and_partial_chunk_distinguished():
    e = event('a')
    full = worker.source_coverage([e], [selected(e)], ['a'], 'episode', Tokenizer())
    partial = worker.source_coverage([e], [selected(e, 4)], ['a'], 'episode', Tokenizer())
    assert full['all_annotated_raw_event_text_visible'] is True
    assert partial['all_gold_units_retrieved'] is True and partial['all_annotated_raw_event_text_visible'] is False


def test_absent_gold_and_source_corruption_not_hidden():
    e = event('a')
    assert worker.source_coverage([e], [], [], 'episode', Tokenizer())['all_gold_units_retrieved'] is None
    wrong = selected(e); wrong['text'] = 'fabricated'
    with pytest.raises(ValueError, match='source tokens'):
        worker.source_coverage([e], [wrong], ['a'], 'episode', Tokenizer())
