from types import SimpleNamespace as N
import pytest
from mindset.memory.witness_closure import (calendar_constraints, validate_calendar, dependency_closure,
                                            render_packet, select_packet)


@pytest.mark.parametrize('expression,anchor,start,end,precision', [
    ('yesterday', '2025-01-01', '2024-12-31', '2025-01-01', 'day'),
    ('today', '2024-02-29', '2024-02-29', '2024-03-01', 'day'),
    ('tomorrow', '2024-02-29', '2024-03-01', '2024-03-02', 'day'),
    ('last Friday', '2025-01-03', '2024-12-27', '2024-12-28', 'day'),
    ('last month', '2025-01-17', '2024-12-01', '2025-01-01', 'month'),
    ('last year', '2025-02-01', '2024-01-01', '2025-01-01', 'year'),
])
def test_calendar_is_source_anchored_not_machine_today(expression, anchor, start, end, precision):
    text = 'A report refers to ' + expression + '.'
    note = calendar_constraints('s', text, anchor)[0]
    assert (note['start'], note['end'], note['precision']) == (start, end, precision)
    validate_calendar(note, N(event_id='s', text=text, timestamp=anchor))
    with pytest.raises(ValueError):
        validate_calendar({**note, 'start': '1999-01-01'}, N(event_id='s', text=text, timestamp=anchor))


def test_invalid_date_stays_unknown_and_non_source_note_rejected():
    assert calendar_constraints('s', 'Yesterday was pleasant.', 'not a time') == []
    assert calendar_constraints('s', 'Yesterday was pleasant.', '3:30 pm') == []
    with pytest.raises(ValueError):
        render_packet([{'event_id': 'a', 'text': 'source'}],
                      [dict(event_id='b', precision='day', start='2025-01-01', quote='today')])


def test_dependency_closure_handles_shared_ancestors_and_rejects_cycles():
    assert dependency_closure(['c', 'd'], {'c': ['b'], 'd': ['b'], 'b': ['a']}, set('abcd')) == set('abcd')
    with pytest.raises(ValueError):
        dependency_closure(['a'], {'a': ['b'], 'b': ['a']}, set('ab'))
    with pytest.raises(KeyError):
        dependency_closure(['b'], {'b': ['a']}, {'b'})


def test_budgeted_solver_counts_shared_context_once_and_is_closed():
    passages = {x: {'event_id': x, 'text': x} for x in 'abcd'}
    roots = ['c', 'd']; deps = {'c': ['a'], 'd': ['a']}
    selected, packet, audit = select_packet(roots, passages, deps, {}, len, 100,
        {x: i for i, x in enumerate('abcd')}, max_sources=3)
    assert [s['event_id'] for s in selected] == ['a', 'c', 'd']
    assert audit['retained_rank_mass'] == '3/2'
    assert audit['closure_supported']


def test_closure_not_silently_truncated_when_budget_too_small():
    passages = {x: {'event_id': x, 'text': x} for x in 'abc'}
    selected, _, audit = select_packet(['c'], passages, {'c': ['b'], 'b': ['a']}, {}, len, 100,
        {x: i for i, x in enumerate('abc')}, max_sources=2)
    assert [s['event_id'] for s in selected] == ['c']
    assert audit['closure_supported'] is False
    assert audit['fallback'] == 'no_nonempty_closed_packet_within_budget'


def test_calendar_annotation_is_in_the_token_budget():
    passages = {'a': {'event_id': 'a', 'text': 'Today.'}, 'b': {'event_id': 'b', 'text': 'plain'}}
    notes = {'a': calendar_constraints('a', 'Today.', '2025-01-01')}
    selected, packet, audit = select_packet(['a', 'b'], passages, {}, notes, len, 30,
        {'a': 0, 'b': 1}, close=False, calendar=True)
    assert [s['event_id'] for s in selected] == ['b']
    assert len(packet) <= 30


def test_reconstruction_worker_entrypoints_parse():
    import ast
    from pathlib import Path
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    for name in ('prepare_witness_closure.py', 'parse_witness_context.py', 'answer_witness_closure.py'):
        ast.parse((scripts / name).read_text(encoding='utf-8'), filename=name)


def test_exact_root_search_matches_independent_all_source_subset_oracle():
    from fractions import Fraction
    import random
    rng = random.Random(20261015)
    names = list('abcdef'); roots = ['f', 'd', 'b']
    passages = {eid: {'event_id': eid, 'text': eid} for eid in names}
    order = {eid: i for i, eid in enumerate(names)}
    for _ in range(30):
        dependencies = {names[i]: [names[rng.randrange(i)]] for i in range(1, 6) if rng.random() < .6}
        oracle = []
        for mask in range(1, 1 << len(names)):
            selected = {eid for i, eid in enumerate(names) if mask & (1 << i)}
            if len(selected) > 3 or not selected.intersection(roots):
                continue
            if any(not set(dependencies.get(eid, ())).issubset(selected) for eid in selected):
                continue
            mass = sum((Fraction(1, i + 1) for i, eid in enumerate(roots) if eid in selected), Fraction())
            oracle.append((-mass, len(selected)))
        chosen, _, audit = select_packet(roots, passages, dependencies, {}, len, 1000, order, max_sources=3)
        if not oracle:
            assert audit['closure_supported'] is False
        else:
            assert (-Fraction(audit['retained_rank_mass']), len(chosen)) == min(oracle)
