import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('scope_control_costs',
    Path(__file__).resolve().parents[1] / 'scripts/audit_scope_control_costs.py')
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def test_quantiles_and_bad_samples():
    assert worker.percentile([1, 3], .5) == 2
    assert worker.percentile([1, 3], .95) == pytest.approx(2.9)
    for values in ([], [-1], [float('nan')]):
        with pytest.raises(ValueError):
            worker.percentile(values, .5)


def test_fresh_invalid_requests_count_but_cache_zero_does_not():
    row = {'id': 'x', 'arm': 'relation_only', 'request_id': 'r', 'cache_hit': False,
        'input_tokens': 3, 'output_tokens': 1, 'raw_response': '{}', 'incomplete': False,
        'validation_error': 'bad output'}
    payload = {'request_id': 'r', 'text': '{}', 'input_tokens': 3, 'output_tokens': 1,
               'incomplete': False, 'latency_seconds': 2}
    report = worker.cost_rows([row, {**row, 'id': 'y', 'cache_hit': True}], [payload])
    group = report['groups']['relation_only']
    assert group['fresh_requests'] == 1 and group['invalid_or_incomplete_not_dropped'] == 1
    assert group['request_p50_ms'] == 2000
    with pytest.raises(ValueError, match='differs'):
        worker.cost_rows([row], [{**payload, 'text': 'different'}])


def test_missing_and_conflicting_cache_payloads_fail_closed():
    row = {'id': 'x', 'arm': 'x', 'request_id': 'r', 'cache_hit': False}
    with pytest.raises(ValueError, match='Missing'):
        worker.cost_rows([row], [])
    with pytest.raises(ValueError, match='Nonunique'):
        worker.cost_rows([], [{'request_id': 'r', 'text': 'a'}, {'request_id': 'r', 'text': 'b'}])
