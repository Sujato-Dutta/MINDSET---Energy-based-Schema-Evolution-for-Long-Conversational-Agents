import json
from copy import deepcopy

import pytest

from mindset.memory.scope_instrument_controls import (
    ARMS, control_messages, output_schema, validate_control, summarize_controls)


def test_same_sources_but_different_output_costs():
    bodies = [json.loads(control_messages(a, 'I like jazz.', 'I like rock.')[1]['content']) for a in ARMS]
    assert bodies[0]['reports'] == bodies[1]['reports']
    assert set(output_schema('relation_only')['properties']) == {'relationship'}
    assert 'frames' in output_schema('grounded_frames')['properties']
    assert 'expected_contradiction' not in str(bodies)


def test_relation_only_is_not_a_semantic_or_provenance_certificate():
    result = validate_control('relation_only', {'relationship': 'compatible'}, 'a', 'b')
    assert not any(result[k] for k in ('literal_provenance_verified',
        'semantic_correctness_verified', 'world_truth_or_update_verified'))
    for payload in ({'relationship': 'true'}, {'relationship': 'compatible', 'frames': []}, []):
        with pytest.raises(ValueError):
            validate_control('relation_only', payload, 'a', 'b')


def test_metadata_quote_rejected_even_with_correction():
    frame = {'source_id': 'A', 'holder_quote': 'I', 'predicate_quote': 'like', 'value_quote': 'jazz',
        'context_quotes': [], 'time_quotes': ['2025-01-01'],
        'canonical_relation': 'preference', 'modality': 'asserted', 'value_mode': 'set_member'}
    other = deepcopy(frame); other['source_id'] = 'B'; other['value_quote'] = 'rock'
    with pytest.raises(ValueError, match='quote'):
        validate_control('grounded_frames', {'frames': [frame, other], 'relationship': 'compatible'},
            'I like jazz.', 'I like rock.')
    prompt = control_messages('grounded_frames', 'I like jazz.', 'I like rock.')
    assert 'Never copy' in prompt[0]['content']


def test_rejections_not_hidden_from_error_accounting():
    def row(rid, expected, prediction, error):
        return {'id': rid, 'arm': 'test', 'split': 'development', 'expected_contradiction': expected,
            'prediction': {'relationship': prediction}, 'validation_error': error,
            'input_tokens': 10, 'output_tokens': 2, 'cache_hit': False}
    result = summarize_controls([row('a', False, 'incompatible', 'quote error'),
        row('b', True, 'incompatible', 'quote error'), row('c', False, 'compatible', None)])
    group = result['groups']['test/development']
    assert group['cases'] == 3 and group['valid_payloads'] == 1
    assert group['accepted_false_conflicts'] == 0
    assert group['raw_false_conflicts_including_rejected'] == 1
    assert group['missed_or_invalid_conflicts'] == 1
    assert group['invalid_compatible_cases'] == 1
    assert group['logical_output_tokens'] == 6


def test_unknown_arm_rejected():
    with pytest.raises(ValueError):
        control_messages('new_unvalidated_arm', 'a', 'b')
