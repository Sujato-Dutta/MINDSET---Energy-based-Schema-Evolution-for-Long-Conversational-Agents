from copy import deepcopy
import pytest
from mindset.memory.invariant_readout import registry, validate_registry, readout, messages, INSTRUCTION


def passage(text, i=1, stamp='12:00 pm on 28 August, 2023'):
    return dict(event_id=str(i), speaker='Alice', timestamp=stamp,
                text=f'[{stamp}] | Speaker: Alice\n{text}', token_start=0, token_end=50)


def program(answer='fallback', operation='number', operands=None):
    return dict(answer=answer, operation=operation, operands=operands or [])


def numeric(reg):
    return [u['id'] for u in reg['units'] if u['kind'] == 'number']


def test_exact_source_registry_rejects_forged_value_or_source():
    selected = [passage('I lead four engineers, now five.')]; reg = registry(selected)
    validate_registry(reg, selected)
    bad = deepcopy(reg); bad['units'][0]['values'] = ['999']
    with pytest.raises(ValueError): validate_registry(bad, selected)
    with pytest.raises(ValueError): validate_registry(reg, [passage('I lead six engineers.')])


def test_relative_date_uses_source_time_and_natural_precision():
    reg = registry([passage('I bought it yesterday.')]); uid = reg['units'][0]['id']
    out = readout(program(operation='date', operands=[[uid]]), reg)
    assert out['answer'] == '27 August 2023' and out['status'] == 'binding_invariant'
    reg = registry([passage('I bought it last month.')]); uid = reg['units'][0]['id']
    assert readout(program(operation='date', operands=[[uid]]), reg)['answer'] == 'July 2023'


def test_headers_are_not_numeric_operands_and_missing_year_stays_unknown():
    reg = registry([passage('No explicit numeric quantity.')]); assert not reg['units']
    reg = registry([passage('I left yesterday.', stamp='28 August')]); assert not reg['units']


def test_binding_agreement_is_not_first_binding_execution():
    reg = registry([passage('I lead four engineers; another report says five.')]); a, b = numeric(reg)
    out = readout(program(answer='uncertain', operands=[[a, b]]), reg)
    assert out['answer'] == 'uncertain' and out['first_binding_answer'] == '4'
    assert out['status'] == 'binding_sensitive' and out['worlds'] == 2


def test_same_value_different_surface_bindings_are_invariant():
    reg = registry([passage('There are five engineers, exactly 5.')]); a, b = numeric(reg)
    out = readout(program(operands=[[a, b]]), reg)
    assert out['answer'] == '5' and out['status'] == 'binding_invariant'


def test_arithmetic_groups_and_reversed_direction_are_distinct():
    reg = registry([passage('I originally led four people, now five.')]); a, b = numeric(reg)
    assert readout(program(operation='difference', operands=[[b], [a]]), reg)['answer'] == '1'
    assert readout(program(operation='difference', operands=[[a], [b]]), reg)['answer'] == '-1'
    assert readout(program(operation='sum', operands=[[a], [b]]), reg)['answer'] == '9'


def test_forged_unused_alternative_and_wrong_type_are_rejected():
    reg = registry([passage('I led four people yesterday.')]); a = numeric(reg)[0]
    with pytest.raises(ValueError): readout(program(operands=[[a, 'fake']]), reg)
    with pytest.raises(ValueError): readout(program(operation='date', operands=[[a]]), reg)
    with pytest.raises(ValueError): readout(program(operation='direct', operands=[[a]]), reg)


def test_budget_fails_closed_without_silent_world_trimming():
    reg = registry([passage('There are four or five.')]); a, b = numeric(reg)
    out = readout(program(operation='sum', operands=[[a, b]] * 8), reg, max_worlds=100)
    assert out['answer'] == 'fallback' and out['status'] == 'enumeration_budget_exceeded'
    assert out['worlds'] == 256


def test_explicit_date_not_decomposed_into_fake_count_operands():
    reg = registry([passage('On 21 October 2023 I bought 3 figurines.')])
    assert len(numeric(reg)) == 1
    d = next(u['id'] for u in reg['units'] if u['kind'] == 'date')
    assert readout(program(operation='date', operands=[[d]]), reg)['answer'] == '21 October 2023'


def test_invalid_arity_and_payload_never_execute_generated_code():
    reg = registry([passage('I own two bicycles.')]); a = numeric(reg)[0]
    for op, args in [('difference', [[a]]), ('number', [[a], [a]]), ('__import__', [[a]])]:
        with pytest.raises(ValueError): readout(program(operation=op, operands=args), reg)
    with pytest.raises(ValueError): readout({'answer': 'x'}, reg)


def test_full_cartesian_counterexample_not_just_diagonal_samples():
    reg = registry([passage('The values are four and five.')]); a, b = numeric(reg)
    # Diagonal subtraction would always be zero. Cross bindings are not equal.
    out = readout(program(operation='difference', operands=[[a, b], [a, b]]), reg)
    assert out['outcomes'] == ['-1', '0', '1'] and out['status'] == 'binding_sensitive'


def test_literal_registry_does_not_call_different_values_semantically_comparable():
    reg = registry([passage('I own four plants and five bicycles.')])
    assert not reg['semantic_truth_verified']
    assert all('not a certified' in u['interpretation'] for u in reg['units'])


def test_locomo_prompt_stays_byte_identical_during_template_fix():
    import json
    original = [dict(role='user', content='evidence and question')]
    reg = registry([passage('No quantity.')])
    expected = [dict(role='system', content=INSTRUCTION), dict(role='user',
        content='evidence and question\n\nOptional source-unit inventory (P is evidence passage number):\n' + json.dumps([]))]
    assert messages(original, reg) == expected
    assert original == [dict(role='user', content='evidence and question')]


def test_existing_system_prompt_is_merged_without_invalid_midstream_system():
    original = [dict(role='system', content='original answer instructions'),
                dict(role='user', content='evidence and question')]
    prompt = messages(original, registry([passage('No quantity.')]))
    assert [m['role'] for m in prompt] == ['system', 'user']
    assert original[0]['content'] in prompt[0]['content']
    assert original[0]['content'] == 'original answer instructions'
