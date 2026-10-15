import pytest
from mindset.memory.invariant_readout import registry
from mindset.memory.typed_binding_readout import typed_readout


def setup():
    selected = [dict(event_id='a', speaker='Alice', timestamp='2025-01-01',
                    text='[2025-01-01] | Speaker: Alice\nI lead four engineers. I own five plants.')]
    reg = registry(selected); a, b = [u['id'] for u in reg['units']]
    payload = dict(answer='direct answer', operation='number', operands=[[a]])
    return reg, a, b, payload


def test_host_can_expose_ambiguity_omitted_by_model():
    reg, a, b, p = setup()
    signatures = {u: dict(signature=['same-property']) for u in (a, b)}
    small = typed_readout(p, reg, signatures, expand=False)
    full = typed_readout(p, reg, signatures, expand=True)
    assert small['answer'] == '4' and full['answer'] == 'direct answer'
    assert full['admitted_operands'] == [[a, b]] and full['status'] == 'binding_sensitive'


def test_distinct_property_not_promoted_as_alternative():
    reg, a, b, p = setup()
    signatures = {a: dict(signature=['engineers']), b: dict(signature=['plants'])}
    assert typed_readout(p, reg, signatures, expand=True)['admitted_operands'] == [[a]]


def test_unknown_scope_falls_back_not_invented_binding():
    reg, a, b, p = setup()
    signatures = {a: dict(signature=None), b: dict(signature=['plants'])}
    result = typed_readout(p, reg, signatures, expand=True)
    assert result['answer'] == 'direct answer' and result['typing_status'] == 'unresolved_source_type'


def test_incompatible_declared_alternatives_are_not_pruned_conveniently():
    reg, a, b, p = setup(); p['operands'] = [[a, b]]
    signatures = {a: dict(signature=['engineers']), b: dict(signature=['plants'])}
    result = typed_readout(p, reg, signatures, expand=True)
    assert result['typing_status'] == 'operand_mixes_structural_properties'


def test_every_source_unit_must_have_explicit_known_or_unknown_type():
    reg, a, b, p = setup()
    with pytest.raises(ValueError): typed_readout(p, reg, {a: dict(signature=['a'])}, expand=True)
