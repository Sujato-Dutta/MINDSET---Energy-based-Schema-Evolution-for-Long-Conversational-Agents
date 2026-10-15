"""Host-admitted source-binding alternatives for the executable readout pilot.

Structural signatures are conservative parser hypotheses, not proven semantic
equivalence. This prevents the model's singleton declaration from being called
an exhaustive uncertainty set. Guarantees remain conditional on these signatures.
"""
from copy import deepcopy
from mindset.memory.invariant_readout import readout, validate_program


def typed_readout(payload, reg, signatures, *, expand):
    validate_program(payload, reg)
    if payload['operation'] == 'direct':
        return {**readout(payload, reg), 'admitted_operands': [], 'typing_status': 'direct'}
    units = {u['id']: u for u in reg['units']}
    if set(signatures) != set(units):
        raise ValueError('Every source unit needs an explicit known/unknown signature')
    groups = []; reasons = []
    for group in payload['operands']:
        keys = [signatures[uid]['signature'] for uid in group]
        # Mixing known property signatures is not repaired by discarding the
        # inconvenient alternative. Reject the attempted typed execution.
        if any(k is None for k in keys):
            reasons.append('unresolved_source_type')
        elif any(k != keys[0] for k in keys):
            reasons.append('operand_mixes_structural_properties')
        if reasons:
            break
        candidates = list(group)
        if expand:
            candidates += [uid for uid in units if uid not in candidates
                           and units[uid]['kind'] == units[group[0]]['kind']
                           and signatures[uid]['signature'] == keys[0]]
        groups.append(candidates)
    if reasons:
        return dict(answer=payload['answer'], first_binding_answer=payload['answer'],
            replaced=False, status='typing_fallback', worlds=0, outcomes=[], guarantee=None,
            admitted_operands=[], typing_status=reasons[0])
    # The inference schema's 16-ID generation cap is not a reason to silently
    # trim host uncertainty. Larger classes fail closed to the direct answer.
    if any(len(group) > 16 for group in groups):
        return dict(answer=payload['answer'], first_binding_answer=payload['answer'],
            replaced=False, status='typing_fallback', worlds=0, outcomes=[], guarantee=None,
            admitted_operands=groups, typing_status='host_alternative_budget_exceeded')
    expanded = deepcopy(payload); expanded['operands'] = groups
    result = readout(expanded, reg)
    result.update(admitted_operands=groups, typing_status='source_structural_signature_only',
        semantic_binding_completeness_verified=False)
    return result
