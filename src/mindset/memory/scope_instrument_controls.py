"""Isolated scope-instrument controls. No memory updates or semantic certificates.

The relationship-only arm tests whether generated frames earn their inference
cost. The frame arm fixes observation metadata being copied into literal quotes.
Neither arm establishes world truth, energy usefulness, or novel memory dynamics.
"""
import json

from mindset.memory.source_scope_contract import SCHEMA, SYSTEM, validate_measurement
from mindset.utils.hashing import stable_hash

POLICY = """Measure whether two reports can both be true, not the truth of the world.
Respect the reported holder, property, object, context, factual time and modality.
Different liked objects, contexts, years, holders or compatible plans are not
contradictions. Do not infer exclusive values without support. Same-scope
positive/negative assertions or incompatible explicit exact counts can conflict.
Return underdetermined when the reports do not resolve compatibility.
Observation time is provenance metadata, NOT an assertion that every fact holds
at that time. A stated factual time or recurring context takes precedence.
Do not generate a summary, answer a benchmark question, or update memory."""

RELATION_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {'relationship': SCHEMA['properties']['relationship']},
    'required': ['relationship'],
}
FRAME_CORRECTION = """The reports' observation_time field is metadata only. Never copy
it into any quote field unless those exact characters occur in that source's text.
time_quotes contains only literal text spans stating factual time. If the source
has no such span, return []. Unknown fields stay empty; do not substitute metadata.
A valid quote does not by itself verify a correct holder/property assignment."""
ARMS = ('relation_only', 'grounded_frames')


def output_schema(arm):
    if arm not in ARMS:
        raise ValueError('Unknown scope control arm')
    return RELATION_SCHEMA if arm == 'relation_only' else SCHEMA


def control_messages(arm, left, right):
    schema = output_schema(arm)
    system = POLICY + '\nReturn exactly the supplied JSON schema.'
    if arm == 'grounded_frames':
        system += '\n' + SYSTEM + '\n' + FRAME_CORRECTION
    # Identical source text and observation metadata across arms and r01.
    reports = [{'source_id': sid, 'speaker': 'user',
                'observation_time': '2025-01-01', 'text': text}
               for sid, text in (('A', left), ('B', right))]
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': json.dumps(
                {'reports': reports, 'output_schema': schema}, ensure_ascii=False)}]


def validate_control(arm, payload, left, right):
    if arm == 'grounded_frames':
        return validate_measurement(payload, left, right)
    output_schema(arm)
    if not isinstance(payload, dict) or set(payload) != {'relationship'}:
        raise ValueError('Relationship-only output required')
    if payload['relationship'] not in RELATION_SCHEMA['properties']['relationship']['enum']:
        raise ValueError('Unknown relationship')
    return {'literal_provenance_verified': False, 'semantic_correctness_verified': False,
            'world_truth_or_update_verified': False, 'scope': 'provisional pair label only'}


def summarize_controls(rows):
    """Keep rejects visible; accepted-only errors are NOT overall accuracy."""
    groups = {}
    for arm, split in sorted({(r['arm'], r['split']) for r in rows}):
        subset = [r for r in rows if (r['arm'], r['split']) == (arm, split)]
        valid = [r for r in subset if r['validation_error'] is None]
        def incompatible(r):
            return isinstance(r.get('prediction'), dict) and r['prediction'].get('relationship') == 'incompatible'
        groups[f'{arm}/{split}'] = {
            'cases': len(subset), 'valid_payloads': len(valid),
            'invalid_or_incomplete': len(subset) - len(valid),
            'expected_conflicts': sum(r['expected_contradiction'] for r in subset),
            'accepted_false_conflicts': sum(incompatible(r) and not r['expected_contradiction'] for r in valid),
            'raw_false_conflicts_including_rejected': sum(incompatible(r) and not r['expected_contradiction'] for r in subset),
            'missed_or_invalid_conflicts': sum(r['expected_contradiction'] and
                (r['validation_error'] is not None or not incompatible(r)) for r in subset),
            'accepted_underdetermined': sum(r['prediction']['relationship'] == 'underdetermined' for r in valid),
            'invalid_compatible_cases': sum(not r['expected_contradiction'] and r['validation_error'] is not None for r in subset),
            'logical_input_tokens': sum(r['input_tokens'] for r in subset),
            'logical_output_tokens': sum(r['output_tokens'] for r in subset),
            'fresh_recorded_calls': sum(not r['cache_hit'] and not r.get('parent_reuse', False) for r in subset),
        }
    return {'groups': groups,
        'scope': 'synthetic instrument/representation controls; not native QA, OOD, energy contribution or efficiency benchmark',
        'no_rejected_rows_dropped': True, 'full_runs_authorized': False}


CONTROL_HASH = stable_hash({'policy': POLICY, 'frame_correction': FRAME_CORRECTION,
    'frame_system': SYSTEM, 'schemas': [output_schema(a) for a in ARMS],
    'observation_time': '2025-01-01'})
