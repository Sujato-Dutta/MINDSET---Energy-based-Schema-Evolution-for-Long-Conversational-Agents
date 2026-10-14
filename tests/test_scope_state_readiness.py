import importlib.util
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('scope_state_readiness', SCRIPTS / 'audit_scope_state_inputs.py')
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def row(rid, left, right, a, b, *, expected=True, relation='incompatible', error=None):
    def frame(sid, quotes):
        return {'source_id': sid, **dict(zip(worker.CORE, quotes)), 'canonical_relation': 'preference',
                'modality': 'asserted', 'value_mode': 'set_member'}
    return {'id': rid, 'left': left, 'right': right, 'expected_contradiction': expected,
        'family': 'scope_test', 'validation_error': error,
        'prediction': {'frames': [frame('A', a), frame('B', b)], 'relationship': relation}}


def test_missing_negation_and_core_alias_not_promoted_to_truth():
    report = worker.readiness([row('x', 'I like jazz.', 'I do not like jazz.',
        ('I', 'like', 'jazz'), ('I', 'like', 'jazz'))])
    assert report['literal_negation_absent_from_core_quote_count'] == 1
    assert report['conflict_control_core_quote_alias_count'] == 1
    assert report['raw_source_lossless'] and report['no_automatic_world_state_updates']


def test_negation_preserved_and_rejections_counted():
    report = worker.readiness([
        row('x', 'I like jazz.', 'I do not like jazz.', ('I', 'like', 'jazz'), ('I', 'do not like', 'jazz')),
        row('y', 'a', 'b', ('', '', ''), ('', '', ''), error='quote error')])
    assert report['literal_negation_absent_from_core_quote_count'] == 0
    assert report['conflict_control_core_quote_alias_count'] == 0
    assert report['cases'] == 2 and report['rejected_cases'] == 1


def test_unresolved_core_frame_separate_from_pair_decision():
    report = worker.readiness([row('x', 'I like jazz.', 'I like rock.',
        ('', 'like', 'jazz'), ('I', 'like', 'rock'), expected=False, relation='compatible')])
    assert report['accepted_frames_with_missing_core_fields'] == 1
    assert report['accepted_compatible_fixture_families'] == ['scope_test']
