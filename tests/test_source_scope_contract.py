from copy import deepcopy
import json
import pytest
from mindset.memory.source_scope_contract import scope_messages,validate_measurement


def payload():
    def frame(source,value):
        return {'source_id':source,'holder_quote':'I','predicate_quote':'like','value_quote':value,
            'context_quotes':[],'time_quotes':[],'canonical_relation':'likes','modality':'asserted','value_mode':'set_member'}
    return {'frames':[frame('A','jazz'),frame('B','rock')],'relationship':'compatible'}


def test_quotes_verified_not_semantic_truth():
    result=validate_measurement(payload(),'I like jazz.','I like rock.')
    assert result['literal_provenance_verified']
    assert not result['semantic_correctness_verified'] and not result['world_truth_or_update_verified']


def test_generated_cross_source_quote_rejected():
    p=payload(); p['frames'][1]['value_quote']='jazz'
    with pytest.raises(ValueError,match='quote'): validate_measurement(p,'I like jazz.','I like rock.')


def test_duplicate_frame_and_unknown_labels_rejected():
    p=payload(); p['frames'][1]['source_id']='A'
    with pytest.raises(ValueError): validate_measurement(p,'I like jazz.','I like rock.')
    p=payload(); p['frames'][0]['modality']='truth'
    with pytest.raises(ValueError): validate_measurement(p,'I like jazz.','I like rock.')


def test_unresolved_fields_do_not_get_promoted_to_facts():
    p=payload(); p['frames'][0]['holder_quote']=''; p['frames'][0]['value_mode']='unknown'
    p['relationship']='underdetermined'
    assert not validate_measurement(p,'I like jazz.','I like rock.')['world_truth_or_update_verified']


def test_explicit_field_schema_and_no_expected_labels_in_prompt():
    prompt=scope_messages('I like jazz.','I like rock.')
    body=json.loads(prompt[1]['content'])
    assert set(body)=={'reports','output_schema'}
    assert body['output_schema']['properties']['relationship']['enum']==['compatible','incompatible','underdetermined']
    assert 'expected_contradiction' not in str(prompt)
    assert 'holder_quote' in prompt[0]['content']


def test_confirmation_fixtures_disjoint_and_not_present_in_contract_prompt():
    import importlib.util
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('scope_confirmation',root/'scripts/source_scope_confirmation_fixtures.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    fixtures=module.confirmation_fixtures()
    assert len(fixtures)==24 and len({r['id'] for r in fixtures})==24
    assert sum(r['expected_contradiction'] for r in fixtures)==4
    assert len({r['family'] for r in fixtures})==6
    original_spec=importlib.util.spec_from_file_location('scope_original',root/'scripts/stress_source_semantics.py')
    original=importlib.util.module_from_spec(original_spec); original_spec.loader.exec_module(original)
    assert not ({(r['left'],r['right']) for r in fixtures}&{(r['left'],r['right']) for r in original.fixtures()})
    for case in fixtures:
        body=json.loads(scope_messages(case['left'],case['right'])[1]['content'])
        assert 'expected_contradiction' not in body and 'split' not in body
