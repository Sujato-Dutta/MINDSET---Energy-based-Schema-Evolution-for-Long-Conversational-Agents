import importlib.util
import json
from pathlib import Path

import pytest

from mindset.memory.source_relation_contract import relation_messages, validate_relation, RELATIONS

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('relation_fixtures', ROOT / 'scripts/source_relation_fixtures.py')
fixtures_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures_module)


def test_balanced_synthetic_panel_and_label_free_inputs():
    fixtures = fixtures_module.relation_fixtures()
    assert len(fixtures) == 32 and len({f['id'] for f in fixtures}) == 32
    for split in ('development', 'lexical_confirmation'):
        subset = [f for f in fixtures if f['split'] == split]
        assert all(sum(f['expected_relationship'] == relation for f in subset) == 2 for relation in RELATIONS)
    for f in fixtures:
        body = json.loads(relation_messages(f['reports'])[1]['content'])
        assert set(body) == {'reports', 'output_schema'} and body['reports'] == f['reports']
        assert 'expected_direction' not in str(body) and 'expected_relationship' not in str(body)


def test_complete_negated_source_preserved_not_normalized_atom():
    reports = fixtures_module.relation_fixtures()[24]['reports']
    body = json.loads(relation_messages(reports)[1]['content'])
    assert body['reports'] == reports
    assert any('not' in r['text'] for r in reports)
    result = validate_relation({'relationship': 'same_scope_conflict', 'direction': 'none'}, reports)
    assert result['complete_input_sources_retained']
    assert not result['semantic_correctness_verified'] and not result['world_truth_or_update_verified']


def test_invalid_chronology_shape_rejected_without_claiming_true_dates():
    reports = fixtures_module.relation_fixtures()[4]['reports']
    for payload in ({'relationship': 'temporal_change', 'direction': 'none'},
                    {'relationship': 'independent', 'direction': 'a_to_b'},
                    {'relationship': 'truth', 'direction': 'none'}):
        with pytest.raises(ValueError):
            validate_relation(payload, reports)
    result = validate_relation({'relationship': 'temporal_change', 'direction': 'b_to_a'}, reports)
    assert not result['semantic_correctness_verified']  # admissible does not mean correct


def test_retrospective_controls_defeat_observation_order_shortcut():
    fixtures = fixtures_module.relation_fixtures()
    retrospective = [f for f in fixtures if f['expected_direction'] == 'b_to_a']
    assert len(retrospective) == 2  # one reverse-chronology case per split
    assert all(f['reports'][0]['observation_time'] < f['reports'][1]['observation_time'] for f in retrospective)


def test_reports_require_unique_ids_and_do_not_admit_extra_label_fields():
    reports = fixtures_module.relation_fixtures()[0]['reports']
    duplicate = [{**r, 'source_id': 'A'} for r in reports]
    with pytest.raises(ValueError):
        relation_messages(duplicate)
    with pytest.raises(ValueError):
        relation_messages([{**reports[0], 'expected_relationship': 'equivalent'}, reports[1]])
