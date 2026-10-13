from dataclasses import replace
import numpy as np
import pytest
from mindset.methods.factorized_state import FactorizedMindsetMemory, atomic_features
from mindset.methods.aligned_readout import AlignedMindsetMemory
from mindset.methods.compact import CompactConfig
from test_compact import Encoder, Reranker, event


def memory(texts, **kwargs):
    m = FactorizedMindsetMemory(CompactConfig(complexity_penalty=.1, switch_margin=.05),
                               encoder=Encoder(), local_reranker=Reranker(), **kwargs)
    m.ingest([event(i, text) for i, text in enumerate(texts, 1)])
    return m


def selected(m, question, date=''):
    return [h.memory_id for h in m.select_scored(question, list(range(len(m._episode_order))),
            list(range(len(m._episode_order), 0, -1)), 8, date)]


def test_each_attribute_has_its_own_version_and_one_raw_source():
    texts = ['I live in Paris. I work for Acme.', 'I moved to Rome. I now work for Beta.']
    m = memory(texts)
    assert {s.topic for s in m.schemas.values()} == {'sam::location', 'sam::employer'}
    assert sum(s.status == 'historical' for s in m.schemas.values()) == 2
    assert selected(m, 'Who does Sam currently work for?') == ['e2']
    assert selected(m, 'Where does Sam live now?') == ['e2']
    assert selected(m, 'Who did Sam work for on 2025-01-01?') == ['e1']
    assert len(m.episodes) == 2 and len(m._documents) == 2
    assert m.validate_invariants()['transition_count'] == 4
    assert [m.episodes[f'e{i}'].text for i in (1, 2)] == texts
    assert m.telemetry()['llm_ingestion_calls'] == 0


def test_clause_contexts_do_not_bleed_into_other_facts():
    features = atomic_features('Sam', 'I prefer tea at home. I prefer coffee at work.')
    assert len(features) == 2
    assert {f.contexts for f in features} == {('at home',), ('at work',)}
    m = memory(['I prefer tea at home. I prefer coffee at work.'])
    assert len(m.schemas) == 2
    assert m.validate_invariants()['episode_count'] == 1
    m = memory(['I prefer tea at home. I prefer coffee at work.'], write_policy='nearest')
    assert len(m.schemas) == 1
    assert m.validate_invariants()['transition_count'] == 2
    schema = next(iter(m.schemas.values()))
    np.testing.assert_array_equal(m._sums[schema.schema_id], m.episodes['e1'].vector)


def test_local_cost_and_transition_invariant_to_unrelated_history():
    first = event(1, 'My budget is 100 monthly dollars.')
    last = replace(event(2, 'My budget is 200 monthly dollars now.'), event_id='last', timestamp='2025-02-01')
    padding = [replace(event(3, 'Weather discussion.'), event_id=f'padding-{i}', timestamp='2025-01-02') for i in range(256)]
    a = memory([]); a.ingest([first, last])
    b = memory([]); b.ingest([first, *padding, last])
    ca = next(c for c in a.observation_audit() if c['source_id'] == 'last')
    cb = next(c for c in b.observation_audit() if c['source_id'] == 'last')
    assert ca['action'] == cb['action'] == 'supersede'
    assert ca['selected_adjusted_energy'] == pytest.approx(cb['selected_adjusted_energy'])
    legacy_cost = memory([], local_complexity=False); legacy_cost.ingest([first, *padding, last])
    assert next(c for c in legacy_cost.observation_audit() if c['source_id'] == 'last')['action'] == 'reinforce'


def test_atomic_off_control_exposes_primary_slot_omission():
    texts = ['I live in Paris. I work for Acme.', 'I moved to Rome. I now work for Beta.']
    m = memory(texts, atomic=False)
    assert {s.topic for s in m.schemas.values()} == {'sam::location'}
    assert selected(m, 'Who does Sam currently work for?') == ['e1', 'e2']


def test_ambiguous_negated_and_quoted_claims_are_not_resolved_by_clause_order():
    assert not atomic_features('Sam', 'My budget is 100. My budget is 200.')
    for text in ['I do not live in Paris.', '"I live in Paris."', 'I used to live in Paris.']:
        assert not atomic_features('Sam', text)
    m = memory(['My budget is 100. My budget is 200.'])
    assert m.validate_invariants()['grounded_episode_count'] == 1


def test_state_disabled_retains_ranking_and_nearest_has_no_versions():
    texts = ['I live in Paris. I work for Acme.', 'I moved to Rome. I now work for Beta.']
    m = memory(texts, state_readout=False)
    assert selected(m, 'Where does Sam live now?') == ['e1', 'e2']
    m = memory(texts, write_policy='nearest')
    assert all(s.status == 'active' for s in m.schemas.values())
    assert m.validate_invariants()['transition_count'] == 4


def test_raw_embeddings_and_sources_are_not_duplicated_by_atomization():
    texts = ['I live in Paris. I work for Acme.', 'I moved to Rome. I now work for Beta.']
    m = memory(texts)
    old = AlignedMindsetMemory(m.config, encoder=Encoder(), local_reranker=Reranker())
    old.ingest([event(i, text) for i, text in enumerate(texts, 1)])
    assert m._episode_order == old._episode_order and m._documents == old._documents
    np.testing.assert_array_equal(m._episode_vectors, old._episode_vectors)
    assert m.telemetry()['max_observation_energy_gap'] == 0
    assert all(h.source_event_ids == (h.memory_id,) for h in m.retrieve('Sam location employer', 8))


def test_incremental_and_batch_ingestion_match():
    texts = ['I live in Paris. I work for Acme.', 'I live in Rome. I work for Beta.', 'I live in Rome. I work for Beta.']
    a = memory(texts); b = memory([])
    for i, text in enumerate(texts, 1): b.ingest([event(i, text)])
    assert [r.operation for r in a.operation_log.all()] == [r.operation for r in b.operation_log.all()]
    assert a.validate_invariants() == b.validate_invariants()
