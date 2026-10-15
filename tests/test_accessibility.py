from dataclasses import replace
import numpy as np
import pytest
from mindset.methods.accessibility import AccessibilityMemory, projected_roots, top_roots
from mindset.methods.joint_one_price import OnePriceJointMemory
from mindset.methods.compact import CompactConfig
from test_compact import Encoder, Reranker, event


def memory(mode='full', budget=8):
    return AccessibilityMemory(CompactConfig(complexity_penalty=.1, switch_margin=.05),
                               access_mode=mode, probe_budget=budget,
                               encoder=Encoder(), local_reranker=Reranker())


@pytest.mark.parametrize('n', [1, 3, 6, 7, 17])
def test_one_column_projection_matches_complete_ranking(n):
    rng = np.random.default_rng(n)
    scores = rng.normal(size=(20, n)).astype(np.float32)
    scores[0] = 0  # Exact ID-tie case.
    names = [f's{n-i:03d}' for i in range(n)]
    before = top_roots(scores, names, 7)
    for col in range(n):
        values = rng.normal(size=20).astype(np.float32); values[0] = 0
        actual = projected_roots(scores, before, names, 6, col, values)
        changed = scores.copy(); changed[:, col] = values
        np.testing.assert_array_equal(actual, top_roots(changed, names, 6))
    values = rng.normal(size=20).astype(np.float32); values[0] = 0
    actual = projected_roots(scores, before, names, 6, n, values, 's000')
    np.testing.assert_array_equal(actual, top_roots(np.column_stack((scores, values)), [*names, 's000'], 6))


def test_no_access_preserves_original_actual_mutations():
    events = [event(1, 'I live in Paris.'), event(2, 'I moved to Rome.'),
              event(3, 'I prefer tea at home.'), event(4, 'I prefer coffee at work.'),
              event(5, 'I prefer tea at home.'), event(6, 'The garden is beautiful.')]
    a = memory('no_access')
    b = OnePriceJointMemory(a.config, encoder=Encoder(), local_reranker=Reranker())
    a.ingest(events); b.ingest(events)
    assert a.operation_log.all() == b.operation_log.all()
    assert a._decision_certificates['e5']['selected_schema_id'] == b._decision_certificates['e5']['selected_schema_id']
    assert a.validate_invariants()['grounded_episode_count'] == 6
    assert a.telemetry()['route_interventions'] == 0


def test_incremental_route_scores_match_real_centroids_and_links():
    a = memory()
    for e in [event(1, 'I live in Paris.'), event(2, 'I moved to Rome.'),
              event(3, 'I prefer tea at home.'), event(4, 'I prefer coffee at work.')]:
        a.ingest([e])
        real = np.asarray([a.schemas[sid].centroid for sid in a._route_ids])
        np.testing.assert_allclose(a._route_centers, real, atol=1e-6)
        np.testing.assert_allclose(a._route_scores, a._probe_vectors @ real.T, atol=1e-6)
    for sid, schema in a.schemas.items():
        for linked in [schema.predecessor_schema_id, *schema.successor_schema_ids, *a._split_links[sid]]:
            if linked:
                assert a._component(sid) == a._component(linked)


def test_bounded_deterministic_probes_and_raw_evidence():
    events = [event(i, f'Tea coffee home work item {i}') for i in range(1, 45)]
    a = memory(budget=4); a.ingest(events)
    b = memory(budget=4); b.ingest(events[:20]); b.ingest(events[20:])
    assert a._probe_ids == b._probe_ids
    assert len(a._probe_ids) == 4
    assert a.schemas.keys() == b.schemas.keys()
    assert all(e.text == a.episodes[e.event_id].text for e in events)
    hits = a.retrieve('tea', 8)
    assert len(hits) == 8 and a.local_reranker.calls == [44]
    assert a.telemetry()['llm_ingestion_calls'] == 0


def test_risk_scoring_is_non_mutating():
    a = memory(); a.ingest([event(1, 'I prefer tea at home.')])
    incoming = replace(a.episodes['e1'], episode_id='next', vector=unit([0., 1., *([0.] * 7)]))
    a._prepare_routes(incoming)
    plan = a._plans(next(iter(a.schemas.values())), incoming)[0]
    original = a._route_scores.copy(); center = plan['schema'].centroid.copy()
    a._risk(plan, incoming)
    np.testing.assert_array_equal(a._route_scores, original)
    np.testing.assert_array_equal(plan['schema'].centroid, center)


def unit(v):
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def test_risk_matches_independent_actual_post_state_replay():
    """Verify the surrogate against mutated schemas, not its own projection."""
    from dataclasses import asdict
    from mindset.methods.idea_transition_tools import state_for_plan
    from mindset.methods.joint import typed_features
    rng = np.random.default_rng(20261014)
    events = [event(i, f'An unrelated memo item {i}.') for i in range(1, 26)]
    vectors = {e.event_id: unit(rng.normal(size=16)) for e in events}
    a = AccessibilityMemory(CompactConfig(complexity_penalty=.1, switch_margin=0., nearest_schema_k=2),
        encoder=Encoder(), local_reranker=Reranker(), precomputed_embeddings=vectors, probe_budget=20)
    a.ingest(events)
    incoming = replace(a.episodes['e25'], episode_id='incoming', text='Another unrelated memo.',
        vector=unit(rng.normal(size=16)), features=typed_features('Sam', 'Another unrelated memo.'))
    a._prepare_routes(incoming)

    def accessible(state, vector):
        ids = sorted(state.schemas)
        scores = np.asarray([state.schemas[sid].centroid for sid in ids]) @ vector
        roots = top_roots(scores[None, :], ids, 2)[0]
        selected = [ids[i] for i in roots]; seen = set(selected)
        for sid in selected:
            s = state.schemas[sid]
            for link in [s.predecessor_schema_id, *s.successor_schema_ids, *state._split_links[sid]]:
                if link and link not in seen:
                    selected.append(link); seen.add(link)
        return {eid for sid in selected for eid in state.schemas[sid].all_provenance()}

    before = {eid: eid in accessible(a, a.episodes[eid].vector) for eid in a._probe_ids}
    nearest = sorted(a.schemas.values(), key=lambda s: (-float(s.centroid @ incoming.vector), s.schema_id))[:2]
    for schema in nearest:
        for plan in a._plans(schema, incoming):
            projected = a._risk(plan, incoming)
            state = state_for_plan(a, incoming, dict(schema_id=schema.schema_id,
                action=plan['action'], mutation=plan['mutation'], cost=plan['cost'],
                terms=plan['terms'].as_dict(), signals=asdict(plan['signals'])))
            history = sum(before[eid] and eid not in accessible(state, state.episodes[eid].vector)
                          for eid in a._probe_ids)
            current = int(incoming.episode_id not in accessible(state, incoming.vector))
            assert projected['history_routes_lost'] == history
            assert projected['new_source_inaccessible'] == current
            assert projected['access_risk'] == history + current
