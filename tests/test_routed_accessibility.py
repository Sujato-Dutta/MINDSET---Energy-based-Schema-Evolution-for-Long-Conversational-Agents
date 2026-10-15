from dataclasses import replace
from mindset.methods.compact import CompactConfig
from mindset.methods.routed_accessibility import RoutedAccessibilityMemory
from mindset.retrieval.hybrid import ranked_indices, reciprocal_rank_fusion
from test_compact import Encoder, Reranker, event


def test_provenance_off_is_exact_flat_hybrid_not_weakened_baseline():
    m = RoutedAccessibilityMemory(CompactConfig(ablation='no_provenance'),
        encoder=Encoder(), local_reranker=Reranker())
    m.ingest([event(i, f'Tea coffee home work item {i}') for i in range(1, 60)])
    q = m._encode(['tea'])[0]
    fused = reciprocal_rank_fusion([ranked_indices(m._episode_vectors @ q), ranked_indices(m._bm25.scores('tea'))])
    expected = sorted(fused, key=lambda i: (-fused[i], i))[:48]
    assert m.candidate_pool('tea')[0] == expected


def test_routed_pool_has_unique_raw_sources_and_one_ce_pass():
    m = RoutedAccessibilityMemory(CompactConfig(), encoder=Encoder(), local_reranker=Reranker())
    m.ingest([event(i, f'Tea coffee home work item {i}') for i in range(1, 60)])
    hits = m.retrieve('tea', 8)
    assert len(set(m.last_candidate_pool)) == 48
    assert m.local_reranker.calls == [48]
    assert len(hits) == 8
    assert all(h.source_event_ids == (h.memory_id,) for h in hits)


def test_only_read_path_changes_for_the_same_controller():
    from mindset.methods.accessibility import AccessibilityMemory
    events = [event(i, f'Tea coffee home work item {i}') for i in range(1, 25)]
    a = RoutedAccessibilityMemory(CompactConfig(), encoder=Encoder(), local_reranker=Reranker())
    b = AccessibilityMemory(a.config, encoder=Encoder(), local_reranker=Reranker())
    a.ingest(events); b.ingest(events)
    assert a.operation_log.all() == b.operation_log.all()
    assert a.access_records == b.access_records
