import numpy as np
import pytest
from mindset.methods.joint import JointMindsetMemory,typed_features,utc_time
from mindset.methods.compact import CompactConfig
from mindset.acl.latency import latency_summary
from mindset.acl.pipeline import retrieval_request_k
from test_compact import Encoder,Reranker,event

def memory(**kwargs):
    return JointMindsetMemory(CompactConfig(complexity_penalty=.1,switch_margin=.2),encoder=Encoder(),local_reranker=Reranker(),**kwargs)

def test_untyped_attributes_are_not_contradictions():
    assert not typed_features("Sam","I'm thrilled.").attributes
    assert typed_features("Sam","I am an engineer.").attributes==(("occupation","engineer"),)

def test_actual_exception_energy_has_no_historical_damage():
    m=memory(); m.ingest([event(1,"I live in Paris."),event(2,"I live in Rome.")])
    record=m.operation_log.all()[-1]
    assert record.stored_as_exception and record.energy_terms["reinforce"]["historical_damage"]==0
    assert m._decision_certificates["e2"]["selected_mutation"]=="exception"

def test_joint_minimum_is_over_all_schema_action_pairs():
    m=memory(); m.ingest([event(1,"I prefer tea at home."),event(2,"I prefer coffee at work."),event(3,"I prefer tea at home.")])
    cert=m._decision_certificates["e3"]
    assert len({p["schema_id"] for p in cert["feasible_candidates"]})==2
    assert cert["selected_adjusted_energy"]==min(p["adjusted_energy"] for p in cert["feasible_candidates"])
    assert m.validate_invariants()["grounded_episode_count"]==3

def test_utc_time_has_fixed_default_and_naive_zone():
    assert utc_time("2025-01-01")==utc_time("2025-01-01T00:00:00Z")
    assert utc_time("nonsense")==0

def test_unknown_context_is_not_known_same_and_update_keeps_history():
    m=memory(); m.ingest([event(1,"I live in Paris."),event(2,"I moved to Rome.")])
    assert m.last_features.context_match==.5
    assert m.operation_log.all()[-1].operation=="supersede"
    assert len([s for s in m.schemas.values() if s.status=="historical"])==1

def test_backfill_is_exact_and_one_ce_pass():
    m=memory(); m.ingest([event(i,f"Tea coffee home work item {i}") for i in range(1,61)])
    hits=m.retrieve("tea",8)
    assert len(m.last_candidate_pool)==48 and len(set(m.last_candidate_pool))==48
    assert m.local_reranker.calls==[48] and len(hits)==8
    assert all(h.source_event_ids==(h.memory_id,) for h in hits)

def test_latency_without_answers_is_not_end_to_end():
    result=latency_summary([{"retrieval_seconds":.1},{"retrieval_seconds":.2}])
    assert result["retrieval_latency_p50_ms"]==pytest.approx(150)
    assert result["retrieval_latency_p95_ms"]==pytest.approx(195)
    assert result["end_to_end_p50_ms"] is None and result["end_to_end_queries"]==0

def test_actual_e2e_percentiles_and_cache_exclusion():
    result=latency_summary([{"retrieval_seconds":.1,"end_to_end_seconds":1.1},
                          {"retrieval_seconds":.2,"end_to_end_seconds":1.2},
                          {"retrieval_seconds":0,"end_to_end_seconds":0,"cache_hit":True}])
    assert result["end_to_end_p50_ms"]==pytest.approx(1150)
    with pytest.raises(ValueError): latency_summary([{"retrieval_seconds":.1,"end_to_end_seconds":1.}, {"retrieval_seconds":.2}])

def test_lightmem_eight_returned_hits_only():
    assert retrieval_request_k("lightmem",{"retrieval":{"candidate_k":40,"lightmem_hits":8}})==8
    assert retrieval_request_k("mindset",{"retrieval":{"candidate_k":40}})==40
