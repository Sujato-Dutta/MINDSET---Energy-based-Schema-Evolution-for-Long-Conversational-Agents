from types import SimpleNamespace as N
import numpy as np
import pytest
from mindset.methods.compact import CompactConfig, CompactMindsetMemory, select_transition
from mindset.methods.base import ConversationEvent

class Encoder:
    def encode(self,texts,**kwargs):
        vocabulary=["paris","rome","tea","coffee","home","work","live","prefer","garden"]
        rows=np.asarray([[float(w in t.lower()) for w in vocabulary] for t in texts],dtype=np.float32)
        rows[:, -1]+=.01
        return rows/np.linalg.norm(rows,axis=1,keepdims=True)

class Reranker:
    def __init__(self): self.calls=[]
    def predict(self,pairs,**kwargs):
        self.calls.append(len(pairs))
        return np.asarray([sum(w.lower() in d.lower() for w in q.split()) for q,d in pairs])

def event(i,text):
    return ConversationEvent(f"e{i}","c",f"s{i}",f"2025-01-{i:02d}","Sam",text)

def memory(config=None):
    return CompactMindsetMemory(config,encoder=Encoder(),local_reranker=Reranker())

@pytest.mark.parametrize("advantage,expected",[(.10,"reinforce"),(.18,"reinforce"),(.19,"supersede")])
def test_strict_hysteresis(advantage,expected):
    energies={"reinforce":N(total=1.),"supersede":N(total=1.-advantage)}
    assert select_transition(energies,{"reinforce","supersede"},.18)[0]==expected

def test_compact_temporal_update_retains_both_versions():
    m=memory(); m.ingest([event(1,"I live in Paris."),event(2,"I moved to Rome.")])
    assert m.operation_log.all()[-1].operation=="supersede"
    assert len([s for s in m.schemas.values() if s.status=="historical"])==1
    assert m.validate_invariants()["grounded_episode_count"]==2
    assert m.telemetry()["llm_ingestion_calls"]==0

def test_compact_context_split_and_noise_confirmation():
    m=memory(); m.ingest([event(1,"I prefer tea at home."),event(2,"I prefer coffee at work.")])
    assert m.operation_log.all()[-1].operation=="split"
    assert all(s.status=="active" for s in m.schemas.values())
    m=memory(); m.ingest([event(1,"I live in Paris."),event(2,"I live in Rome.")])
    assert m.operation_log.all()[-1].stored_as_exception
    m.ingest([event(3,"I live in Rome.")])
    assert m.operation_log.all()[-1].operation=="supersede"

def test_exact_sufficient_statistics_and_medoid():
    m=memory(CompactConfig(complexity_penalty=1.))
    m.ingest([event(1,"I prefer tea."),event(2,"I prefer tea and coffee."),event(3,"I prefer coffee.")])
    schema=next(iter(m.schemas.values()))
    vectors=np.asarray([m.episodes[i].vector for i in schema.support_episode_ids])
    total=vectors.sum(axis=0)
    np.testing.assert_allclose(m._sums[schema.schema_id],total,atol=1e-7)
    expected=np.clip((1-np.linalg.norm(total)/len(vectors))/2,0,1)
    assert schema.mean_distortion==pytest.approx(expected,abs=1e-6)
    closest=sorted(schema.support_episode_ids)[int(np.argmax(
        np.asarray([m.episodes[i].vector for i in sorted(schema.support_episode_ids)])@schema.centroid))]
    assert schema.medoid_episode_id==closest

def test_bounded_single_reranking_pass_and_raw_provenance():
    m=memory(); m.ingest([event(i,f"I prefer tea item {i}.") for i in range(1,25)])
    hits=m.retrieve("Sam tea",8)
    assert len(hits)==8 and len(m.local_reranker.calls)==1
    assert m.local_reranker.calls[0]<=m.config.retrieval_candidate_k
    assert all(h.source_event_ids==(h.memory_id,) for h in hits)

def test_owner_blocking_and_no_backdated_supersession():
    m=memory()
    m.ingest([event(3,"I live in Paris."),event(1,"I moved to Rome.")])
    assert not any(s.status=="historical" for s in m.schemas.values())
    assert m.operation_log.all()[-1].stored_as_exception


def test_switch_margin_does_not_penalize_context_split_or_new_topic():
    energies={"reinforce":N(total=.5),"split":N(total=.3),"new":N(total=.2)}
    assert select_transition(energies,{"reinforce","split"},.9)[0]=="split"
    assert select_transition(energies,{"reinforce","new"},.9)[0]=="new"
    energies["supersede"]=N(total=.1)
    assert select_transition(energies,{"reinforce","split","supersede"},.3)[0]=="split"


def test_exact_damage_term_matches_explicit_old_episode_rescoring():
    m=memory(); m.ingest([event(1,"I prefer tea."),event(2,"I prefer tea and garden.")])
    schema=next(iter(m.schemas.values()))
    m.ingest([event(3,"I prefer tea and coffee.")])
    incoming=m.episodes["e3"]
    # Reconstruct the pre-arrival sufficient statistics independently.
    vectors=np.asarray([m.episodes[i].vector for i in ("e1","e2")])
    old_sum=vectors.sum(axis=0); old_center=old_sum/np.linalg.norm(old_sum)
    new_center=(old_sum+incoming.vector)/np.linalg.norm(old_sum+incoming.vector)
    explicit=max(0.,float(np.mean((vectors@old_center-vectors@new_center)/2)))
    record=m.operation_log.all()[-1]
    assert record.energy_terms["reinforce"]["historical_damage"]==pytest.approx(explicit,abs=1e-7)


def test_energy_decision_certificate_and_portable_statistics(tmp_path):
    import json
    m=memory(); m.ingest([event(1,"I live in Paris."),event(2,"I moved to Rome.")])
    assert m.telemetry()["max_energy_optimality_gap"]==0
    p=m.snapshot(tmp_path/"snapshot.json"); payload=json.loads(p.read_text())
    certificate=payload["transitions"][-1]["decision_certificate"]
    assert certificate["optimality_gap"]==0 and "supersede" in certificate["feasible_actions"]
    arrays=np.load(p.with_suffix(".npz"))
    for sid,total in zip(payload["sufficient_statistics"]["schema_order"],arrays["_sufficient_sums"]):
        np.testing.assert_array_equal(total,m._sums[sid])


def test_ingestion_batching_preserves_state_and_energy_decisions():
    texts=["I live in Paris.","I live in Rome.","I live in Rome.",
           "I prefer tea at home.","I prefer coffee at work."]
    events=[event(i+1,t) for i,t in enumerate(texts)]
    whole=memory(); whole.ingest(events)
    incremental=memory()
    for e in events: incremental.ingest([e])
    assert [r.operation for r in whole.operation_log.all()]==[r.operation for r in incremental.operation_log.all()]
    assert whole.validate_invariants()==incremental.validate_invariants()
    for sid,s in whole.schemas.items():
        np.testing.assert_array_equal(s.centroid,incremental.schemas[sid].centroid)


def test_structured_owner_and_session_confirmation_are_not_cross_user():
    m=memory()
    e=event(1,"I live in Paris.")
    m.ingest([e,ConversationEvent("other","c","s2","2025-01-02","Pat","I live in Rome.")])
    assert len(m.schemas)==2 and all(s.status=="active" for s in m.schemas.values())
    m=memory(); m.ingest([event(1,"I live in Paris."),
        ConversationEvent("b","c","same","2025-01-02","Sam","I live in Rome."),
        ConversationEvent("c","c","same","2025-01-03","Sam","I live in Rome.")])
    assert m.operation_log.all()[-1].stored_as_exception
    assert not any(s.status=="historical" for s in m.schemas.values())


def test_duplicate_batch_ids_fail_before_any_memory_mutation():
    m=memory(); e=event(1,"I live in Paris.")
    with pytest.raises(ValueError,match="immutable"): m.ingest([e,e])
    assert not m.episodes and not m.schemas and not m.operation_log.all()
