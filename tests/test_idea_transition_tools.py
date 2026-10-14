from dataclasses import replace
import numpy as np
from mindset.methods.joint import JointMindsetMemory, typed_features
from mindset.methods.compact import CompactConfig
from mindset.methods.idea_transition_tools import feasible_plans, state_for_plan, apply_plan
from test_compact import Encoder, Reranker, event

def test_branch_preserves_parent_models_and_future_siblings():
    m=JointMindsetMemory(CompactConfig(.1,.05),encoder=Encoder(),local_reranker=Reranker())
    m.ingest([event(1,'I live in Paris.')])
    # Derive the incoming immutable representation without touching m.
    other=JointMindsetMemory(CompactConfig(.1,.05),encoder=Encoder(),local_reranker=Reranker())
    other.ingest([event(1,'I live in Paris.'),event(2,'I moved to Rome.')])
    e=other.episodes['e2']; before=m.validate_invariants(); log=m.operation_log.all()
    plans=feasible_plans(m,e)
    assert m.validate_invariants()==before and m.operation_log.all()==log
    states=[state_for_plan(m,e,p) for p in plans]
    assert all(s.encoder is m.encoder and s.local_reranker is m.local_reranker for s in states)
    assert m.validate_invariants()==before and 'e2' not in m.episodes
    assert any(s.operation_log.all()[-1].operation=='supersede' for s in states)
    assert any(s.operation_log.all()[-1].stored_as_exception for s in states)
    assert all(s.validate_invariants()['grounded_episode_count']==2 for s in states)
    merge=next(p for p in plans if p['mutation']=='exception')
    apply_plan(m,e,merge)
    assert m.validate_invariants()['grounded_episode_count']==2

def test_tools_greedy_matches_joint_decisions_and_state():
    events=[event(1,'I live in Paris.'),event(2,'I moved to Rome.'),event(3,'I live in Rome.')]
    ref=JointMindsetMemory(CompactConfig(.1,.05),encoder=Encoder(),local_reranker=Reranker())
    ref.ingest(events)
    m=JointMindsetMemory(CompactConfig(.1,.05),encoder=ref.encoder,local_reranker=ref.local_reranker)
    priority={'reinforce':0,'supersede':1,'split':2,'new':3}
    for eid in ref._episode_order:
        e=ref.episodes[eid]; plans=feasible_plans(m,e)
        p=min(plans,key=lambda p:(p['cost'],priority[p['action']],p['schema_id'] or ''))
        apply_plan(m,e,p)
    assert [r.operation for r in m.operation_log.all()]==[r.operation for r in ref.operation_log.all()]
    assert list(m.schemas)==list(ref.schemas)
    for sid in m.schemas:
        assert m.schemas[sid].all_provenance()==ref.schemas[sid].all_provenance()
        np.testing.assert_allclose(m.schemas[sid].centroid,ref.schemas[sid].centroid)
