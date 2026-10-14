"""Synthetic, independent mechanism tests; no benchmark correctness labels."""
from dataclasses import replace
from collections import defaultdict

import pytest

from mindset.methods.compact import CompactConfig
from mindset.methods.ideas_regret_program_proof import (
    FutureQueryRegretMemory, CompositionalEditMemory, ProofCarryingMemory,
    accessible_sources, bind_query, extend_program, probe_regret,
    source_obligations, verify_plan,
)
from mindset.methods.idea_transition_tools import adopt_state, feasible_plans, state_for_plan
from mindset.methods.joint import JointMindsetMemory
from test_compact import Encoder, Reranker, event


def memory(cls=JointMindsetMemory, **kwargs):
    return cls(CompactConfig(complexity_penalty=.1, switch_margin=.2),
               encoder=Encoder(), local_reranker=Reranker(), **kwargs)


@pytest.mark.parametrize("cls", [FutureQueryRegretMemory, CompositionalEditMemory, ProofCarryingMemory])
def test_untyped_fallback_preserves_raw_outputs(cls):
    events = [event(1, "Tea is pleasant."), event(2, "Coffee item.")]
    base, proposed = memory(), memory(cls)
    base.ingest(events)
    proposed.ingest(events)
    assert proposed.retrieve("tea", 8) == base.retrieve("tea", 8)
    assert proposed.telemetry()["idea_noop_episodes"] == 2
    assert proposed.validate_invariants()["grounded_episode_count"] == 2


def test_future_probes_explicit_update_cover_current_and_previous_sources():
    m = memory(FutureQueryRegretMemory)
    m.ingest([event(1, "I live in Paris."), event(2, "I moved to Rome.")])
    record = m.idea_records["e2"]
    assert record["supported"] and record["surrogate_regret"] == 0
    assert {p["route"] for p in record["probes"]} == {"current", "historical", "changes"}
    assert accessible_sources(m, "sam", "location", "current") == {"e2"}
    assert accessible_sources(m, "sam", "location", "historical") == {"e1"}
    assert record["limitation"] == "source_access_surrogate_not_real_future_answer_regret"
    assert m.validate_invariants()["valid"]


def test_probe_labels_are_observed_source_ids_not_question_references():
    m = memory(FutureQueryRegretMemory)
    m.ingest([event(1, "I live in Paris."), event(2, "I moved to Rome.")])
    probes = source_obligations(m, m.episodes["e2"])
    assert all(set(p.source_ids) <= set(m.episodes) for p in probes)
    assert not source_obligations(m, replace(m.episodes["e2"],
        features=replace(m.episodes["e2"].features, explicit_change=False)))
    assert probe_regret(m, probes) == 0


def test_unknown_or_ambiguous_query_cannot_bind():
    m = memory(FutureQueryRegretMemory)
    m.ingest([event(1, "I live in Paris."), event(2, "I moved to Rome.")])
    assert bind_query(m, "Where do they currently live?") is None
    assert bind_query(m, "Where does Sam live?") is None
    assert bind_query(m, "Where does Sam currently live?") == ("sam", "location", "current", ())
    assert m.retrieve("Where does Sam currently live?", 8)[0].memory_id == "e2"
    assert m.retrieve("Where did Sam previously live?", 8)[0].memory_id == "e1"


@pytest.mark.parametrize("query", [
    "Where does Sam's sister currently live?",
    "Where does Sam’s sister currently live?",
    "Where does the sister of Sam currently live?",
    "Where did Sam live before 2020?",
    "Where did Sam live in January previously?",
    "Where did Sam previously live last year?",
])
def test_nested_owner_and_date_constraints_are_unsupported(query):
    m = memory(FutureQueryRegretMemory)
    m.ingest([event(1, "I live in Paris."), event(2, "I moved to Rome.")])
    assert bind_query(m, query) is None
    m.retrieve(query, 8)
    assert not m.last_idea_certificate["binding_supported"]
    assert m.last_idea_certificate["anchor_count"] == 0


@pytest.mark.parametrize("cls", [FutureQueryRegretMemory, CompositionalEditMemory, ProofCarryingMemory])
def test_state_anchor_enters_pool_before_ce_with_strict_48_budget(monkeypatch, cls):
    m = memory(cls)
    events = [event(1, "I live in Paris.")]
    events += [replace(event(i, f"Tea coffee item {i}."), timestamp="2025-02-01")
               for i in range(2, 60)]
    events += [replace(event(60, "I moved to Rome."), timestamp="2025-03-01")]
    m.ingest(events)
    def fixed_base_pool(self, query):
        # Isolate the outside-pool admission contract from model score noise.
        return list(range(48)), defaultdict(set)
    monkeypatch.setattr(JointMindsetMemory, "candidate_pool", fixed_base_pool)
    assert m._index_by_id["e60"] not in range(48)
    hits = m.retrieve("Where does Sam currently live?", 8)
    assert len(m.last_candidate_pool) == len(set(m.last_candidate_pool)) == 48
    assert m.last_candidate_pool[:24] == tuple(range(24))
    assert m.local_reranker.calls == [48]
    assert len(hits) <= 8 and hits[0].memory_id == "e60"
    scored_ids = {m._episode_order[i] for i in m.last_candidate_pool}
    assert all(h.memory_id in scored_ids for h in hits)
    assert m.last_idea_certificate["anchor_count"] == 1
    assert not m.last_idea_certificate["oracle_used"]


def test_compositional_program_reassigns_confirming_exception():
    m = memory(CompositionalEditMemory)
    m.ingest([event(1, "I live in Paris."), event(2, "I live in Rome."), event(3, "I live in Rome.")])
    record = m.idea_records["e3"]
    assert record["program_length"] >= 2
    assert record["program"][0] == "version"
    assert any(isinstance(step, dict) and step.get("reassign") == ["e2"] for step in record["program"])
    active = [s for s in m.schemas.values() if s.status == "active"]
    assert len(active) == 1 and set(active[0].support_episode_ids) == {"e2", "e3"}
    historical = [s for s in m.schemas.values() if s.status == "historical"]
    assert historical[0].support_episode_ids == ["e1"]
    assert historical[0].exception_episode_ids == []
    assert m.validate_invariants()["grounded_episode_count"] == 3


def test_program_counterfactual_does_not_mutate_parent():
    m = memory()
    m.ingest([event(1, "I live in Paris."), event(2, "I live in Rome.")])
    incoming = replace(m.episodes["e2"], episode_id="e3", session_id="s3", timestamp="2025-01-03")
    plans = feasible_plans(m, incoming)
    plan = next(p for p in plans if p["mutation"] == "version")
    count = len(m.operation_log.all())
    state, steps = extend_program(m, incoming, plan)
    assert len(m.operation_log.all()) == count
    assert len(state.operation_log.all()) == count + 1
    assert len(steps) == 2
    assert m.local_reranker is state.local_reranker
    assert m.encoder is state.encoder


def test_compatible_schema_merge_is_real_and_does_not_mutate_parent_donor():
    m = memory()
    m.ingest([event(1, "I live in Paris.")])
    second = replace(m.episodes["e1"], episode_id="e2", session_id="s2", timestamp="2025-01-02")
    sid = next(iter(m.schemas))
    # A synthetic fragmented starting representation, no dataset labels.
    adopt_state(m, state_for_plan(m, second,
        {"schema_id": sid, "action": "split", "mutation": "branch", "cost": 0.}))
    incoming = replace(second, episode_id="e3", session_id="s3", timestamp="2025-01-03")
    plan = next(p for p in feasible_plans(m, incoming) if p["mutation"] == "merge")
    before = {s.schema_id: s.snapshot() for s in m.schemas.values()}
    state, steps = extend_program(m, incoming, plan)
    assert any(isinstance(step, dict) and "merge" in step for step in steps)
    assert {s.schema_id: s.snapshot() for s in m.schemas.values()} == before
    assert sum(s.status == "active" for s in state.schemas.values()) == 1
    assert state.validate_invariants()["grounded_episode_count"] == 3


def test_proof_certificate_names_sources_and_conditions_not_truth():
    m = memory(ProofCarryingMemory)
    m.ingest([event(1, "I live in Paris."), event(2, "I moved to Rome.")])
    cert = m.idea_records["e2"]["certificate"]
    assert cert["valid"]
    assert cert["premises"]["same_owner_slot"]
    assert cert["premises"]["strict_temporal_order"]
    assert cert["premises"]["conflicting_value"]
    assert not cert["premises"]["explicit_same_context"]
    assert cert["scope"] == "extracted_representation_only"
    assert cert["source_ids"] == ["e1", "e2"]


def test_proof_backdated_or_unsupported_change_is_deferred():
    m = memory(ProofCarryingMemory)
    m.ingest([event(3, "I live in Paris."), event(1, "I moved to Rome.")])
    assert m.idea_records["e1"]["deferred"]
    assert all(s.status == "active" for s in m.schemas.values())
    assert m.validate_invariants()["valid"]


def test_forged_features_cannot_pass_source_certificate():
    m = memory()
    m.ingest([event(1, "I live in Paris."), event(2, "I moved to Rome.")])
    episode = m.episodes["e2"]
    forged = replace(episode, features=replace(episode.features, attributes=(("location", "berlin"),)))
    cert = verify_plan(m, forged, {"schema_id": None, "action": "new", "mutation": "create", "cost": 0.})
    assert not cert["valid"] and not cert["premises"]["source_extraction_matches"]


def test_cross_owner_and_nonconflicting_splits_are_rejected():
    m = memory()
    m.ingest([event(1, "I live in Paris at home.")])
    incoming = replace(m.episodes["e1"], episode_id="e2", timestamp="2025-01-02", speaker="Rae")
    sid = next(iter(m.schemas))
    cert = verify_plan(m, incoming, {"schema_id": sid, "action": "split", "mutation": "branch", "cost": 0.})
    assert not cert["valid"]


def test_candidate_state_scores_structure_not_benchmark_labels():
    m = memory()
    m.ingest([event(1, "I live in Paris.")])
    incoming = replace(m.episodes["e1"], episode_id="e2", session_id="s2", timestamp="2025-01-02",
                       text="I moved to Rome.")
    from mindset.methods.joint import typed_features
    incoming = replace(incoming, features=typed_features(incoming.speaker, incoming.text))
    m.episodes[incoming.episode_id] = incoming
    probes = source_obligations(m, incoming)
    losses = {p["mutation"]: probe_regret(state_for_plan(m, incoming, p), probes)
              for p in feasible_plans(m, incoming)}
    assert losses["version"] == 0
    assert losses["exception"] > 0
