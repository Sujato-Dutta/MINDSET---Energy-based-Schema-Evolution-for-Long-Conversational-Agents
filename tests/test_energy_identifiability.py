import pytest
from mindset.evaluation.energy_identifiability import attribute, choose, contributions


def plan(action, mutation, *, sid="s1", d=0., c=0., h=0., b=0., price=0., switch=0.):
    total = d + c + h + b + price
    return {"schema_id": sid, "action": action, "mutation": mutation,
            "terms": {"distortion": d, "contradiction": c, "historical_damage": h,
                      "inconsistency": b, "fragmentation": 1., "total": total},
            "adjusted_energy": total + switch}


def cert(plans):
    chosen = choose(plans)
    return {"feasible_candidates": plans, "selected_schema_id": chosen["schema_id"],
            "selected_mutation": chosen["mutation"],
            "selected_adjusted_energy": chosen["adjusted_energy"]}


def test_weighted_fragmentation_not_raw_count():
    assert contributions(plan("new", "create", price=.17))["fragmentation_price"] == .17


def test_action_attribution_and_energy_off():
    result = attribute(cert([plan("reinforce", "exception", c=1.),
                             plan("supersede", "version", price=.2)]))
    assert result["controls"]["contradiction"]["action_changed"]
    assert result["controls"]["energy_off"]["action_changed"]


def test_switching_cost_can_preserve_current_state():
    result = attribute(cert([plan("reinforce", "exception", c=.23),
                             plan("supersede", "version", price=.2, switch=.05)]))
    assert result["controls"]["switching_cost"]["mutation_changed"]


def test_target_only_change_is_not_action_change():
    result = attribute(cert([plan("reinforce", "merge", sid="s1", h=.2),
                             plan("reinforce", "merge", sid="s2", d=.1)]))
    control = result["controls"]["historical_damage"]
    assert control["target_changed"] and not control["action_changed"]


def test_single_candidate_is_constraint_determined():
    result = attribute(cert([plan("reinforce", "merge", d=.7)]))
    assert result["candidate_count"] == 1 and result["runner_up_gap"] is None
    assert not any(v["decision_changed"] for v in result["controls"].values())


def test_exact_ties_preserve_implemented_order():
    assert choose([plan("supersede", "version"), plan("reinforce", "merge")])["action"] == "reinforce"


def test_corrupt_certificate_rejected():
    value = cert([plan("reinforce", "merge")]); value["selected_schema_id"] = "wrong"
    with pytest.raises(ValueError): attribute(value)


def test_nonfinite_energy_rejected():
    with pytest.raises(ValueError): contributions(plan("reinforce", "merge", d=float("nan")))


def test_observer_preserves_joint_decisions_and_states():
    import importlib.util
    from pathlib import Path
    from mindset.methods.joint import JointMindsetMemory
    from mindset.methods.compact import CompactConfig
    from test_compact import Encoder, event
    spec = importlib.util.spec_from_file_location("energy_observer",
        Path(__file__).resolve().parents[1] / "scripts/audit_energy_identifiability.py")
    worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    config = CompactConfig(complexity_penalty=.1, switch_margin=.05)
    baseline = JointMindsetMemory(config, encoder=Encoder())
    observed = worker.ObservedJoint(config, encoder=Encoder()); observed._observer_rows = []
    events = [event(1, "I live in Paris."), event(2, "I live in Rome."),
              event(3, "I live in Rome."), event(4, "I prefer tea at home."),
              event(5, "I prefer coffee at work."), event(6, "I prefer tea at home.")]
    baseline.ingest(events); observed.ingest(events)
    assert observed.operation_log.all() == baseline.operation_log.all()
    assert observed._decision_certificates == baseline._decision_certificates
    assert {k: v.snapshot() for k, v in observed.schemas.items()} == {
        k: v.snapshot() for k, v in baseline.schemas.items()}
    assert observed.validate_invariants() == baseline.validate_invariants()
    assert len(observed._observer_rows) == len(events)
