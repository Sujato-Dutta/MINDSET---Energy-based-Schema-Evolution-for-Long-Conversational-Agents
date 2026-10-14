"""CPU-only fixtures; execute through DGX Slurm, never on the laptop."""
from dataclasses import replace
from math import inf, log
import numpy as np
import pytest

from mindset.methods.compact import CompactConfig
from mindset.methods.joint import JointMindsetMemory
from mindset.methods.ideas_cdm_mdl import (
    ConformalDeferredMemory, MDLDescriptionMemory, action_nonconformity,
    conformal_actions, description_length, elias_gamma_nats, kt_nats,
    split_conformal_quantile, synthetic_cdm_corpus, _DescriptionCache,
)
from mindset.utils.hashing import stable_hash
from test_compact import Encoder, Reranker, event


def calibration(quantile=.7):
    records = ([{"history_id": f"independent-source-{i}", "score": quantile} for i in range(9)]
               if quantile is not None else [{"history_id": "independent-source", "score": .5}])
    return {"label_source": "independent_synthetic_trajectories", "records": records,
            "records_sha256": stable_hash(records), "quantile": quantile, "alpha": .1,
            "n_independent_histories": len(records)}


def memory(cls, **kwargs):
    return cls(CompactConfig(complexity_penalty=.1, switch_margin=.05),
               encoder=Encoder(), local_reranker=Reranker(), **kwargs)


def test_quantile_correct_n_plus_one_rank():
    assert split_conformal_quantile([.1, .2, .3, .4], .4) == .3
    assert split_conformal_quantile([.1, .2, .3, .4], .1) == inf
    # Exact boundary: n=9, alpha=.1 chooses n-th score, not interpolation.
    assert split_conformal_quantile([i/10 for i in range(9)], .1) == .8


@pytest.mark.parametrize("scores,alpha", [([], .1), ([float("nan")], .1), ([2.], .1), ([.1], 0)])
def test_quantile_rejects_invalid(scores, alpha):
    with pytest.raises(ValueError):
        split_conformal_quantile(scores, alpha)


def test_action_score_aggregates_targets_not_duplicate_probability_mass():
    plans = [{"action": "reinforce", "cost": 0.}, {"action": "reinforce", "cost": 1.},
             {"action": "supersede", "cost": 0.}]
    assert action_nonconformity(plans) == {"reinforce": .5, "supersede": .5}
    assert conformal_actions(plans, calibration(.5)) == ("reinforce", "supersede")
    assert conformal_actions(plans, calibration(.49)) == ()
    assert conformal_actions(plans, calibration(None)) == ("reinforce", "supersede")


def test_corpus_one_observation_per_independent_history_and_repeatable():
    a = synthetic_cdm_corpus(seed=831, histories=60)
    b = synthetic_cdm_corpus(seed=831, histories=60)
    assert a == b
    assert len({h["id"] for h in a}) == 60
    assert all(h["id"].startswith("pilot-cdm-") for h in a)
    assert all(len(h["events"]) == len(h["actions"]) == 3 for h in a)
    assert all(h["calibration_index"] in (1, 2) for h in a)


def test_cdm_rejects_tampered_or_nonindependent_calibration():
    artifact = calibration()
    artifact["records"][0]["score"] = .3
    with pytest.raises(ValueError, match="changed"):
        memory(ConformalDeferredMemory, calibration=artifact)
    artifact = calibration()
    artifact["label_source"] = "energy_predictions"
    with pytest.raises(ValueError, match="independent"):
        memory(ConformalDeferredMemory, calibration=artifact)


def test_cdm_deferral_keeps_active_state_and_raw_sources():
    m = memory(ConformalDeferredMemory, calibration=calibration(None))
    m.ingest([event(1, "I live in Paris."), event(2, "I moved to Rome.")])
    assert m._decision_certificates["e2"]["deferred"]
    assert m.operation_log.all()[-1].stored_as_exception
    assert all(s.status == "active" for s in m.schemas.values())
    assert set(m.episodes) == {"e1", "e2"}
    assert m.validate_invariants()["grounded_episode_count"] == 2
    assert m.shadow_states["e2"]["resolved_by"] is None
    assert set(m.shadow_states["e2"]["plausible_actions"]) == {"reinforce", "supersede"}
    assert m.telemetry()["natural_coverage_guarantee"] is False


def test_cdm_empty_prediction_set_does_not_confidently_commit_change():
    m = memory(ConformalDeferredMemory, calibration=calibration(0.))
    m.ingest([event(1, "I live in Paris."), event(2, "I moved to Rome.")])
    assert m._decision_certificates["e2"]["action_prediction_set"] == []
    assert all(s.status == "active" for s in m.schemas.values())
    assert m.validate_invariants()["grounded_episode_count"] == 2


def test_cdm_threshold_must_be_actual_finite_sample_quantile():
    artifact = calibration(.7)
    artifact["quantile"] = .6
    with pytest.raises(ValueError, match="quantile"):
        memory(ConformalDeferredMemory, calibration=artifact)


def test_cdm_resolution_requires_later_independent_matching_observation(monkeypatch):
    import mindset.methods.ideas_cdm_mdl as module
    m = memory(ConformalDeferredMemory, calibration=calibration(None))
    m.ingest([event(1, "I live in Paris."), event(2, "I moved to Rome.")])
    monkeypatch.setattr(module, "conformal_actions", lambda p, a: ("supersede",))
    m.ingest([event(3, "I live in Rome.")])
    assert m.shadow_states["e2"]["resolved_by"] == "e3"
    assert m.shadow_states["e2"]["resolved_action"] == "supersede"
    # Older raw source remains; resolution doesn't fabricate retrospective truth.
    assert m.episodes["e2"].text == "I moved to Rome."
    assert m.validate_invariants()["grounded_episode_count"] == 3


def test_gamma_code_and_kt_exact_coding_identity():
    assert elias_gamma_nats(1) == pytest.approx(log(2))
    assert elias_gamma_nats(4) == pytest.approx(5 * log(2))
    assert kt_nats([1], 2) == pytest.approx(log(2))
    assert kt_nats([2], 2) == pytest.approx(-log(.5 * .75))
    assert kt_nats([1, 1], 2) == pytest.approx(-log(.5 * .25))
    with pytest.raises(ValueError):
        elias_gamma_nats(0)


def test_mdl_cost_nonnegative_preserves_sources_and_selects_code_minimum():
    m = memory(MDLDescriptionMemory)
    m.ingest([event(1, "I live in Paris."), event(2, "I moved to Rome."),
              event(3, "I live in Rome."), event(4, "I prefer tea at home.")])
    assert m.validate_invariants()["grounded_episode_count"] == 4
    assert description_length(m)["total"] > 0
    for cert in m._decision_certificates.values():
        assert cert["selected_code_nats"]["total"] == min(
            c["code_nats"]["total"] for c in cert["feasible_candidates"])
    assert m.telemetry()["base_energy_used_for_selection"] is False


def test_mdl_not_current_weighted_energy_and_bounded_common_retrieval():
    m = memory(MDLDescriptionMemory)
    m.ingest([event(i, f"Tea home item {i}.") for i in range(1, 61)])
    hits = m.retrieve("tea", 8)
    assert len(hits) == 8 and m.local_reranker.calls == [48]
    assert all(h.source_event_ids == (h.memory_id,) for h in hits)
    cert = m._decision_certificates["e2"]
    assert cert["selected_code_nats"]["total"] != cert["feasible_candidates"][0].get("cost", -1)


def test_mdl_cost_ignores_answer_question_or_gold_metadata():
    m = memory(MDLDescriptionMemory)
    m.ingest([event(1, "I live in Paris."), event(2, "I moved to Rome.")])
    before = description_length(m)
    m.episodes["e1"] = replace(m.episodes["e1"], metadata={"answer": "fabricated", "gold": ["e2"]})
    assert description_length(m) == before


def test_mdl_code_rejects_unencoded_duplicate_source_assignment():
    m = memory(MDLDescriptionMemory)
    m.ingest([event(1, "I live in Paris."), event(2, "I prefer tea at home.")])
    schemas = list(m.schemas.values())
    schemas[1].exception_episode_ids.append("e1")
    with pytest.raises(ValueError, match="exactly one"):
        description_length(m)


def test_mdl_batched_and_incremental_identical_controller_decisions():
    events = [event(1, "I live in Paris."), event(2, "I live in Rome."),
              event(3, "I live in Rome."), event(4, "I prefer tea at home."),
              event(5, "I prefer coffee at work.")]
    whole = memory(MDLDescriptionMemory)
    incremental = memory(MDLDescriptionMemory)
    whole.ingest(events)
    for item in events:
        incremental.ingest([item])
    assert [r.operation for r in whole.operation_log.all()] == [r.operation for r in incremental.operation_log.all()]
    assert description_length(whole) == description_length(incremental)


def test_mdl_cached_and_fresh_codes_identical_at_every_prefix():
    m = memory(MDLDescriptionMemory)
    for item in [event(1, "I live in Paris."), event(2, "I moved to Rome."),
                 event(3, "I live in Rome."), event(4, "I prefer tea at home."),
                 event(5, "I prefer coffee at work.")]:
        m.ingest([item])
        assert description_length(m) == description_length(m, use_cache=False)
        assert description_length(m) == description_length(m, use_cache=False)
    assert m._mdl_description_cache.hits["schema_length"] > 0


def test_mdl_cache_invalidates_observation_changes_and_alphabet_growth():
    m = memory(MDLDescriptionMemory)
    m.ingest([event(1, "I live in Paris."), event(2, "I prefer tea at home.")])
    old = description_length(m)
    source = m.episodes["e1"]
    features = replace(source.features, tokens=(*source.features.tokens, "new-symbol"))
    m.episodes["e1"] = replace(source, features=features)
    current = description_length(m)
    assert current == description_length(m, use_cache=False)
    assert current != old
    assert m._mdl_description_cache.hits["rebuilt_prefix"] > 0


def test_mdl_cache_is_bounded_and_keys_are_branch_specific():
    cache = _DescriptionCache(max_statistics=2, max_lengths=3)
    for i in range(10):
        cache._remember(cache.statistics, (i,), {"cost": i}, cache.max_statistics)
        cache._remember(cache.lengths, (i,), float(i), cache.max_lengths)
    assert len(cache.statistics) == 2
    assert len(cache.lengths) == 3
    assert (0,) not in cache.statistics and (9,) in cache.statistics
