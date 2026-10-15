from copy import deepcopy
import pytest
from mindset.memory.value_reconstruction import compare, post_state, score


def inputs(scope="same", direction="a_to_b"):
    p = {"kind": "value_change", "scope": scope, "direction": direction}
    d = {"update_proposal": True, "selected": {k:{"value":{"value_kind":"number", "canonical":v}}
        for k,v in (("a","4"),("b","5"))}}
    return p,d


@pytest.mark.parametrize("scope,expected", [("same","revise"),("different","split")])
def test_reconstruction_agrees_with_strong_rule_when_interpretation_is_given(scope, expected):
    p,d = inputs(scope); r = compare(p,d)
    assert r["energy_action"] == r["rule_action"] == expected
    assert r["minimum_objective"] == [0,1]
    assert not r["actual_memory_mutated"]


def test_history_ablation_exposes_actual_overwrite_loss():
    p,d = inputs(); r = compare(p,d,history=False)
    assert r["energy_action"] == "overwrite" and r["rule_action"] == "revise"
    plan = next(x for x in r["plans"] if x["action"] == "overwrite")
    assert score(plan["post_state"], {"kind":"number","value":"4"}, {"kind":"number","value":"5"}, "same")["historical_readout_loss"] == 1


def test_current_ablation_retains_old_value_without_evidence_of_a_gain():
    p,d = inputs(); assert compare(p,d,current=False)["energy_action"] == "reinforce"


def test_unknown_interpretation_cannot_be_promoted_by_optimizer():
    p,d = inputs(); d["update_proposal"] = False
    assert compare(p,d)["energy_action"] == "defer" and not compare(p,d)["evaluated"]


def test_source_direction_is_separate_from_ab_order():
    p,d = inputs(direction="b_to_a"); r = compare(p,d)
    assert r["selected_post_state"]["history"]["value"] == "5"
    assert r["selected_post_state"]["current"]["shared"][0]["value"] == "4"


def test_input_values_and_states_not_mutated_by_plans():
    p,d = inputs(); before = deepcopy(d); compare(p,d)
    assert d == before
    assert post_state("4","5","same","reinforce")["history"] == "4"


def test_same_value_reinforcement_without_creating_a_version():
    p,d=inputs();p["kind"]="same_value"; d["selected"]["b"]["value"]["canonical"]="4"
    d["update_proposal"]=False;d["same_value_support_proposal"]=True
    r=compare(p,d);assert r["energy_action"]==r["rule_action"]=="reinforce"
    assert r["minimum_objective"]==[0,0]
