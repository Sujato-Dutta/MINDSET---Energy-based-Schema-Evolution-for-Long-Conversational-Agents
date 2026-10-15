"""Finite post-state diagnostic under provisional source-value interpretation.

No weights fitted. Lexicographic reconstruction/complexity is NOT claimed novel.
This compares actual readout consequences with a strong history-preserving rule.
"""
from copy import deepcopy
from mindset.utils.hashing import stable_hash

ACTIONS = ("reinforce", "overwrite", "revise", "split", "defer")


def post_state(old, new, scope, action):
    left = "shared" if scope == "same" else "left"
    right = "shared" if scope == "same" else "right"
    state = {"history": old, "current": {left: [old]}, "pending": [], "new_facets": 0}
    if action == "reinforce": return state
    if action == "overwrite":
        state["history"] = new; state["current"] = {right: [new]}
    elif action == "revise":
        state["current"] = {right: [new]}; state["new_facets"] = 1
    elif action == "split":
        state["current"].setdefault(right, []).append(new); state["new_facets"] = 1
    elif action == "defer":
        state["pending"] = [new]
    else: raise ValueError("Unknown diagnostic edit")
    return state


def score(state, old, new, scope, *, history=True, current=True):
    left = "shared" if scope == "same" else "left"
    right = "shared" if scope == "same" else "right"
    def agrees(values, expected):
        return {stable_hash(x) for x in values} == {stable_hash(expected)}
    old_loss = int(state["history"] != old) if history else 0
    current_loss = int(not agrees(state["current"].get(right, []), new)) if current else 0
    if scope == "different" and current:
        current_loss += int(not agrees(state["current"].get(left, []), old))
    return {"historical_readout_loss": old_loss, "current_readout_loss": current_loss,
        "objective": [old_loss + current_loss, state["new_facets"]]}


def compare(payload, decoded, *, history=True, current=True):
    admitted = (payload["kind"] == "value_change" and decoded["update_proposal"]) or (
        payload["kind"] == "same_value" and decoded.get("same_value_support_proposal", False))
    if not admitted:
        return {"energy_action": "defer", "rule_action": "defer", "evaluated": False,
            "reason": "no_admitted_distinct_value_interpretation", "semantic_correctness_verified": False}
    values = {p: {"kind": decoded["selected"][p]["value"]["value_kind"],
                  "value": deepcopy(decoded["selected"][p]["value"]["canonical"])} for p in ("a", "b")}
    if payload["direction"] == "b_to_a": old, new = values["b"], values["a"]
    else: old, new = values["a"], values["b"]
    plans = []
    for action in ACTIONS:
        state = post_state(old, new, payload["scope"], action)
        plans.append({"action": action, "post_state": state, "loss": score(state, old, new, payload["scope"], history=history, current=current)})
    chosen = min(plans, key=lambda p: (*p["loss"]["objective"], ACTIONS.index(p["action"])))
    rule = "split" if payload["scope"] == "different" else "reinforce" if old == new else "revise"
    return {"energy_action": chosen["action"], "rule_action": rule, "evaluated": True,
        "plans": plans, "selected_post_state": chosen["post_state"], "minimum_objective": chosen["loss"]["objective"],
        "actual_memory_mutated": False, "semantic_correctness_verified": False,
        "scope": "conditional two-report readout diagnostic; NOT learned world dynamics or benchmark QA"}
