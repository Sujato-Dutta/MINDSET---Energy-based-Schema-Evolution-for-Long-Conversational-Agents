"""Same-state decision attribution, NOT a causal QA or novelty certificate.

The feasible set is held fixed. Removing an energy term here does not rebuild
the history, change its eligibility rules, or measure the resulting answers.
"""
from __future__ import annotations

import math

TERMS = ("distortion", "contradiction", "historical_damage",
         "fragmentation_price", "inconsistency", "switching_cost")
PRIORITY = {"reinforce": 0, "supersede": 1, "split": 2, "new": 3}


def signature(plan):
    return (plan["schema_id"], plan["action"], plan["mutation"])


def contributions(plan):
    raw = plan["terms"]
    values = {name: float(raw[name]) for name in
              ("distortion", "contradiction", "historical_damage", "inconsistency")}
    # Raw fragmentation is a schema count, not its calibrated energy price.
    values["fragmentation_price"] = float(raw["total"]) - sum(values.values())
    values["switching_cost"] = float(plan["adjusted_energy"]) - float(raw["total"])
    if not all(math.isfinite(v) and v >= -1e-8 for v in values.values()):
        raise ValueError("Invalid or unreconstructable energy terms")
    return values


def choose(plans, omitted=(), *, energy_off=False):
    if not plans or set(omitted) - set(TERMS):
        raise ValueError("Missing candidates or unknown energy term")
    def key(plan):
        values = contributions(plan)
        # Preserve the implementation's exact floating-point comparison at the
        # unablated baseline; re-summing in another order can perturb a tie.
        cost = 0.0 if energy_off else float(plan["adjusted_energy"]) - sum(values[k] for k in omitted)
        return (cost, PRIORITY[plan["action"]], plan["schema_id"])
    return min(plans, key=key)


def attribute(certificate):
    plans = certificate["feasible_candidates"]
    if len({signature(p) for p in plans}) != len(plans):
        raise ValueError("Duplicate feasible decisions")
    selected = choose(plans)
    if (selected["schema_id"] != certificate["selected_schema_id"] or
            selected["mutation"] != certificate["selected_mutation"] or
            not math.isclose(float(selected["adjusted_energy"]),
                             float(certificate["selected_adjusted_energy"]), abs_tol=1e-8)):
        raise ValueError("Certificate differs from the implemented minimizer")
    ordered_costs = sorted(float(p["adjusted_energy"]) for p in plans)
    controls = {}
    for name in (*TERMS, "energy_off"):
        alternative = choose(plans, (name,) if name != "energy_off" else (),
                             energy_off=name == "energy_off")
        controls[name] = {
            "selected": signature(alternative),
            "decision_changed": signature(alternative) != signature(selected),
            "action_changed": alternative["action"] != selected["action"],
            "mutation_changed": alternative["mutation"] != selected["mutation"],
            "target_changed": alternative["schema_id"] != selected["schema_id"],
        }
    return {
        "selected": signature(selected), "candidate_count": len(plans),
        "distinct_actions": len({p["action"] for p in plans}),
        "distinct_mutations": len({p["mutation"] for p in plans}),
        "distinct_targets": len({p["schema_id"] for p in plans}),
        "runner_up_gap": ordered_costs[1] - ordered_costs[0] if len(plans) > 1 else None,
        "controls": controls,
        "claim_scope": "one-step fixed-state attribution; no semantic correctness or QA effect",
    }
