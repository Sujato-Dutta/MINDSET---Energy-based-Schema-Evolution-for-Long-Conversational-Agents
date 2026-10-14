"""Isolated one-price Joint candidate; not frozen production or a novelty claim.

Remove an empirically inactive switching margin, while retaining the original
calibrated creation price, feasible-state rules, operations and retrieval.
Inherited lambda was selected in the ORIGINAL joint calibration; this does not
pretend it was independently fitted for a one-parameter model.
"""
from dataclasses import replace
from mindset.methods.compact import CompactConfig
from mindset.methods.joint import JointMindsetMemory


class OnePriceJointMemory(JointMindsetMemory):
    def __init__(self, config=None, **kwargs):
        original = config or CompactConfig()
        self.original_switch_margin = original.switch_margin
        super().__init__(replace(original, switch_margin=0.), **kwargs)

    def telemetry(self):
        return {**super().telemetry(), "candidate": "one_price_joint_isolated",
            "decision_scalars": {"complexity_penalty": self.config.complexity_penalty},
            "removed_original_switch_margin": self.original_switch_margin,
            "calibration_scope": "lambda inherited from original two-scalar calibration; no new fitting",
            "scope": "cohort equivalence must be checked; no global equivalence or novelty claim"}
