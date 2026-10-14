import importlib.util
from pathlib import Path
import pytest
from test_energy_identifiability import plan, cert

spec = importlib.util.spec_from_file_location("energy_margin_audit",
    Path(__file__).resolve().parents[1] / "scripts/verify_energy_identifiability.py")
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def test_tie_changes_are_not_strict_preferences():
    effect = worker.margins(cert([plan("reinforce", "merge", d=.2),
                                 plan("new", "create", price=.1)]), "distortion")
    assert effect["decision_changed"] and effect["action_advantage"] == pytest.approx(.1)
    effect = worker.margins(cert([plan("reinforce", "merge", d=.2),
                                 plan("new", "create", price=0.)]), "distortion")
    assert effect["decision_changed"] and effect["action_advantage"] == 0.


def test_target_change_does_not_prove_action_necessity():
    effect = worker.margins(cert([plan("reinforce", "merge", sid="s1", h=.2),
                                 plan("reinforce", "merge", sid="s2", d=.1)]), "historical_damage")
    assert effect["schema_id_advantage"] == pytest.approx(.1)
    assert effect["action_advantage"] == 0.


def test_zero_energy_is_not_numerical_margin_test():
    with pytest.raises(ValueError): worker.margins(cert([plan("reinforce", "merge")]), "energy_off")
