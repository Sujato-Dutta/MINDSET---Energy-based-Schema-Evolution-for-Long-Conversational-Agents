import importlib.util
from pathlib import Path
from mindset.memory.attribute_contrast_contract import contrast_messages

spec = importlib.util.spec_from_file_location("bound_contrast_controls",
    Path(__file__).resolve().parents[1] / "scripts/bound_contrast_controls.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_fixed_new_controls_do_not_put_labels_in_model_input():
    rows = module.fresh_controls()
    assert len(rows) == len({r["id"] for r in rows}) == 12
    assert sum(r["expected_kind"] == "attribute_change" for r in rows) == 6
    assert sum(r["expected_direction"] == "b_to_a" for r in rows) == 2
    assert all("expected_kind" not in contrast_messages(r["reports"])[1]["content"] for r in rows)
