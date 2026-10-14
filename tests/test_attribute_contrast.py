import importlib.util
from pathlib import Path
import sys
import pytest
from mindset.memory.attribute_contrast_contract import contrast_messages, validate_contrast


def reports():
    return [{"source_id": "A", "speaker": "user", "observation_time": "2025-01-01", "text": "I led three analysts."},
            {"source_id": "B", "speaker": "user", "observation_time": "2025-06-01", "text": "I now lead eight analysts."}]


def payload():
    return {"kind": "attribute_change", "variable": "team size", "a_quote": "I led three analysts.",
            "b_quote": "I now lead eight analysts.", "direction": "a_to_b"}


def test_quotes_are_not_semantic_truth():
    cert = validate_contrast(payload(), reports())
    assert cert["literal_quotes_verified"] and cert["provisional_update_proposal"]
    assert not cert["semantic_correctness_verified"] and not cert["world_truth_or_update_verified"]


def test_metadata_is_not_literal_factual_time():
    p = payload(); p["a_quote"] = "2025-01-01"
    with pytest.raises(ValueError): validate_contrast(p, reports())


def test_wrong_source_quote_rejected():
    p = payload(); p["a_quote"] = p["b_quote"]
    with pytest.raises(ValueError): validate_contrast(p, reports())


def test_unresolved_chronology_is_not_update_proposal():
    p = payload(); p["direction"] = "unknown"
    assert not validate_contrast(p, reports())["provisional_update_proposal"]


def test_nonchanges_do_not_generate_atoms():
    p = {"kind": "event_progression", "variable": "", "a_quote": "", "b_quote": "", "direction": "none"}
    assert not validate_contrast(p, reports())["provisional_update_proposal"]
    p["variable"] = "life phase"
    with pytest.raises(ValueError): validate_contrast(p, reports())


def test_only_reports_and_schema_in_prompt():
    assert len(contrast_messages(reports())) == 2
    with pytest.raises(ValueError): contrast_messages([reports()[0]])


def test_fixed_cases_keep_labels_out_of_model_input():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("contrast_fixtures", scripts / "attribute_contrast_fixtures.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        rows = module.contrast_fixtures()
        assert len(rows) == 32 and len({r["id"] for r in rows}) == 32
        assert sum(r["expected_kind"] == "attribute_change" for r in rows) == 10
        assert all("expected_kind" not in contrast_messages(r["reports"])[1]["content"] for r in rows)
    finally:
        sys.path.remove(str(scripts))
