import pytest
from mindset.memory.source_binding import normalize_binding, offsets, reference_uncertainty


def sources(a="My only employer is Orion Labs.", b="In 2018 my only employer was Cedar Works."):
    return [{"source_id": "A", "speaker": "user", "observation_time": "2025-01-01", "text": a},
            {"source_id": "B", "speaker": "user", "observation_time": "2025-06-01", "text": b}]


def payload(a, b):
    return {"kind": "attribute_change", "direction": "b_to_a", "variable": "employer", "a_quote": a, "b_quote": b}


def test_unique_rebinding_preserves_original_chronology():
    reports = sources(); p = payload(reports[1]["text"], reports[0]["text"])
    bound = normalize_binding(p, reports)
    assert bound["binding_status"] == "unique_literal_source_rebinding"
    assert [r["source_id"] for r in bound["bindings"]] == ["B", "A"]
    assert bound["direction"] == p["direction"] == "b_to_a"
    assert p["a_quote"] == reports[1]["text"]
    assert not bound["semantic_correctness_verified"]


def test_source_rebinding_does_not_use_observation_dates():
    reports = sources(); reports[0]["observation_time"], reports[1]["observation_time"] = "2040-01-01", "1980-01-01"
    bound = normalize_binding(payload(reports[1]["text"], reports[0]["text"]), reports)
    assert bound["direction"] == "b_to_a"
    assert bound["bindings"][0]["source_id"] == "B"


def test_nonchange_extra_quotes_remain_provenance_not_state():
    reports = sources("They had it before.", "They have another one now.")
    p = {"kind": "insufficient", "direction": "none", "variable": "", "a_quote": reports[0]["text"], "b_quote": reports[1]["text"]}
    bound = normalize_binding(p, reports)
    assert bound["nonchange_extra_fields_ignored_for_state"] and len(bound["bindings"]) == 2
    assert not bound["provisional_update_proposal"]


def test_fabricated_quote_not_repaired():
    with pytest.raises(ValueError): normalize_binding(payload("I am CEO.", "I was CTO."), sources())


def test_identical_positive_witnesses_rejected():
    with pytest.raises(ValueError): normalize_binding(payload("same", "same"), sources("same", "same"))


def test_repeated_literal_occurrences_retained():
    assert offsets("same same", "same") == [[0, 4], [5, 9]]


def test_unknown_direction_is_not_update():
    reports = sources(); p = payload(reports[0]["text"], reports[1]["text"]); p["direction"] = "unknown"
    assert not normalize_binding(p, reports)["provisional_update_proposal"]


def test_empty_negative_payload_keeps_no_atoms():
    p = {"kind": "event_progression", "direction": "none", "variable": "", "a_quote": "", "b_quote": ""}
    assert normalize_binding(p, sources())["bindings"] == []


def test_reference_guard_rejects_different_parser_text():
    class Doc:
        text = "wrong text"
    reports = sources(); p = payload(reports[0]["text"], reports[1]["text"])
    with pytest.raises(ValueError): reference_uncertainty(normalize_binding(p, reports), reports, {"A": Doc(), "B": Doc()})
