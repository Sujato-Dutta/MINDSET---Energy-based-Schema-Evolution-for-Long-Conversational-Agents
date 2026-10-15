from copy import deepcopy
from types import SimpleNamespace as N
import pytest
from mindset.memory.source_pointing import registry, validate_registry, decode_pointer, schema
from mindset.utils.hashing import stable_hash


def fixture(a="I lead four engineers.", b="I now lead five engineers."):
    reports = [{"source_id": sid, "speaker": "user", "observation_time": date, "text": text}
               for sid, date, text in [("A", "2025-01-01", a), ("B", "2025-06-01", b)]]
    docs = {r["source_id"]: N(text=r["text"], sents=[]) for r in reports}
    return reports, registry(reports, docs)


def payload():
    return {"kind": "attribute_change", "variable": "team size", "direction": "a_to_b",
            "a_unit": "A:u000", "b_unit": "B:u000"}


def test_host_copies_exact_source_not_generated_quote():
    reports, value = fixture()
    result = decode_pointer(payload(), value, reports)
    assert [q["quote"] for q in result["binding"]["bindings"]] == [r["text"] for r in reports]
    assert not result["generated_evidence_text"] and not result["semantic_correctness_verified"]
    assert "five" in result["binding"]["bindings"][1]["quote"]


@pytest.mark.parametrize("key,replacement", [("a_unit", "B:u000"), ("b_unit", "A:u000"),
    ("a_unit", "A:missing"), ("b_unit", "I now lead 5 engineers."), ("kind", "confirmed_fact")])
def test_invalid_pointer_rejected(key, replacement):
    reports, value = fixture(); p = payload(); p[key] = replacement
    with pytest.raises(ValueError): decode_pointer(p, value, reports)


def test_tampered_text_rejected_even_if_registry_rehashed():
    reports, value = fixture(); value = deepcopy(value)
    value["units"][1]["text"] = "I now lead 5 engineers."
    value["registry_sha256"] = stable_hash(value["units"])
    with pytest.raises(ValueError): validate_registry(value, reports)


def test_containing_scope_must_cover_interval():
    reports, value = fixture(); value["units"][0]["scope_span"] = [1, 3]
    value["registry_sha256"] = stable_hash(value["units"])
    with pytest.raises(ValueError): validate_registry(value, reports)


def test_nonchange_cannot_smuggle_witnesses():
    reports, value = fixture(); p = payload(); p.update(kind="insufficient", direction="none", variable="")
    with pytest.raises(ValueError): decode_pointer(p, value, reports)
    p.update(a_unit="none", b_unit="none")
    assert not decode_pointer(p, value, reports)["binding"]["provisional_update_proposal"]


def test_repeated_identical_report_is_not_positive_witness_pair():
    reports, value = fixture("My count jumped from 420 to 540.", "My count jumped from 420 to 540.")
    with pytest.raises(ValueError): decode_pointer(payload(), value, reports)


def test_schema_binds_each_source_independently():
    reports, value = fixture(); s = schema(value, reports)
    assert s["properties"]["a_unit"]["enum"] == ["none", "A:u000"]
    assert s["properties"]["b_unit"]["enum"] == ["none", "B:u000"]
