from types import SimpleNamespace as N
from copy import deepcopy
import pytest
from mindset.memory.value_pointing import number_literal, decode
from mindset.utils.hashing import stable_hash


class Doc(list):
    def __init__(self,text): super().__init__(); self.text=text


def fixture(a="I volunteer.", b="I am volunteering.", ca=None, cb=None, family="predicate"):
    if family == "number":
        a = f"I lead {ca} engineers."; b = f"I lead {cb} engineers."
    reports = [{"source_id":sid,"speaker":"user","observation_time":date,"text":text}
        for sid,date,text in [("A","2025-01-01",a),("B","2025-06-01",b)]]
    units=[]
    for sid,text,canonical in [("A",a,ca),("B",b,cb)]:
        units.append({"id":sid+":u000","source_id":sid,"char_start":0,"char_end":len(text),"text":text,
            "scope_span":[0,len(text)],"role":"assertion","kind":"full_report"})
        word = str(canonical) if family == "number" else "volunteer" if sid == "A" else "volunteering"
        start = text.index(word)
        units.append({"id":sid+":v000","source_id":sid,"char_start":start,"char_end":start+len(word),"text":word,
            "scope_span":[0,len(text)],"role":"value","value_kind":family,
            "canonical":canonical or {"lemma":"volunteer","negation":[],"modals":[]}})
    registry = {"reports_sha256":stable_hash(reports),"registry_sha256":stable_hash(units),"units":units}
    p={"kind":"value_change","variable":"volunteering status","scope":"same","direction":"a_to_b",
        "a_assertion":"A:u000","b_assertion":"B:u000","a_value":"A:v000","b_value":"B:v000"}
    return reports,registry,p,{r["source_id"]:Doc(r["text"]) for r in reports}


def test_morphological_rephrasing_is_not_distinct_value_evidence():
    reports,v,p,docs=fixture(); d=decode(p,v,reports,docs)
    assert d["canonical_values_equal"] and not d["update_proposal"]
    assert "selected_values_normalize_identically" in d["holds"]
    assert d["proposed_kind"] == "value_change"  # Never rewrite original model label.


def test_distinct_values_do_not_certify_property_or_world_truth():
    reports,v,p,docs=fixture(ca="4",cb="5",family="number")
    d=decode(p,v,reports,docs)
    assert d["update_proposal"] and not d["semantic_correctness_verified"]
    assert d["same_property_holder_scope_is_model_interpretation"]


def test_unknown_context_is_retained_as_an_uncertainty_hold():
    reports,v,p,docs=fixture(ca="4",cb="5",family="number");p["scope"]="unknown"
    assert not decode(p,v,reports,docs)["update_proposal"]


def test_missing_value_does_not_become_a_positive_state():
    reports,v,p,docs=fixture();p["a_value"]="none"
    d=decode(p,v,reports,docs);assert not d["complete_value_witnesses"] and not d["update_proposal"]


def test_benign_event_provenance_is_allowed_without_state_admission():
    reports,v,p,docs=fixture();p["kind"]="event_progression"
    assert not decode(p,v,reports,docs)["update_proposal"]


@pytest.mark.parametrize("field,replacement",[("a_value","B:v000"),("b_value","5"),("a_assertion","A:v000")])
def test_generated_or_wrong_role_pointer_rejected(field,replacement):
    reports,v,p,docs=fixture();p[field]=replacement
    with pytest.raises(ValueError):decode(p,v,reports,docs)


def test_tampered_value_literal_rejected_even_with_rehashed_registry():
    reports,v,p,docs=fixture();v=deepcopy(v);v["units"][1]["text"]="active"
    v["registry_sha256"]=stable_hash(v["units"])
    with pytest.raises(ValueError):decode(p,v,reports,docs)


def test_rehashed_number_metadata_cannot_silently_change_a_literal_count():
    reports,v,p,docs=fixture(ca="4",cb="5",family="number")
    v["units"][1]["canonical"]="999"; v["registry_sha256"]=stable_hash(v["units"])
    with pytest.raises(ValueError):decode(p,v,reports,docs)


def test_different_activity_verbs_do_not_certify_values_of_one_status():
    reports,v,p,docs=fixture()
    v["units"][1]["canonical"]["lemma"]="give"
    v["registry_sha256"]=stable_hash(v["units"])
    d=decode(p,v,reports,docs)
    assert not d["update_proposal"] and "different_predicates_are_not_values_of_one_predicate" in d["holds"]


def test_different_numeric_units_remain_unresolved():
    reports,v,p,docs=fixture(ca="4",cb="5",family="number")
    v["units"][1]["grammatical_head"]="engineer";v["units"][3]["grammatical_head"]="gigabyte"
    v["registry_sha256"]=stable_hash(v["units"])
    assert "numeric_unit_correspondence_unresolved" in decode(p,v,reports,docs)["holds"]


@pytest.mark.parametrize("text,expected",[("four","4"),("5","5"),("one hundred","100"),("100 million",None),("5th",None),("three bananas",None)])
def test_number_normalization_does_not_drop_mixed_digit_word_parts(text,expected):
    assert number_literal(text)==expected
