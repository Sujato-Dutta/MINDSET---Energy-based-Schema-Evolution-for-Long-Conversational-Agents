from types import SimpleNamespace as N
from mindset.memory.source_binding_candidate import normalize_binding, reference_uncertainty


class Doc(list):
    def __init__(self, text, tokens):
        super().__init__(tokens); self.text = text


def token(text, index, pos, dep, children=()):
    return N(text=text, idx=index, lemma_=text.lower(), pos_=pos, dep_=dep, children=list(children))


def fixture():
    reports = [{"source_id": "A", "speaker": "user", "observation_time": "2025-01-01", "text": "My commute was by train."},
               {"source_id": "B", "speaker": "user", "observation_time": "2025-06-01", "text": "Her commute is by bicycle."}]
    p = {"kind": "attribute_change", "direction": "a_to_b", "variable": "commute",
         "a_quote": reports[0]["text"], "b_quote": reports[1]["text"]}
    return reports, normalize_binding(p, reports)


def test_possessive_holder_is_unresolved_even_if_tagger_calls_it_determiner():
    reports, binding = fixture()
    docs = {"A": Doc(reports[0]["text"], [token("My", 0, "PRON", "poss")]),
            "B": Doc(reports[1]["text"], [token("Her", 0, "DET", "poss")])}
    result = reference_uncertainty(binding, reports, docs)
    assert not result["update_proposal"] and result["warnings"][0]["text"] == "Her"


def test_guard_does_not_guess_named_holder_for_third_person():
    reports, binding = fixture()
    docs = {"A": Doc(reports[0]["text"], []),
            "B": Doc(reports[1]["text"], [token("Her", 0, "PRON", "poss")])}
    assert not reference_uncertainty(binding, reports, docs)["update_proposal"]
