from types import SimpleNamespace as N
from mindset.memory.source_reference_guard import reference_uncertainty
from mindset.memory.source_binding import normalize_binding


class Doc(list):
    def __init__(self, text, tokens):
        super().__init__(tokens); self.text = text


def test_opaque_one_is_held_even_when_tagged_noun():
    a = "I repaired my bicycle chain."; b = "I replaced my old one."
    reports = [{"source_id": sid, "speaker": "user", "observation_time": date, "text": text}
        for sid, date, text in [("A", "2025-01-01", a), ("B", "2025-06-01", b)]]
    binding = normalize_binding({"kind": "attribute_change", "variable": "bike condition", "direction": "a_to_b",
        "a_quote": a, "b_quote": b}, reports)
    t = N(text="one", idx=b.index("one"), lemma_="one", pos_="NOUN", dep_="dobj", children=[])
    result = reference_uncertainty(binding, reports, {"A": Doc(a, []), "B": Doc(b, [t])})
    assert not result["update_proposal"]
    assert result["warnings"][0]["reason"] == "opaque_argument_head_independent_of_pronoun_tag"
