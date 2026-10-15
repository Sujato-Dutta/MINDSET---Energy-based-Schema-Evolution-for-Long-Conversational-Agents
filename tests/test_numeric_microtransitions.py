from types import SimpleNamespace as N
from mindset.memory.numeric_microtransitions import value, collapsed_from_to


def test_to_phrase_cannot_become_part_of_from_endpoint():
    text = "420 to 540"
    after = N(text="540", idx=7, i=2, ent_type_="", dep_="pobj", children=[])
    to = N(text="to", idx=4, i=1, dep_="prep", children=[after], subtree=[])
    to.subtree = [to, after]
    number = N(text="420", idx=0, i=0, pos_="NUM", ent_type_="", dep_="pobj", children=[to], subtree=[])
    number.subtree = [number, to, after]
    before = value(N(children=[number]), text)
    assert before["text"] == "420" and before["numeric_normalization_alternative"] == "420"


def test_date_value_not_silently_counted_as_quantity():
    token = N(text="2025", idx=0, i=0, pos_="NUM", ent_type_="DATE", dep_="pobj", children=[], subtree=[])
    token.subtree = [token]
    assert value(N(children=[token]), "2025") is None


def test_collapsed_range_still_copies_both_literal_endpoints():
    text = "from 900 to 700"
    before = N(text="900", idx=5, i=1, ent_type_="", dep_="quantmod", pos_="NUM")
    to = N(text="to", idx=9, i=2, ent_type_="", dep_="quantmod", pos_="ADP", lemma_="to")
    after = N(text="700", idx=12, i=3, ent_type_="CARDINAL", dep_="pobj", pos_="NUM", children=[before, to])
    prep = N(text="from", idx=0, i=0, children=[after])
    result = collapsed_from_to(prep, [prep, before, to, after], text)
    assert [r["text"] for r in result] == ["900", "700"]


def test_collapsed_range_does_not_silently_remove_approximation():
    text = "from about 900 to 700"
    about = N(text="about", idx=5, i=1, ent_type_="")
    before = N(text="900", idx=11, i=2, ent_type_="", dep_="quantmod", pos_="NUM")
    to = N(text="to", idx=15, i=3, ent_type_="", dep_="quantmod", pos_="ADP", lemma_="to")
    after = N(text="700", idx=18, i=4, ent_type_="CARDINAL", dep_="pobj", pos_="NUM", children=[before, to])
    prep = N(text="from", idx=0, i=0, children=[after])
    assert collapsed_from_to(prep, [prep, about, before, to, after], text) is None
