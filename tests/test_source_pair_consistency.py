from copy import deepcopy
import pytest
from mindset.evaluation.source_pair_consistency import consistency


def fixture():
    reports = [{"source_id": s, "speaker": "user", "observation_time": date, "text": text}
        for s,date,text in [("A","2025-01-01","I lead four engineers."), ("B","2025-06-01","I lead five engineers.")]]
    spans = [{"source_id": s, "episode_id": eid, "char_start": 0, "char_end": len(reports[i]["text"])}
        for i,(s,eid) in enumerate([("A","early"),("B","late")])]
    parents = {"forward": {"reports": reports, "source_spans": spans}}
    reversed_reports = [{**reports[1], "source_id": "A"}, {**reports[0], "source_id": "B"}]
    reversed_spans = [{**spans[1], "source_id": "A"}, {**spans[0], "source_id": "B"}]
    parents["reverse"] = {"reports": reversed_reports, "source_spans": reversed_spans}
    def row(key, direction):
        return {"id": key, "parent_pair_key": key, "reports": parents[key]["reports"], "error": None,
            "prediction": {"kind": "attribute_change", "variable": "team size", "direction": direction},
            "candidate": {"update_proposal": True}}
    return parents, [row("forward", "a_to_b"), row("reverse", "b_to_a")]


def test_reversed_source_labels_do_not_reverse_the_actual_update():
    parents, rows = fixture(); a = consistency(rows, parents)
    assert a["natural_rows"] == 2 and a["unordered_source_span_groups"] == 1
    assert a["mirrored_groups"] == 1 and a["mirrored_relation_direction_disagreement_groups"] == 0
    assert a["update_admitted_source_groups"] == 1


def test_same_direction_label_after_source_swap_is_detected():
    parents, rows = fixture(); rows[1]["prediction"]["direction"] = "a_to_b"
    assert consistency(rows, parents)["mirrored_relation_direction_disagreement_groups"] == 1


def test_structural_failure_is_not_silently_a_consistent_negative():
    parents, rows = fixture(); rows[1].update(error="rewritten quote", candidate=None)
    a = consistency(rows, parents)
    assert a["mirrored_structural_failure_groups"] == 1
    assert a["groups"][0]["members"][1]["interpretation"]["valid"] is False


def test_guard_admission_difference_is_separate_from_relationship_label():
    parents, rows = fixture(); rows[1]["candidate"]["update_proposal"] = False
    a = consistency(rows, parents)
    assert a["mirrored_update_admission_disagreement_groups"] == 1
    assert a["mirrored_relation_direction_disagreement_groups"] == 0


def test_different_source_episode_is_not_collapsed_as_an_independent_repeat():
    parents, rows = fixture(); parents = deepcopy(parents)
    parents["reverse"]["source_spans"][0]["episode_id"] = "distinct-episode"
    a = consistency(rows, parents)
    assert a["unordered_source_span_groups"] == 2 and a["mirrored_groups"] == 0


def test_changed_parent_source_is_rejected():
    parents, rows = fixture(); rows[1] = deepcopy(rows[1]); rows[1]["reports"][0]["text"] += " Changed."
    with pytest.raises(ValueError): consistency(rows, parents)
