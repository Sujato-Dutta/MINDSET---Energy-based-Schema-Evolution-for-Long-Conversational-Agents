"""Fixed regression + new controlled contrasts; labels never model inputs."""
from source_relation_fixtures import relation_fixtures


def contrast_fixtures():
    rows = []
    for case in relation_fixtures():
        if case["expected_relationship"] in ("temporal_change", "unknown"):
            rows.append({"id": "regression:" + case["id"], "split": "previously_exposed_regression",
                "reports": case["reports"],
                "expected_kind": "attribute_change" if case["expected_relationship"] == "temporal_change" else "insufficient",
                "expected_direction": case["expected_direction"]})
    examples = [
        ("attribute_change", "In 2020 I led exactly three analysts.", "I now lead exactly eight analysts.", "a_to_b"),
        ("attribute_change", "My sole official job title is now Associate Professor.", "In 2021 my sole official title was Lecturer.", "b_to_a"),
        ("attribute_change", "My bicycle weighed exactly 12 kilograms before the modification.", "After the modification my bicycle weighs exactly 10 kilograms.", "a_to_b"),
        ("attribute_change", "In 2022 my only regular commute was by train.", "I replaced that commute; I now only cycle to work.", "a_to_b"),
        ("attribute_change", "My only current subscription costs 19 euros per month.", "In 2020 my only subscription cost 7 euros per month.", "b_to_a"),
        ("attribute_change", "On Monday I owned exactly two cats in total.", "On Tuesday I owned exactly three cats in total.", "a_to_b"),
        ("event_progression", "Yesterday I found a new climbing route.", "Climbing has made me feel more confident.", "none"),
        ("event_progression", "I met a drummer at a festival.", "I went on tour with that drummer.", "none"),
        ("event_progression", "I started a stretching routine last month.", "Keeping up my stretching routine is hard.", "none"),
        ("event_progression", "I submitted a painting to an exhibition.", "The exhibition experience inspired me.", "none"),
        ("event_progression", "I attended a woodworking class.", "Woodworking has opened new possibilities for me.", "none"),
        ("event_progression", "I registered for a language course.", "I enjoyed attending that course.", "none"),
        ("insufficient", "I lead four engineers in total at work.", "I hope to lead five engineers next year.", "none"),
        ("insufficient", "I own a trumpet.", "I also own a trombone.", "none"),
        ("insufficient", "My only regular shift on weekdays is morning.", "My only regular shift on weekends is evening.", "none"),
        ("insufficient", "My brother manages three engineers.", "My sister manages six engineers.", "none"),
        ("insufficient", "I like apples.", "I also like oranges.", "none"),
        ("insufficient", "I am a teacher at school.", "I am also a volunteer coach after school.", "none"),
        ("insufficient", "It was my usual choice.", "Now the other one is my usual choice.", "none"),
        ("insufficient", "Someone had three of those.", "They now have four of those.", "none"),
        ("insufficient", "That changed last year.", "This changed again yesterday.", "none"),
        ("insufficient", "My garden has three apple trees.", "My orchard has seven pear trees.", "none"),
        ("insufficient", "I lead three engineers in total at work.", "My monthly rent is now 700 euros.", "none"),
        ("insufficient", "In March 2024 I owned a guitar.", "In March 2024 I did not own a guitar.", "none"),
    ]
    for i, (kind, a, b, direction) in enumerate(examples):
        rows.append({"id": f"new_control:{i:02d}", "split": "new_authored_controls",
            "expected_kind": kind, "expected_direction": direction,
            "reports": [{"source_id": "A", "speaker": "user", "observation_time": "2025-01-01", "text": a},
                        {"source_id": "B", "speaker": "user", "observation_time": "2025-06-01", "text": b}]})
    return rows
