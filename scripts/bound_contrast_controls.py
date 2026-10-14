"""New lexical controls for fresh outputs; not domain OOD confirmation."""


def fresh_controls():
    examples = [
        ("attribute_change", "My only primary home is now in Oslo.", "In 2012 my only primary home was in Kigali.", "b_to_a"),
        ("attribute_change", "Last year I managed exactly seven technicians in total.", "I now manage exactly twelve technicians in total.", "a_to_b"),
        ("attribute_change", "My account's total current storage limit is exactly 50 gigabytes.", "In 2020 my account's total storage limit was exactly 32 gigabytes.", "b_to_a"),
        ("attribute_change", "In 2021 my only active project code was Zephyr.", "I replaced that project code; my only active project code is now Nimbus.", "a_to_b"),
        ("attribute_change", "My only regular commute was by ferry.", "I replaced that commute; I now only take the train to work.", "a_to_b"),
        ("attribute_change", "My monthly travel spending limit was exactly 300 euros.", "I changed my monthly travel spending limit to exactly 450 euros.", "a_to_b"),
        ("event_progression", "I discovered a new birdwatching spot.", "Birdwatching has helped me appreciate nature.", "none"),
        ("insufficient", "Someone tracked four of those.", "They now track six of those.", "none"),
        ("insufficient", "It was my preferred choice.", "The other one is my preferred choice now.", "none"),
        ("insufficient", "Our group currently has three guides.", "Your group currently has five guides.", "none"),
        ("insufficient", "On 8 April 2024 I owned exactly two saxophones.", "On 8 April 2024 I owned exactly five saxophones.", "none"),
        ("insufficient", "My only regular practice time on weekdays is afternoon.", "My only regular practice time on weekends is morning.", "none"),
    ]
    return [{"id": f"fresh_control:{i:02d}", "split": "fresh_lexical_same_family_controls",
        "expected_kind": kind, "expected_direction": direction,
        "reports": [{"source_id": "A", "speaker": "user", "observation_time": "2025-01-01", "text": a},
                    {"source_id": "B", "speaker": "user", "observation_time": "2025-06-01", "text": b}]}
        for i, (kind, a, b, direction) in enumerate(examples)]
