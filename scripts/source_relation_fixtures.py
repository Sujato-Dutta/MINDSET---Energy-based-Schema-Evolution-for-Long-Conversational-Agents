"""Frozen synthetic transition-type cases; not a benchmark or parameter fitting.

Two pairs per type in development and two lexically different pairs in
confirmation. Observation order includes retrospective descriptions. Expected
types/directions are post-inference diagnostics and never model inputs.
"""


def relation_fixtures():
    groups = {
        'equivalent': [
            ('I own a violin.', 'I possess a violin.', 'none'),
            ('My only current pet is a hamster.', 'I currently have only one pet, a hamster.', 'none'),
            ('I enjoy canoeing.', 'I like canoeing.', 'none'),
            ('My sole current home is in Bergen.', 'Bergen is where my only current home is.', 'none')],
        'temporal_change': [
            ('In 2019 my only primary residence was in Dublin.', 'My only primary residence is now in Kyoto.', 'a_to_b'),
            ('My only current employer is Orion Labs.', 'In 2018 my only employer was Cedar Works.', 'b_to_a'),
            ('In 2017 my sole primary home was in Lima.', 'My sole primary home is now in Tallinn.', 'a_to_b'),
            ('I currently own exactly six goats.', 'In 2016 I owned exactly two goats.', 'b_to_a')],
        'context_branch': [
            ('My only regular workout time on weekdays is morning.', 'My only regular workout time on weekends is evening.', 'none'),
            ('My monthly spending limit for groceries is 150 dollars.', 'My monthly spending limit for hobbies is 80 dollars.', 'none'),
            ('My default meal at work is soup.', 'My default meal at home is rice.', 'none'),
            ('My sole regular transport on school days is the bus.', 'My sole regular transport on holidays is a bicycle.', 'none')],
        'accumulation': [
            ('I like pears.', 'I also like plums.', 'none'),
            ('I own a microscope.', 'I also own a camera.', 'none'),
            ('I enjoy pottery.', 'I also enjoy kayaking.', 'none'),
            ('I own a banjo.', 'I also own a mandolin.', 'none')],
        'modal_coexistence': [
            ('I plan to adopt a ferret next year.', 'I currently do not own a ferret.', 'none'),
            ('I hope to become a surgeon.', 'I am currently a medical student.', 'none'),
            ('I might buy a sailboat next summer.', 'I do not currently own a sailboat.', 'none'),
            ('I want to move to Zurich next year.', 'My current primary home is in Leeds.', 'none')],
        'independent': [
            ('My brother is a pilot.', 'My sister is a journalist.', 'none'),
            ('My height is 180 centimeters.', 'My annual tuition is 7000 dollars.', 'none'),
            ('My daughter plays the flute.', 'My son plays the cello.', 'none'),
            ('My savings are 4000 dollars.', 'My weight is 65 kilograms.', 'none')],
        'same_scope_conflict': [
            ('In March 2024 I owned a clarinet.', 'In March 2024 I did not own a clarinet.', 'none'),
            ('On 1 May 2024 I had exactly seven hens.', 'On 1 May 2024 I had exactly nine hens.', 'none'),
            ('In February 2023 I enjoyed rowing.', 'In February 2023 I did not enjoy rowing.', 'none'),
            ('On 2 June 2023 my total number of pets was exactly one.', 'On 2 June 2023 my total number of pets was exactly four.', 'none')],
        'unknown': [
            ('I prefer that option.', 'I prefer the other one now.', 'none'),
            ('Someone told me it changed.', 'It might still be the same.', 'none'),
            ('They had it before.', 'They have another one now.', 'none'),
            ('That is my favorite.', 'This is my favorite now.', 'none')],
    }
    rows = []
    for relation, pairs in groups.items():
        for i, (left, right, direction) in enumerate(pairs):
            split = 'development' if i < 2 else 'lexical_confirmation'
            rows.append({'id': f'{split}:{relation}:{i % 2}', 'split': split,
                'expected_relationship': relation, 'expected_direction': direction,
                'reports': [{'source_id': 'A', 'speaker': 'user', 'observation_time': '2025-01-01', 'text': left},
                            {'source_id': 'B', 'speaker': 'user', 'observation_time': '2025-06-01', 'text': right}]})
    return rows
