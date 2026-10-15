"""Revalidate deterministic annotations before any reader request."""
from mindset.acl.clients import messages
from mindset.acl.locomo_protocol_diagnostic import short_phrase_messages
from mindset.memory.witness_closure import render_packet, validate_calendar
from mindset.utils.hashing import stable_hash


def transform_plan(planned, retrieval, cases):
    for row in planned:
        saved = retrieval[row['id']]
        events = {e.event_id: e for e in cases[(row['dataset'], row['question_id'])].events}
        notes = saved['reconstruction']['calendar_constraints']
        selected = {s['event_id']: s for s in row['selected']}
        for note in notes:
            if note['event_id'] not in selected or note['quote'] not in selected[note['event_id']]['text']:
                raise ValueError('Calendar expression is absent from selected literal passage')
            validate_calendar(note, events[note['event_id']])
        evidence = render_packet(row['selected'], notes)
        if evidence != saved['annotated_evidence']:
            raise ValueError('Annotated packet cannot be reconstructed')
        row['evidence'] = evidence
        row['calendar_constraints'] = notes
        row['messages'] = (short_phrase_messages(row['question'], evidence, row['question_date'], category=row['category'])
                           if row['dataset'] == 'locomo' else messages(row['question'], evidence, row['question_date']))
        row['prompt_sha256'] = stable_hash(row['messages'])
    return planned
