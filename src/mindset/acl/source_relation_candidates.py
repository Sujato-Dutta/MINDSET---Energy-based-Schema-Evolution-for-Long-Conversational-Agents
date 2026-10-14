"""Source-only natural relationship candidates, no outcome-conditioned admission."""
from mindset.methods.joint import utc_time
from mindset.memory.source_relation_contract import relation_messages
from mindset.utils.hashing import stable_hash


def measured_pairs(pairs, spans, events, *, question_date='', anchor_limit=4):
    """One closest neighbor per distinct seed span, in frozen neighbor order.

    Candidate choice is made before any relationship labels or QA outcomes.
    Complete literal sentence spans, speaker and observation time are retained.
    This bounded natural instrument budget is not a final-method definition.
    """
    if anchor_limit < 1:
        raise ValueError('Positive anchor budget required')
    by_id = {e.event_id: e for e in events}
    if len(by_id) != len(events):
        raise ValueError('Duplicate source episodes')
    deadline = utc_time(question_date); used = set(); records = []
    for a, b in pairs:
        x, y = spans[a], spans[b]; anchor = (x.eid, x.start, x.end)
        if anchor in used:
            continue
        if len(used) == anchor_limit:
            break
        sides = []; reports = []
        for sid, s in (('A', x), ('B', y)):
            event = by_id[s.eid]
            if not (0 <= s.start < s.end <= len(event.text)) or event.text[s.start:s.end] != s.text:
                raise ValueError('Report is not its literal source span')
            if event.speaker.casefold() != s.speaker:
                raise ValueError('Report author mismatch')
            timestamp = utc_time(event.timestamp)
            if deadline and (not timestamp or timestamp > deadline):
                raise ValueError('Undated/future report admitted to dated question')
            reports.append({'source_id': sid, 'speaker': event.speaker,
                'observation_time': event.timestamp, 'text': s.text})
            sides.append({'source_id': sid, 'episode_id': s.eid,
                'char_start': s.start, 'char_end': s.end})
        if x.eid == y.eid or x.speaker != y.speaker:
            raise ValueError('Cross-author or same-episode neighbor admitted')
        relation_messages(reports)  # structural contract, not semantic validation
        identity = {'reports': reports, 'source_spans': sides}
        records.append({**identity, 'pair_key': stable_hash(identity)})
        used.add(anchor)
    return records
