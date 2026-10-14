"""Revalidate prepared sources without modifying the sealed input selector."""
from mindset.acl.source_relation_candidates import measured_pairs
from mindset.methods.semantic_witness import Span
from mindset.memory.source_relation_contract import relation_messages


def verify_prepared_pair(pair, events, *, question_date=''):
    if not isinstance(pair, dict) or set(pair) != {'reports', 'source_spans', 'pair_key'}:
        raise ValueError('Invalid prepared pair fields')
    reports, sides = pair['reports'], pair['source_spans']
    relation_messages(reports)
    if not isinstance(sides, list) or len(sides) != 2:
        raise ValueError('Two source spans required')
    spans = []
    for i, sid in enumerate(('A', 'B')):
        r, s = reports[i], sides[i]
        if r['source_id'] != sid or not isinstance(s, dict) or set(s) != {
            'source_id', 'episode_id', 'char_start', 'char_end'} or s['source_id'] != sid:
            raise ValueError('Source ordering or span identity changed')
        if not all(isinstance(s[k], int) and not isinstance(s[k], bool) for k in ('char_start', 'char_end')):
            raise ValueError('Integer source offsets required')
        spans.append(Span(s['episode_id'], s['char_start'], s['char_end'], r['text'], r['speaker'].casefold()))
    rebuilt = measured_pairs([(0, 1)], spans, events, question_date=question_date)
    if rebuilt != [pair]:
        raise ValueError('Prepared source identity or observation metadata changed')
    return True
