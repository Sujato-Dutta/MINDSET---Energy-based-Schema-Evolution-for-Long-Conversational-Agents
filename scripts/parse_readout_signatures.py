"""Isolated pinned spaCy source-signature worker; no questions/answer labels."""
import hashlib
import json
import os
import sys
from mindset.memory.report_frames import project_reports, predicate, subjects, lemma


def parse(nlp, rows):
    result = {}
    for row, doc in zip(rows, nlp.pipe([r['body'] for r in rows], batch_size=64)):
        signatures = {}; reports = project_reports(doc, row['body'], row['speaker'])
        for u in row['units']:
            key = None; reason = 'unresolved_source_scope'
            if u['kind'] == 'number':
                matches = [r for r in reports if r['value_span'] == [u['start'], u['end']]]
                if len(matches) == 1:
                    r = matches[0]
                    key = ['number', r['owner'], r['holder_terms'], r['predicate'], r['unit_terms'], r['contexts']]
                    reason = 'literal_numeric_report_signature'
            else:
                tokens = [t for t in doc if u['start'] <= t.idx < u['end']]
                root = predicate(tokens[-1]) if tokens else None
                if root is not None:
                    own = subjects(root); sent = root.sent
                    risky = (sent.text.rstrip().endswith('?') or any(t.is_quote for t in sent)
                        or any(t.dep_ == 'neg' or (t.dep_ in {'aux', 'auxpass'} and lemma(t) in
                            {'would', 'could', 'might', 'may', 'should', 'will', 'can'})
                            for a in [root, *root.ancestors] for t in a.children)
                        or any(lemma(a) in {'say', 'tell', 'ask', 'quote', 'hear', 'read', 'hope', 'want', 'plan'}
                               for a in [root, *root.ancestors]))
                    owner = None
                    if len(own) == 1 and own[0].lower_ == 'i':
                        owner = row['speaker'].casefold()
                    elif len(own) == 1 and own[0].pos_ == 'PROPN':
                        owner = ' '.join(t.lower_ for t in own[0].subtree if t.pos_ == 'PROPN')
                    objects = [t for t in root.children if t.dep_ in {'obj', 'dobj', 'attr', 'oprd'}]
                    # An unidentified 'it/them' is not made comparable to another
                    # report merely because the main verb is identical.
                    concrete = objects and all(t.pos_ in {'NOUN', 'PROPN'} for t in objects)
                    if owner and not risky and concrete:
                        nouns = sorted({lemma(t) for obj in objects for t in obj.subtree
                                        if t.pos_ in {'NOUN', 'PROPN'} and t.ent_type_ not in {'DATE', 'TIME'}})
                        key = ['date', owner, lemma(root), nouns]
                        reason = 'literal_calendar_event_signature'
            signatures[u['id']] = dict(signature=key, reason=reason, semantic_equivalence_verified=False)
        result[row['id']] = dict(source_sha256=hashlib.sha256(row['body'].encode()).hexdigest(), signatures=signatures)
    return result


if __name__ == '__main__':
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('DGX Slurm only')
    import spacy
    nlp = spacy.load('en_core_web_sm')
    if spacy.__version__ != '3.8.7' or nlp.meta['version'] != '3.8.0':
        raise ValueError('Unpinned parser')
    rows = json.load(sys.stdin)
    json.dump(dict(parser_sha256=hashlib.sha256(nlp.to_bytes()).hexdigest(), sources=parse(nlp, rows)), sys.stdout)
