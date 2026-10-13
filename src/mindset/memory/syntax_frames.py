"""Frozen dependency-frame projection, with extractive value spans.

Pretrained syntax is NOT comprehensive semantic parsing or NLI. These fixed
projection/admissibility policies remain engineering assumptions, not learned
MINDSET coefficients. No benchmark answers or gold IDs enter this module.
"""
import re

CHANGE_VERBS = frozenset(('change', 'switch', 'update', 'increase', 'decrease',
                          'reduce', 'raise', 'lower', 'adjust', 'set'))
MODALS = frozenset(('would', 'could', 'might', 'may', 'should', 'will', 'can'))


def lemma(token):
    return token.lemma_.casefold() or token.text.casefold()


def noun_terms(head):
    return tuple(dict.fromkeys(lemma(t) for t in sorted(head.subtree, key=lambda t: t.i)
        if t == head or (t.pos_ in ('NOUN', 'PROPN') and t.dep_ in ('compound', 'pobj'))))


def source_span(tokens, text):
    tokens = sorted(tokens, key=lambda t: t.i)
    if not tokens: return None
    start, end = tokens[0].idx, tokens[-1].idx + len(tokens[-1].text)
    return {'start': start, 'end': end, 'text': text[start:end]}


def subjects(root):
    result = [t for t in root.children if t.dep_ == 'nsubj']
    if not result and root.dep_ == 'conj': return subjects(root.head)
    return result


def self_possession(head):
    return any(t.dep_ == 'poss' and t.text.casefold() == 'my' for t in head.children)


def contexts(root, argument_heads=()):
    result = []
    for prep in root.children:
        if prep.dep_ != 'prep': continue
        for obj in prep.children:
            if obj.dep_ != 'pobj' or obj in argument_heads: continue
            if obj.pos_ == 'PROPN' or obj.ent_type_ in ('DATE', 'MONEY', 'CARDINAL', 'QUANTITY'): continue
            # Observable prepositional adjunct; not guessed world-state context.
            words = [lemma(t) for t in sorted(prep.subtree, key=lambda t: t.i)
                     if not t.is_punct and t.pos_ != 'DET']
            if words: result.append(' '.join(words))
    return tuple(sorted(set(result)))


def project_frames(doc, text):
    frames = []
    for sent in doc.sents:
        if sent.text.rstrip().endswith('?') or any(t.is_quote for t in sent): continue
        for root in sent:
            if root.pos_ not in ('VERB', 'AUX') or root.dep_ not in ('ROOT', 'conj'): continue
            children = list(root.children)
            if any(t.dep_ == 'neg' or (t.dep_ == 'aux' and lemma(t) in MODALS)
                   or (t.dep_ == 'mark' and lemma(t) in ('if', 'unless')) for t in children): continue
            # Unknown/hypothetical/past narrative does not become current state.
            if any(t.dep_ in ('advcl', 'mark') and lemma(t) in ('if', 'unless') for t in root.subtree): continue
            changed = lemma(root) in CHANGE_VERBS or any(
                t.dep_ == 'advmod' and lemma(t) in ('now', 'currently') for t in children)
            past = 'Past' in root.morph.get('Tense') or any(
                t.dep_ == 'aux' and t.text.casefold() == 'had' for t in children)
            if past and not changed: continue
            own = subjects(root)
            if len(own) != 1: continue
            subject = own[0]
            direct_self = subject.text.casefold() == 'i'
            possessive = self_possession(subject)
            if not direct_self and not possessive: continue
            objects = [t for t in children if t.dep_ in ('dobj', 'attr', 'acomp', 'oprd')]
            prep_objects = [(p, o) for p in children if p.dep_ == 'prep'
                            for o in p.children if o.dep_ == 'pobj']
            definitions = []
            if possessive and lemma(root) == 'be' and objects:
                definitions = [('property', noun_terms(subject), list(objects[0].subtree), None)]
            elif direct_self and changed:
                owned_objects = [o for o in objects if self_possession(o)]
                to_values = [(p, o) for p, o in prep_objects if lemma(p) == 'to']
                if len(owned_objects) == 1 and len(to_values) == 1:
                    definitions = [('property', noun_terms(owned_objects[0]), list(to_values[0][1].subtree), to_values[0][1])]
            if not definitions and direct_self and not past:
                for obj in objects:
                    quantities = [t for t in obj.subtree if t.dep_ == 'nummod' or t.like_num]
                    if quantities:
                        definitions.append(('quantity', noun_terms(obj), quantities, obj))
                # No generic "I am X" slot: emotions/roles are not one predicate.
                if not definitions and lemma(root) != 'be':
                    for prep, obj in prep_objects:
                        if obj.pos_ == 'PROPN' and lemma(prep) not in ('during', 'on'):
                            definitions.append(('relation', (lemma(root), lemma(prep)), list(obj.subtree), obj))
            for kind, terms, value_tokens, argument in definitions:
                span = source_span(value_tokens, text)
                if not span or not terms: continue
                slot = 'syntax_' + kind + '_' + '_'.join(re.findall(r'[a-z0-9]+', ' '.join(terms)))
                frame = {'slot': slot, 'kind': kind, 'match_terms': list(terms), 'value': span['text'],
                         'value_span': [span['start'], span['end']], 'contexts': list(contexts(root, (argument,) if argument else ())),
                         'explicit_change': bool(changed), 'projection': 'dependency_self_report_v1'}
                if frame not in frames: frames.append(frame)
    # Conflicting within-source assertions are not resolved by their order.
    by_key = {}
    for frame in frames:
        key = (frame['slot'], tuple(frame['contexts']))
        by_key.setdefault(key, []).append(frame)
    return [rows[0] for rows in by_key.values() if len({r['value'].casefold() for r in rows}) == 1]


def project_query(doc):
    tokens = list(doc)
    direct_self = any(t.dep_ == 'nsubj' and t.text.casefold() == 'i' for t in tokens)
    # Nested ownership (my friend's address) is not ownership by the user.
    possessive_self = any(t.dep_ == 'poss' and t.text.casefold() == 'my'
                          and t.head.dep_ != 'poss' for t in tokens)
    present = any(t.pos_ in ('VERB', 'AUX') and 'Pres' in t.morph.get('Tense') for t in tokens)
    terms = list(dict.fromkeys(lemma(t) for t in tokens if t.pos_ in ('VERB', 'NOUN', 'PROPN', 'ADP')))
    query_contexts = sorted({c for root in tokens if root.dep_ == 'ROOT' for c in contexts(root)})
    return {'terms': terms, 'direct_self': direct_self, 'possessive_self': possessive_self,
            'present_tense': present, 'contexts': query_contexts, 'projection': 'dependency_query_v1'}
