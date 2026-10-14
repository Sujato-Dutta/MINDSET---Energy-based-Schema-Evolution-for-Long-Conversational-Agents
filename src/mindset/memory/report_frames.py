"""Extractive numeric reports and grammatical query operators, not world truth.

Past reports are retained as reports; they do not silently become present state.
No answers, benchmark labels, entity-name lists or generated evidence enter here.
"""
import re
from decimal import Decimal, InvalidOperation

SMALL = dict(zip(('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split(), range(20)))
TENS = dict(zip(('twenty thirty forty fifty sixty seventy eighty ninety').split(), range(20, 100, 10)))


def normalize_number(text):
    value = text.strip().casefold().replace(',', '')
    try:
        number = Decimal(value)
        if number.is_finite() and number >= 0: return format(number.normalize(), 'f')
    except InvalidOperation: pass
    total = part = 0
    words = re.findall(r'[a-z]+', value)
    if not words: return None
    for word in words:
        if word == 'and': continue
        if word in SMALL: part += SMALL[word]
        elif word in TENS: part += TENS[word]
        elif word == 'hundred': part = max(1, part) * 100
        elif word in ('thousand', 'million'):
            total += max(1, part) * (1000 if word == 'thousand' else 1000000); part = 0
        else: return None
    return str(total + part)


def lemma(t): return t.lemma_.casefold() or t.text.casefold()


def subjects(root):
    own = [t for t in root.children if t.dep_ in ('nsubj', 'nsubjpass')]
    if not own and root.dep_ in ('conj', 'xcomp'): return subjects(root.head)
    return own


def predicate(head):
    visited = set(); current = head
    while current.i not in visited:
        visited.add(current.i)
        if current.pos_ in ('VERB', 'AUX'): return current
        if current.head == current: break
        current = current.head
    return None


def unit_words(head):
    nouns = [head] + [t for t in head.children if t.dep_ == 'compound' and t.pos_ in ('NOUN', 'PROPN')]
    for prep in head.children:
        if prep.dep_ == 'prep' and lemma(prep) == 'of':
            nouns.extend(t for t in prep.children if t.dep_ == 'pobj' and t.pos_ in ('NOUN', 'PROPN'))
    return tuple(dict.fromkeys(lemma(t) for t in sorted(nouns, key=lambda t: t.i)))


def unit_phrase(head):
    before = [lemma(t) for t in sorted(head.children, key=lambda t: t.i) if t.dep_ == 'compound' and t.pos_ in ('NOUN', 'PROPN')]
    result = ' '.join(before + [lemma(head)])
    for prep in head.children:
        if prep.dep_ == 'prep' and lemma(prep) == 'of':
            objects = [t for t in prep.children if t.dep_ == 'pobj' and t.pos_ in ('NOUN', 'PROPN')]
            if len(objects) == 1: result += ' of ' + lemma(objects[0])
    return result


def project_reports(doc, text, speaker):
    result = []
    # Source code/instructions and quoted text are not self-report assertions.
    if '```' in text or text.lstrip().startswith(('Web search results:', 'URL:')): return []
    for sent in doc.sents:
        if sent.text.rstrip().endswith('?') or any(t.is_quote for t in sent): continue
        for head in sent:
            if head.pos_ not in ('NOUN', 'PROPN') or head.ent_type_ in ('DATE', 'TIME'): continue
            numbers = [t for t in head.children if t.dep_ == 'nummod']
            if len(numbers) != 1: continue
            value_tokens = sorted(numbers[0].subtree, key=lambda t: t.i)
            if any(t.ent_type_ in ('DATE', 'TIME') or t.dep_ == 'conj' for t in value_tokens): continue
            start = value_tokens[0].idx; end = value_tokens[-1].idx + len(value_tokens[-1].text)
            value = text[start:end]; normalized = normalize_number(value)
            if normalized is None: continue
            root = predicate(head)
            if root is None: continue
            ancestors = [root] + list(root.ancestors)
            if any(lemma(t) in ('say', 'tell', 'ask', 'quote', 'hear', 'read') for t in ancestors): continue
            if any(t.dep_ == 'neg' or (t.dep_ in ('aux', 'auxpass') and lemma(t) in ('would', 'could', 'might', 'may', 'should', 'will', 'can'))
                   for a in ancestors for t in a.children): continue
            if any(t.dep_ == 'mark' and lemma(t) in ('if', 'unless') for a in ancestors for t in a.children): continue
            # Bounds/approximations remain ordinary evidence, not exact counts.
            near = [t.text.casefold() for t in sent if start - 24 <= t.idx < start]
            if any(w in near for w in ('about', 'around', 'roughly', 'approximately', 'least', 'most', 'over', 'under', 'nearly')): continue
            own = subjects(root)
            if len(own) != 1: continue
            subject = own[0]; holder = ()
            if subject.text.casefold() == 'i': owner = speaker.casefold(); ownership = 'literal_first_person_speaker'
            elif subject.pos_ == 'PROPN':
                named = [t for t in sorted(subject.subtree, key=lambda t: t.i) if t.pos_ == 'PROPN']
                owner = ' '.join(t.text.casefold() for t in named); ownership = 'explicit_named_subject'
            elif any(t.dep_ == 'poss' and t.text.casefold() == 'my' for t in subject.children):
                owner = speaker.casefold(); holder = (lemma(subject),); ownership = 'explicit_possessed_subject'
            else: continue
            if not owner: continue
            units = unit_words(head); phrase = unit_phrase(head)
            ctx = []
            for prep in root.children:
                if prep.dep_ == 'prep' and lemma(prep) in ('at', 'during'):
                    objects = [t for t in prep.children if t.dep_ == 'pobj']
                    if len(objects) == 1 and objects[0].ent_type_ not in ('DATE', 'TIME'):
                        ctx.append(' '.join(lemma(t) for t in prep.subtree if not t.is_punct and t.pos_ != 'DET'))
            probe = 'I ' + lemma(root) + ' one ' + phrase + '.'
            frame = {'owner': owner, 'ownership': ownership, 'holder_terms': list(holder),
                'unit_terms': list(units), 'unit_head': lemma(head), 'predicate': lemma(root), 'probe': probe,
                'value': value, 'normalized_value': normalized, 'value_span': [start, end],
                'contexts': sorted(set(ctx)), 'tense': 'past' if 'Past' in root.morph.get('Tense') else 'nonpast',
                'kind': 'observed_quantity_report_not_certified_current_state', 'projection': 'quantity_report_v1'}
            if frame not in result: result.append(frame)
    return result


def project_report_query(doc, text):
    terms = tuple(dict.fromkeys(lemma(t) for t in doc if t.pos_ in ('VERB', 'NOUN', 'PROPN', 'ADP')))
    self_ref = any(t.text.casefold() in ('i', 'my', 'me', 'mine') for t in doc)
    quantitative = bool(re.search(r'\bhow\s+(?:many|much)|\b(?:number|quantity|amount|limit|increase|decrease|total)\b', text, re.I))
    compare = bool(re.search(r'\b(?:compare|compared|versus|before and after|initial|initially|originally|first|start|started)\b', text, re.I)
                   and re.search(r'\b(?:now|current|currently|latest|after|compared|versus)\b', text, re.I))
    change = bool(re.search(r'\b(?:increase|decrease|change|changed|adjust|adjusted|raise|lower|reduce)\b', text, re.I))
    if re.search(r'\b(?:total|different|how many times)\b', text, re.I) and not compare and not change: mode = 'aggregate_event'
    elif quantitative and compare: mode = 'report_compare'
    elif quantitative and change: mode = 'report_change'
    else: mode = 'ordinary'
    probes = []
    for head in doc:
        if head.pos_ not in ('NOUN', 'PROPN') or head.ent_type_ in ('DATE', 'TIME'): continue
        root = predicate(head)
        if root is None or lemma(root) == 'be': continue
        probes.append({'unit_head': lemma(head), 'unit_terms': list(unit_words(head)),
                       'predicate': lemma(root), 'probe': 'I ' + lemma(root) + ' one ' + unit_phrase(head) + '.'})
    return {'terms': list(terms), 'self_reference': self_ref, 'quantitative': quantitative, 'mode': mode,
            'probes': probes, 'projection': 'quantity_query_operator_v1'}
