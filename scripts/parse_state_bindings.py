"""Pinned, source-only spaCy proposal worker; stdin/stdout JSON subprocess.

This admits candidate semantic bindings, NOT certified world facts. It must run
in the isolated syntax environment. Never receives benchmark reference answers.
"""
import hashlib
import json
import os
import sys


def proposals(nlp, rows):
    result = {}
    for row, doc in zip(rows, nlp.pipe([r['text'] for r in rows], batch_size=64)):
        chunks = {t.i: c for c in doc.noun_chunks for t in c}
        bindings = []
        for verb in doc:
            if verb.pos_ not in {'VERB', 'AUX'} or verb.dep_ not in {'ROOT', 'conj', 'ccomp', 'xcomp'}:
                continue
            subjects = [c for c in verb.children if c.dep_ in {'nsubj', 'nsubjpass'}]
            if not subjects and verb.dep_ == 'conj':
                subjects = [c for c in verb.head.children if c.dep_ in {'nsubj', 'nsubjpass'}]
            if not subjects:
                continue
            subject = subjects[0]
            first_person = subject.lower_ == 'i'
            unresolved = not first_person and subject.pos_ == 'PRON'
            holder = row['speaker'].casefold() if first_person else chunks.get(subject.i, doc[subject.i:subject.i+1]).text.casefold()
            children = list(verb.children)
            negated = any(t.dep_ == 'neg' for t in children)
            ancestors = list(verb.ancestors)
            modal = (any(t.lemma_.casefold() in {'may', 'might', 'could', 'would', 'should', 'will'} for t in children)
                     or any(t.lemma_.casefold() in {'want', 'hope', 'plan', 'wish', 'imagine'} for t in ancestors))
            values = [t for t in children if t.dep_ in {'dobj', 'obj', 'attr', 'acomp', 'oprd'}]
            context = ''; prep_values = []
            # A PP can attach to the verb OR its object ('like jazz at church').
            # Admit the source context without relying on that attachment choice.
            # Do not cross another verbal clause while walking the subtree.
            preps = [t for t in verb.subtree if t.dep_ == 'prep' and
                     next((a for a in t.ancestors if a.pos_ in {'VERB', 'AUX'}), None) == verb]
            for prep in preps:
                for obj in (t for t in prep.children if t.dep_ == 'pobj'):
                    span = chunks.get(obj.i, doc[obj.i:obj.i+1])
                    if prep.lower_ in {'at', 'during'} and not any(t.ent_type_ in {'DATE', 'TIME'} for t in span):
                        context = span.text.casefold()
                    elif prep.lower_ in {'in', 'to', 'for', 'with'}:
                        prep_values.append(obj)
            values = values or prep_values
            for value in values:
                span = chunks.get(value.i, doc[value.i:value.i+1])
                if not span.text.strip() or value.ent_type_ in {'DATE', 'TIME'}:
                    continue
                lemma = verb.lemma_.casefold()
                # All are linguistic proposal features, not hand-set energy
                # coefficients. Event/state errors must be evaluated downstream.
                eventive = lemma not in {'be', 'have', 'own', 'live', 'work', 'prefer', 'like', 'manage', 'lead'}
                bindings.append(dict(event_id=row['event_id'], holder=holder,
                    variable=lemma, value=span.text, value_start=span.start_char,
                    value_end=span.end_char, context=context, order=row['order'],
                    negated=negated, modal=modal, past='Past' in verb.morph.get('Tense'),
                    eventive=eventive,
                    explicit_change=any(t.lower_ in {'now', 'instead', 'currently'} for t in children),
                    unresolved_holder=unresolved))
        result[row['event_id']] = dict(source_sha256=hashlib.sha256(row['text'].encode()).hexdigest(),
            bindings=bindings, semantic_correctness_verified=False)
    return result


if __name__ == '__main__':
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('DGX Slurm only')
    import spacy
    nlp = spacy.load('en_core_web_sm')
    if spacy.__version__ != '3.8.7' or nlp.meta['version'] != '3.8.0':
        raise ValueError('Unpinned syntax worker')
    rows = json.load(sys.stdin)
    json.dump(dict(parser='spacy3.8.7/en_core_web_sm3.8.0',
        parser_sha256=hashlib.sha256(nlp.to_bytes()).hexdigest(),
        sources=proposals(nlp, rows)), sys.stdout)
