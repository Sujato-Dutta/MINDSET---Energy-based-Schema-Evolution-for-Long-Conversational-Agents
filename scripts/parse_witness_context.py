"""Pinned isolated spaCy subprocess; source-only inputs, no QA references."""
import json
import sys
import hashlib
import spacy


def run():
    import os
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('DGX Slurm only')
    rows = json.load(sys.stdin)
    if spacy.__version__ != '3.8.7':
        raise ValueError('Pinned parser changed')
    nlp = spacy.load('en_core_web_sm')
    if nlp.meta['version'] != '3.8.0':
        raise ValueError('Pinned syntax model changed')
    outputs = {}
    opaque = {'it', 'they', 'them', 'he', 'him', 'she', 'her', 'this', 'that', 'these', 'those'}
    replies = {'thanks', 'thank', 'yeah', 'yes', 'yep', 'wow'}
    for row, doc in zip(rows, nlp.pipe([r['text'] for r in rows], batch_size=64), strict=True):
        sentences = list(doc.sents)
        first = sentences[0] if sentences else []
        markers = [t for t in first if t.text.casefold() in opaque and
            t.dep_ in {'nsubj', 'nsubjpass', 'dobj', 'pobj', 'attr', 'det'}]
        initial = next((t.lemma_.casefold() for t in doc if not t.is_punct and not t.is_space), '')
        reply = initial in replies
        outputs[row['event_id']] = dict(requires_previous_turn=bool(markers or reply),
            source_sha256=hashlib.sha256(row['text'].encode()).hexdigest(),
            markers=[dict(text=t.text, start=t.idx, end=t.idx + len(t.text), dependency=t.dep_) for t in markers],
            reply_marker=reply, semantics_verified=False)
    json.dump(dict(parser_version=spacy.__version__, model_version=nlp.meta['version'], sources=outputs), sys.stdout)


if __name__ == '__main__':
    run()
