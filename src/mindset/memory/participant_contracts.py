"""Observable speaker attribution for query syntax; no inferred identities."""


def canonical_owner(doc,text,speakers):
    owners={s.casefold():s for s in speakers}
    first=any(t.text.casefold() in ('i','my','me','mine') for t in doc)
    names=set(owners)-{'user','assistant','system','tool'}
    candidates=[]
    coordinated_first=any(t.text.casefold() in ('i','my','me','mine')
                         and (t.dep_=='conj' or any(c.dep_=='conj' for c in t.children)) for t in doc)
    for token in doc:
        if token.dep_ not in ('nsubj','poss'): continue
        if any(c.dep_=='conj' for c in token.children): continue
        ts=sorted([token]+[c for c in token.children if c.dep_=='compound'],key=lambda t:t.i)
        label=' '.join(t.text for t in ts).casefold()
        if label not in names: continue
        if token.dep_=='poss':
            # Only a directly possessed property of a copular clause. A named
            # participant's friend's statement does not become their own.
            parent=token.head
            if parent.dep_ not in ('nsubj','attr') or parent.head.lemma_.casefold()!='be': continue
            cases=[c for c in token.children if c.dep_=='case']
            if not cases: continue
            end=max(c.idx+len(c.text) for c in cases); replacement='my'
        else:
            end=ts[-1].idx+len(ts[-1].text); replacement='I'
        candidates.append((label,ts[0].idx,end,replacement))
    if first:
        if candidates or coordinated_first or 'user' not in owners:
            return {'owner':None,'canonical_text':text,'reason':'ambiguous_first_person_owner','rewrites':[]}
        return {'owner':'user','canonical_text':text,'reason':'literal_first_person_user','rewrites':[]}
    if len(candidates)!=1:
        return {'owner':None,'canonical_text':text,'reason':'no_unique_direct_speaker_argument','rewrites':[]}
    owner,a,b,replacement=candidates[0]
    return {'owner':owner,'canonical_text':text[:a]+replacement+text[b:],
            'reason':'exact_metadata_speaker_in_direct_argument_role',
            'rewrites':[{'source_span':[a,b],'source_text':text[a:b],'replacement':replacement}]}
