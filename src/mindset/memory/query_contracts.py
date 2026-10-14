"""Typed observable query/report contracts; no answer-derived admission.

Pretrained NLI cannot replace argument-role/type grounding. Exact head lemmas
and explicit names are conservative admissibility constraints, not semantic
completeness. Unresolved synonyms remain ordinary retrieval rather than claims.
"""
import re

PERIOD=r'(?:day|week|month|year|hour)s?'
FREQUENCY=re.compile(r'\b(?:(?:once|twice|\d+\s+times?)\s+(?:a|per|each|every)\s+'+PERIOD+
    r'|(?:once|twice)\s+(?:daily|weekly|monthly|yearly)|(?:daily|weekly|monthly|yearly))\b',re.I)


def lemma(t): return t.lemma_.casefold() or t.text.casefold()


def head_key(doc):
    roots=[t for t in doc if t.head==t and not t.is_punct]
    return lemma(roots[0]) if len(roots)==1 else None


def explicit_names(doc):
    return tuple(sorted({t.text.casefold().strip('.') for t in doc if t.pos_=='PROPN' and t.text.strip('.')}))


def head_target(head):
    # "type of camera lens": the queried item is lens, not the word type.
    if lemma(head) in ('type','kind','sort','number','amount'):
        inner=[o for p in head.children if p.dep_=='prep' and lemma(p)=='of'
               for o in p.children if o.dep_=='pobj']
        if len(inner)==1: return inner[0]
    return head


def query_contract(doc,text):
    frequency=bool(re.search(r'\bhow often\b',text,re.I))
    count=bool(re.search(r'\bhow many\b',text,re.I))
    exists=bool(re.search(r'\b(?:do|does)\s+I\s+(?:have|own|possess)\b',text,re.I))
    kind=bool(re.search(r'\bwhat\s+(?:type|kind|sort)\b',text,re.I))
    location=any(t.text.casefold()=='where' for t in doc)
    direct_roots=[t for t in doc if t.pos_ in ('VERB','AUX') and lemma(t)!='be'
                  and any(c.dep_=='nsubj' and c.text.casefold()=='i' for c in t.children)]
    targets=[]
    for root in direct_roots:
        for obj in root.children:
            if obj.dep_ in ('dobj','attr'): targets.append(head_target(obj))
    properties=[t for t in doc if t.dep_!='poss' and any(c.dep_=='poss' and c.text.casefold()=='my' for c in t.children)]
    if count:
        targets=[t for t in doc if t.pos_=='NOUN' and any(c.text.casefold()=='many' for c in t.children)] or targets
    if frequency: answer_kind='frequency'
    elif count: answer_kind='quantity'
    elif exists: answer_kind='existence'
    elif kind: answer_kind='object_type'
    elif location: answer_kind='location'
    elif properties: answer_kind='property'; targets=properties
    else: answer_kind='unsupported'
    names=explicit_names(doc)
    # Explicit named argument can be an appositive of a description, e.g.
    # "my therapist, Dr Smith". It stays separate from head-type identity.
    target_names=tuple(sorted({n for t in targets for n in explicit_names(t.subtree)}))
    return {'answer_kind':answer_kind,'argument_heads':sorted({lemma(t) for t in targets}),
            'named_scope':list(names),'target_names':list(target_names),
            'predicates':sorted({lemma(r) for r in direct_roots}),
            'policy':'explicit argument head/role and answer kind; unresolved aliases abstain'}


def report_contract(frame,value_doc,text):
    family=frame['family']
    if family=='property': core=frame['unit_head'].split()[-1]
    elif family=='quantity': core=frame['unit_head']
    elif family=='object': core=head_key(value_doc)
    else: core=None
    names=list(frame.get('argument_names',())) if family=='frequency' else list(explicit_names(value_doc))
    key='named:'+' '.join(names) if family=='frequency' and names else core or frame['unit_head']
    probe=frame['probe']
    if family=='object' and core: probe='I '+frame['predicate']+' a '+core+'.'
    if family=='frequency': probe='I '+frame['predicate']+' '+' '.join(names)+' regularly.'
    return {**frame,'unit_terms':[family,key],'probe':probe,
        'contract':{'family':family,'argument_head':core,'argument_names':names},
        'projection':'typed_report_contract_v1'}


def frequency_frames(doc,text,speaker):
    frames=[]
    for sent in doc.sents:
        if sent.text.rstrip().endswith('?') or any(t.is_quote for t in sent): continue
        matches=list(FREQUENCY.finditer(sent.text))
        if len(matches)!=1: continue
        for root in sent:
            if root.pos_!='VERB' or not any(c.dep_=='nsubj' and c.text.casefold()=='i' for c in root.children): continue
            chain=[root]+list(root.ancestors)
            if any(lemma(t) in ('say','tell','ask','think','believe','hope','wish','plan') for t in chain): continue
            if any(c.dep_=='neg' or (c.dep_=='aux' and lemma(c) in ('would','could','might','may','should','will','can'))
                   or (c.dep_=='mark' and lemma(c) in ('if','unless')) for t in chain for c in t.children): continue
            objects=[c for c in root.children if c.dep_=='dobj']
            if len(objects)!=1: continue
            obj=objects[0]; tokens=sorted(obj.subtree,key=lambda t:t.i)
            names=explicit_names(tokens)
            if not names: continue  # No inferred identity from pronouns.
            argument=' '.join(t.text for t in tokens if t.pos_=='PROPN').strip()
            if not argument: continue
            match=matches[0]; a=sent.start_char+match.start(); b=sent.start_char+match.end(); raw=text[a:b]
            frames.append({'owner':speaker.casefold(),'holder_terms':[],'contexts':[],
                'unit_terms':['frequency'],'unit_head':lemma(obj),'family':'frequency','predicate':lemma(root),
                'probe':'I '+lemma(root)+' '+argument+' regularly.','argument_text':argument,
                'argument_names':list(names),'value':raw,'normalized_value':raw.casefold(),
                'value_span':[a,b],'tense':'past' if 'Past' in root.morph.get('Tense') else 'nonpast',
                'kind':'observable_frequency_report_not_world_truth','projection':'typed_report_contract_v1'})
    return frames


def compatible(contract,frame,text):
    fc=frame['contract']; family=frame['family']; kind=contract['answer_kind']
    allowed={'frequency':{'frequency'},'quantity':{'quantity'},'existence':{'object'},
             'object_type':{'object'},'location':{'relation'},'property':{'property'}}
    if family not in allowed.get(kind,set()): return False
    # Names are input question constraints, not inferred from reference answers.
    words={w.casefold().strip('.') for w in re.findall(r"[\w.]+",text)}
    if not set(contract['named_scope']).issubset(words): return False
    if kind=='location': return True
    if kind=='frequency':
        return bool(contract['target_names']) and set(contract['target_names']).issubset(fc['argument_names'])
    return bool(fc['argument_head'] and fc['argument_head'] in contract['argument_heads'])


def project_query(doc,text,broad_query):
    """Keep source-free typed probes; latest events are not ongoing states."""
    contract=query_contract(doc,text); kind=contract['answer_kind']; probes=[]
    for p in broad_query['probes']:
        family=p['family']
        if family=='property' and kind=='property':
            core=p['unit_head'].split()[-1]
            if core in contract['argument_heads']: probes.append({**p,'unit_terms':[family,core]})
        elif family=='quantity' and kind=='quantity':
            core=p['unit_head']
            if core in contract['argument_heads']: probes.append({**p,'unit_terms':[family,core]})
        elif family=='relation' and kind=='location':
            probes.append({**p,'unit_terms':[family,p['unit_head']]})
    if kind in ('existence','object_type'):
        for predicate in contract['predicates']:
            for core in contract['argument_heads']:
                probes.append({'family':'object','unit_terms':['object',core],
                    'unit_head':predicate,'predicate':predicate,'probe':'I '+predicate+' a '+core+'.'})
    if kind=='frequency' and contract['target_names']:
        name=' '.join(contract['target_names'])
        for predicate in contract['predicates']:
            probes.append({'family':'frequency','unit_terms':['frequency','named:'+name],
                'unit_head':'frequency','predicate':predicate,'probe':'I '+predicate+' '+name+' regularly.'})
    mode=broad_query['mode']
    if kind=='object_type' and re.search(r'\b(?:latest|recent|recently|most recent)\b',text,re.I):
        mode='latest_event'  # A purchase event is not a persistent ownership report.
    if kind=='unsupported': mode='ordinary'
    return {**broad_query,'probes':probes,'mode':mode,'contract':contract,
            'projection':'typed_query_contract_v1'}
