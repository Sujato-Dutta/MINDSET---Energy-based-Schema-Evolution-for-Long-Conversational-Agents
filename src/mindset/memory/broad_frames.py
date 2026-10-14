"""Bounded generic dependency REPORTS; not comprehensive state/truth parsing.

Exact source spans, explicit owners and conservative temporal operators. No
benchmark IDs, answers, gold evidence, generated summaries or trained weights.
"""
from pathlib import Path
import importlib.util
import re

_spec=importlib.util.spec_from_file_location('numeric_report_projection',Path(__file__).with_name('report_frames.py'))
_numeric=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_numeric)
CHANGE=frozenset(('change','switch','update','increase','decrease','reduce','raise','lower','adjust','set'))
MODALS=frozenset(('would','could','might','may','should','will','can'))
TEMPORAL=frozenset(('current','new','old','previous','latest','recent','former'))


def lemma(t): return t.lemma_.casefold() or t.text.casefold()


def possessive(t):
    return t.dep_!='poss' and any(c.dep_=='poss' and c.text.casefold()=='my' for c in t.children)


def label(t):
    return ' '.join(lemma(c) for c in sorted([t]+[c for c in t.children if c.dep_ in ('compound','amod')
        and lemma(c) not in TEMPORAL],key=lambda x:x.i))


def subject(root):
    own=[t for t in root.children if t.dep_ in ('nsubj','nsubjpass')]
    if not own and root.dep_ in ('conj','xcomp'): return subject(root.head)
    return own


def permitted(root,sent):
    if sent.text.rstrip().endswith('?') or any(t.is_quote for t in sent): return False
    chain=[root]+list(root.ancestors)
    if any(lemma(t) in ('say','tell','ask','quote','hear','read','think','believe','hope','wish','plan') for t in chain): return False
    return not any(c.dep_=='neg' or (c.dep_ in ('aux','auxpass') and lemma(c) in MODALS)
        or (c.dep_=='mark' and lemma(c) in ('if','unless')) for t in chain for c in t.children)


def span(head,text):
    # Relative clauses/conjoined alternatives do not silently become a scalar.
    if any(t.dep_ in ('conj','relcl','acl') for t in head.subtree): return None
    ts=sorted(head.subtree,key=lambda t:t.i)
    if not ts: return None
    a,b=ts[0].idx,ts[-1].idx+len(ts[-1].text)
    return a,b,text[a:b]


def project_reports(doc,text,speaker):
    if '```' in text or text.lstrip().startswith(('Web search results:','URL:')): return []
    frames=[]
    for f in _numeric.project_reports(doc,text,speaker):
        frames.append({**f,'unit_terms':['quantity'],'family':'quantity','projection':'broad_extract_report_v1'})
    for sent in doc.sents:
        for root in sent:
            if root.pos_ not in ('VERB','AUX') or not permitted(root,sent): continue
            own=subject(root)
            if len(own)!=1: continue
            who=own[0]; direct=who.text.casefold()=='i'; owned=possessive(who)
            if not direct and not owned: continue
            owner=speaker.casefold(); args=[c for c in root.children if c.dep_ in ('attr','dobj','acomp','oprd')]
            preps=[(p,o) for p in root.children if p.dep_=='prep' for o in p.children if o.dep_=='pobj']
            definitions=[]
            if owned and lemma(root)=='be' and len(args)==1:
                definitions.append(('property',label(who),'My '+label(who)+' is unspecified.',args[0]))
            elif direct and lemma(root) in CHANGE:
                possessed=[a for a in args if possessive(a)]; destinations=[o for p,o in preps if lemma(p)=='to']
                if len(possessed)==1 and len(destinations)==1:
                    name=label(possessed[0]); definitions.append(('property',name,'My '+name+' is unspecified.',destinations[0]))
            if direct and lemma(root) not in CHANGE and lemma(root)!='be':
                for p,o in preps:
                    if o.ent_type_ in ('DATE','TIME') or lemma(p) in ('during','before','after','since','until','on','by'): continue
                    name=lemma(root)+' '+lemma(p)
                    definitions.append(('relation',name,'I '+name+' something.',o))
                for obj in args:
                    if obj.pos_ not in ('NOUN','PROPN') or any(t.like_num for t in obj.subtree): continue
                    if obj.ent_type_ in ('DATE','TIME'): continue
                    definitions.append(('object',lemma(root),'I '+lemma(root)+' something.',obj))
            for family,name,probe,arg in definitions:
                value=span(arg,text)
                if not value: continue
                a,b,raw=value
                # Contexts are only observable non-temporal adjuncts, not guesses.
                contexts=[]
                for p,o in preps:
                    if o==arg or o.ent_type_ in ('DATE','TIME') or lemma(p) not in ('at','during'): continue
                    if o.pos_=='PROPN': continue
                    contexts.append(' '.join(lemma(t) for t in p.subtree if not t.is_punct and t.pos_!='DET'))
                past='Past' in root.morph.get('Tense') or any(c.dep_=='aux' and c.text.casefold()=='had' for c in root.children)
                frame={'owner':owner,'ownership':'literal_first_person_or_direct_possession',
                    'holder_terms':[],'unit_terms':[family],'unit_head':name,'family':family,'predicate':name,'probe':probe,
                    'value':raw,'normalized_value':re.sub(r'\s+',' ',raw.strip().casefold()),'value_span':[a,b],
                    'contexts':sorted(set(contexts)),'tense':'past' if past else 'nonpast',
                    'kind':'observed_report_not_certified_world_state','projection':'broad_extract_report_v1'}
                if frame not in frames: frames.append(frame)
    return frames


def project_report_query(doc,text):
    numeric=_numeric.project_report_query(doc,text)
    probes=[{**p,'unit_terms':['quantity'],'family':'quantity'} for p in numeric['probes']]
    direct=any(t.dep_ in ('nsubj','nsubjpass') and t.text.casefold()=='i' for t in doc)
    owned=[t for t in doc if possessive(t)]
    for obj in owned:
        name=label(obj)
        probes.append({'unit_terms':['property'],'unit_head':name,'family':'property',
                       'probe':'My '+name+' is unspecified.','predicate':name})
    for root in doc:
        if root.pos_ not in ('VERB','AUX') or lemma(root)=='be' or not direct: continue
        own=subject(root)
        if not own or own[0].text.casefold()!='i': continue
        name=lemma(root)
        probes.append({'unit_terms':['object'],'unit_head':name,'family':'object',
                       'probe':'I '+name+' something.','predicate':name})
        prep_names={lemma(p) for p in root.children if p.dep_=='prep'}
        # Missing locative/prepositional argument in a wh-question is explicit
        # grammatical underspecification. Multiple bindings still abstain.
        if any(t.text.casefold()=='where' for t in doc): prep_names.update(('at','in','for','to'))
        for prep in sorted(prep_names):
            name=lemma(root)+' '+prep
            probes.append({'unit_terms':['relation'],'unit_head':name,'family':'relation',
                           'probe':'I '+name+' something.','predicate':name})
    mode=numeric['mode']
    terms=numeric['terms']
    past=bool(re.search(r'\b(?:previously|formerly|used to|before|did|was|were)\b',text,re.I))
    event=bool(re.search(r'\b(?:when|how long|how many times)\b',text,re.I))
    current=bool(re.search(r'\b(?:now|current|currently|latest|most recent|recently|today|still)\b',text,re.I))
    present=any(t.pos_ in ('VERB','AUX') and 'Pres' in t.morph.get('Tense') for t in doc)
    if mode=='ordinary':
        if event: mode='event'
        elif past and not current: mode='historical'
        elif (direct or owned) and (current or present): mode='state_current'
    if mode=='ordinary' and current: mode='state_current'
    return {**numeric,'self_reference':bool(direct or owned),'mode':mode,'probes':probes,
            'projection':'broad_query_operator_v1','temporal_assumptions':'explicit lexical/tense operators; not full semantic parsing'}
