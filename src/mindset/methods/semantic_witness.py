"""Slot-free source witness candidate, isolated and not a novelty claim.

NLI labels describe a model's reading of literal report spans, not world truth.
No report is deleted, no summary/fact is generated, no scalar mixture is fitted.
"""
from dataclasses import dataclass
import re
import numpy as np
from mindset.methods.joint import utc_time
from mindset.methods.transition_packets import intent

ARMS=('joint','flat48','semantic_pool','no_verifier','semantic_cover','no_order')


@dataclass(frozen=True)
class Span:
    eid: str
    start: int
    end: int
    text: str
    speaker: str


def source_spans(events):
    """Literal sentence spans; admit first-person REPORTS, including past/negated.

    A report may be uncertain, a plan or a belief; it is never promoted to truth.
    Quoted/code/search text is not automatically attributed to the speaker.
    Long/opaque sentences remain available through the ordinary retrieval path.
    """
    output=[]
    for e in events:
        if '```' in e.text or re.search(r'["\u201c\u201d]',e.text) or e.text.lstrip().startswith(('Web search results:','URL:')): continue
        spans=[]
        for match in re.finditer(r'[^.!?\n]+[.!?]?',e.text):
            text=match.group().strip()
            if not text or text.endswith('?') or len(text)>700 or re.search(r'["“”]',text): continue
            if not re.search(r'\b(?:I|my|mine)\b',text,re.I): continue
            start=match.start()+len(match.group())-len(match.group().lstrip())
            end=start+len(text)
            if e.text[start:end]!=text: raise ValueError('Non-source sentence span')
            spans.append(Span(e.event_id,start,end,text,e.speaker.casefold()))
        output.extend(spans[:12])
    return output


def semantic_pairs(index,spans,vectors,query_vector,anchors,events,neighbors=4,question_date=''):
    """Query-time lookups over already observed history; max4x4 source pairs."""
    by_episode={}
    for i,s in enumerate(spans): by_episode.setdefault(s.eid,[]).append(i)
    original={e.event_id:e for e in events}; deadline=utc_time(question_date); selected=[]
    for eid in anchors[:4]:
        choices=by_episode.get(eid,[])
        if not choices: continue
        if deadline and (not utc_time(original[eid].timestamp) or utc_time(original[eid].timestamp)>deadline): continue
        left=max(choices,key=lambda i:(float(vectors[i]@query_vector),-i)); source=spans[left]
        candidates=[]
        for other,ids in by_episode.items():
            if other==eid or original[other].speaker.casefold()!=source.speaker: continue
            if deadline and (not utc_time(original[other].timestamp) or utc_time(original[other].timestamp)>deadline): continue
            right=max(ids,key=lambda i:(float(vectors[i]@vectors[left]),-i))
            candidates.append((float(vectors[right]@vectors[left]),other,right))
        candidates.sort(key=lambda item:(-item[0],item[1]))
        selected.extend((left,right) for _,other,right in candidates[:neighbors])
    return list(dict.fromkeys(selected))


def classified_edges(pairs,spans,labels,events):
    if len(labels)!=2*len(pairs): raise ValueError('Bidirectional labels missing')
    order={e.event_id:i for i,e in enumerate(events)}; edges=[]
    for i,(a,b) in enumerate(pairs):
        x,y=spans[a],spans[b]; fwd,rev=labels[2*i:2*i+2]
        if fwd not in ('contradiction','entailment','neutral','too_long') or rev not in ('contradiction','entailment','neutral','too_long'):
            raise ValueError('Unknown semantic label')
        if 'too_long' in (fwd,rev): continue
        relation='contrast' if fwd==rev=='contradiction' else 'equivalent_report' if fwd==rev=='entailment' else 'unknown'
        if relation=='unknown': continue
        if x.speaker!=y.speaker: raise ValueError('Cross-author pair admitted')
        if order[x.eid]>order[y.eid]: x,y=y,x
        edges.append({'left':x.eid,'right':y.eid,'relation':relation,'left_span':[x.start,x.end],
            'right_span':[y.start,y.end],'left_text':x.text,'right_text':y.text,'speaker':x.speaker,
            'label_scope':'model-labelled report relation, not truth/supersession'})
    unique={tuple((e['left'],e['right'],e['relation'],*e['left_span'],*e['right_span'])):e for e in edges}
    return list(unique.values())


def extend_pool(pool,edges,budget=48):
    if len(set(pool))!=len(pool) or len(pool)>budget: raise ValueError('Invalid base pool')
    extra=list(dict.fromkeys(eid for e in edges for eid in (e['left'],e['right']) if eid not in pool))
    n=min(len(extra),max(0,len(pool)-24))
    return [*pool[:-n],*extra[:n]] if n else list(pool)


def choose_packet(arm,ranking,edges,question,pack,order):
    """Finite witness proposals, shared pool/reader/budget; not global optimality."""
    if arm not in ARMS: raise ValueError('Unknown semantic-witness arm')
    if arm in ('joint','flat48','semantic_pool','no_verifier'):
        selected,packet=pack(ranking); return selected,packet,{'policy':'CE-only','proposals':[]}
    position={e:i for i,e in enumerate(ranking)}; mode=intent(question)
    temporal=mode!='ordinary' or bool(re.search(r'\b(?:when|date|year|month|how long)\b',question,re.I))
    contrasts=[e for e in edges if e['relation']=='contrast' and {e['left'],e['right']}.issubset(position)]
    contrasts.sort(key=lambda e:(min(position[e['left']],position[e['right']]),e['left'],e['right']))
    target=contrasts[:1] if mode!='ordinary' else []
    required={eid for e in target for eid in (e['left'],e['right'])}
    equivalents=[e for e in edges if e['relation']=='equivalent_report'] if not temporal else []
    reservations=[(),*(tuple(e[k] for k in ('left','right')) for e in contrasts[:4])]
    if equivalents:
        # A source-equivalence class may affect only this finite query readout;
        # never infer transitivity, erase archives or pretend universal equivalence.
        filtered=[]
        for eid in ranking:
            if any({eid,prior}=={e['left'],e['right']} for prior in filtered for e in equivalents): continue
            filtered.append(eid)
        reservations.append(tuple(filtered[:8]))
    proposals=[]; packets=[]
    for reserved in reservations:
        ids=list(dict.fromkeys([*reserved,*ranking])); selected,packet=pack(ids)
        chosen={s['event_id'] for s in selected}
        witnessed=set()
        for e in target:
            for side in ('left','right'):
                eid=e[side]; needle=e[side+'_text']
                if any(s['event_id']==eid and ' '.join(needle.casefold().split()) in ' '.join(s['text'].casefold().split()) for s in selected): witnessed.add(eid)
        redundant=sum({e['left'],e['right']}.issubset(chosen) for e in equivalents)
        utility=sum(1/(1+position[e]) for e in chosen)
        cost=(len(required-witnessed),redundant,-utility,pack.token_count(packet))
        proposals.append({'reserved':list(reserved),'selected_ids':[s['event_id'] for s in selected],
            'witness_ids':sorted(witnessed),'cost':list(cost)}); packets.append((selected,packet))
    best=min(range(len(proposals)),key=lambda i:(tuple(proposals[i]['cost']),i)); selected,packet=packets[best]
    if arm=='semantic_cover' and temporal:
        reordered=sorted(selected,key=lambda s:(order[s['event_id']],s['event_id']))
        candidate='\n\n'.join(f'[{i}] {s["text"]}' for i,s in enumerate(reordered,1))
        if pack.token_count(candidate)<=pack.budget: selected,packet=reordered,candidate
    return selected,packet,{'policy':'finite_semantic_witness_cover','mode':mode,'target_edges':target,
        'equivalent_report_edges':len(equivalents),'proposals':proposals,'selected_proposal':best,
        'global_minimality_or_world_truth_claim':False}
