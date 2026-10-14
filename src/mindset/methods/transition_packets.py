"""Source-backed transition evidence experiment, not a paper-default method.

Report contrast is not world-state change. This is bounded packet selection,
not thermodynamics, learned coefficients or a global memory optimizer.
"""
from collections import defaultdict
from dataclasses import dataclass
import re
from mindset.methods.joint import utc_time

ARMS = ('joint', 'naive_chain', 'compact_no_energy', 'compact_energy')
SINGLE_SLOTS = frozenset(('location','employer','occupation'))
SLOT_WORDS = {'location':r'\b(?:live|lived|living|reside|residence|move|moved|location|city)\b',
    'employer':r'\b(?:work|worked|working|employer|company|job)\b',
    'occupation':r'\b(?:occupation|profession|job|career|teacher|doctor|engineer)\b'}


@dataclass(frozen=True)
class Report:
    eid: str
    owner: str
    slot: str
    value: str
    context: tuple[str, ...]
    time: float
    explicit_change: bool
    source_value: str = ''


@dataclass(frozen=True)
class Contrast:
    left: str
    right: str
    owner: str
    slot: str
    kind: str
    left_context: tuple[str, ...] = ()
    right_context: tuple[str, ...] = ()
    schema_edge: tuple[str, str] = ()


def intent(query):
    text=query.casefold()
    if re.search(r'\b(?:change|changed|changes|switch|switched|stop|stopped|no longer|before and after|evolve|evolved)\b',text): return 'change'
    if re.search(r'\b(?:previous|previously|before|earlier|formerly|original|originally|used to)\b',text): return 'historical'
    if re.search(r'\b(?:now|current|currently|latest|today|still)\b',text): return 'current'
    return 'ordinary'


def slot_matches(slot,text):
    if slot in SLOT_WORDS: return bool(re.search(SLOT_WORDS[slot],text))
    if slot in ('preference','dislike'): return bool(re.search(r'\b(?:like|love|prefer|hate|dislike)\b',text))
    parts=slot.split('::'); family=parts[0]
    if family not in ('property','quantity','frequency'): return False
    target=parts[-1]
    words=re.findall(r'[a-z0-9]+',target)
    return bool(words) and all(re.search(r'\b'+re.escape(w)+r'(?:s|es)?\b',text) for w in words)


class TransitionCatalog:
    def __init__(self,memory,source_rows=None):
        self.reports=[]; self.by_key=defaultdict(list); self.edges=[]
        self.source_times={eid:utc_time(e.timestamp) for eid,e in memory.episodes.items()}
        for eid in memory._episode_order:
            e=memory.episodes[eid]; stamp=utc_time(e.timestamp)
            # Existing source extraction only, no invented values. Quotes and
            # hypothetical/negated/past-only reports are not current assertions.
            if not stamp or re.search(r'["“”]|\b(?:not|never|if|would|will|might|used to|formerly|previously|worked)\b',e.text,re.I): continue
            for slot,values in e.features.attribute_map().items():
                if slot not in SINGLE_SLOTS and slot not in ('preference','dislike'): continue
                if len(set(values))!=1: continue
                r=Report(eid,e.speaker.casefold(),slot,values[0],tuple(sorted(e.features.contexts)),stamp,e.features.explicit_change,values[0])
                self.reports.append(r); self.by_key[(r.owner,r.slot,r.context)].append(r)
        # Reuse an independently audited dependency projection, source rows only.
        # Property/count/frequency REPORTS extend access, not the frozen schemas.
        for eid,row in (source_rows or {}).items():
            e=memory.episodes[eid]
            if row['text']!=e.text or row['speaker']!=e.speaker or row['timestamp']!=e.timestamp:
                raise ValueError('Projection/source mismatch')
            stamp=utc_time(e.timestamp)
            for f in row['reports']:
                if f['family'] not in ('property','quantity','frequency') or f.get('holder_terms') or f['tense']!='nonpast' or not stamp: continue
                a,b=f['value_span']
                if not 0<=a<b<=len(e.text) or e.text[a:b]!=f['value']: raise ValueError('Generated report value')
                if f['owner']!=e.speaker.casefold(): continue
                slot='::'.join((f['family'],f['predicate'],f['unit_head']))
                if f['family']=='frequency':
                    names=f.get('contract',{}).get('argument_names',())
                    if not names: continue
                    slot='::'.join((f['family'],f['predicate'],' '.join(names)))
                r=Report(eid,f['owner'],slot,f['normalized_value'],tuple(sorted(f['contexts'])),stamp,e.features.explicit_change,f['value'])
                self.reports.append(r); self.by_key[(r.owner,r.slot,r.context)].append(r)
        self.ambiguous_time_groups=0
        for (owner,slot,context),reports in sorted(self.by_key.items()):
            at_time=defaultdict(set)
            for r in reports: at_time[r.time].add(r.value)
            ambiguous={t for t,v in at_time.items() if len(v)>1}
            self.ambiguous_time_groups+=len(ambiguous)
            reports[:]=list(dict.fromkeys(r for r in reports if r.time not in ambiguous))
            reports.sort(key=lambda r:(r.time,memory.episodes[r.eid].ordinal))
            for old,new in zip(reports,reports[1:]):
                if old.value==new.value or old.time>=new.time: continue
                # Unqualified multi-valued likes are not mutually exclusive.
                if slot not in SINGLE_SLOTS and not slot.startswith(('property::','quantity::','frequency::')): continue
                self.edges.append(Contrast(old.eid,new.eid,owner,slot,
                    'change_cue_report_contrast' if new.explicit_change else 'observed_report_contrast',context,context))
        groups=defaultdict(list)
        for key,reports in self.by_key.items():
            if key[2] and reports: groups[key[:2]].append(reports[-1])
        for (owner,slot),reports in sorted(groups.items()):
            for i,a in enumerate(reports):
                for b in reports[i+1:]:
                    if a.context!=b.context and a.value!=b.value:
                        self.edges.append(Contrast(a.eid,b.eid,owner,slot,'observed_context_contrast',a.context,b.context))
        # Actual frozen version edges are separately labelled and source-backed.
        for sid,s in sorted(memory.schemas.items()):
            for child in s.successor_schema_ids:
                new=memory.schemas[child]
                old_ids=[e for e in s.support_episode_ids if memory.episodes[e].features.attributes]
                new_ids=[e for e in new.support_episode_ids if memory.episodes[e].features.attributes]
                if not old_ids or not new_ids: continue
                a=max(old_ids,key=lambda e:(utc_time(memory.episodes[e].timestamp),memory.episodes[e].ordinal))
                b=min(new_ids,key=lambda e:(utc_time(memory.episodes[e].timestamp),memory.episodes[e].ordinal))
                x,y=memory.episodes[a],memory.episodes[b]
                if x.speaker.casefold()!=y.speaker.casefold() or not utc_time(x.timestamp) or utc_time(x.timestamp)>=utc_time(y.timestamp): continue
                for slot in sorted(set(x.features.attribute_map()) & set(y.features.attribute_map()) & SINGLE_SLOTS):
                    if x.features.attribute_map()[slot]==y.features.attribute_map()[slot]: continue
                    self.edges.append(Contrast(a,b,x.speaker.casefold(),slot,'schema_supersession',
                        tuple(sorted(x.features.contexts)),tuple(sorted(y.features.contexts)),(sid,child)))
        self.edges=list(dict.fromkeys(self.edges)); self.incident=defaultdict(list)
        for edge in self.edges:
            self.incident[edge.left].append(edge); self.incident[edge.right].append(edge)

    def bind(self,query,ranked_ids,memory,question_date=''):
        """Retrieval-anchored exact ownership + attribute/explicit context scope.

        No answer, gold ID, category or generated plan enters admission. This
        remains incomplete semantic grounding; no alias inference is promised.
        """
        text=query.casefold(); mode=intent(query); deadline=utc_time(question_date)
        anchors=set(ranked_ids[:4]); candidates=[]
        for edge in self.edges:
            if not anchors.intersection((edge.left,edge.right)): continue
            owner=bool(re.search(r'(?<!\w)'+re.escape(edge.owner)+r'(?!\w)',text))
            if edge.owner=='user': owner=owner or bool(re.search(r'\b(?:i|my|me)\b',text))
            if not owner: continue
            if re.search(r"(?:'s|’s|\bmy)\s+(?:wife|husband|sister|brother|mother|father|friend|daughter|son)\b",text): continue
            if deadline and any(utc_time(memory.episodes[e].timestamp)>deadline for e in (edge.left,edge.right)): continue
            contextual=any(c.casefold() in text for c in (*edge.left_context,*edge.right_context))
            slot=slot_matches(edge.slot,text)
            if not slot: continue
            if edge.kind=='observed_context_contrast':
                if not contextual: continue
            elif mode=='ordinary': continue
            candidates.append(edge)
        position={e:i for i,e in enumerate(ranked_ids)}
        candidates.sort(key=lambda e:(min(position.get(e.left,999),position.get(e.right,999)),
            0 if e.kind=='schema_supersession' else 1,e.left,e.right))
        return candidates[:4],mode

    def chain(self,edge,question_date=''):
        """Entire observed report sequence, NOT inferred validity history."""
        deadline=utc_time(question_date)
        reports=self.by_key.get((edge.owner,edge.slot,edge.right_context),())
        ids=[edge.left,edge.right] if edge.kind=='observed_context_contrast' or not reports else [r.eid for r in reports]
        return list(dict.fromkeys(e for e in ids if not deadline or
            (self.source_times.get(e,0) and self.source_times[e]<=deadline)))

    def source_values(self,edge):
        return {eid:[r.source_value for r in self.reports if r.eid==eid and r.slot==edge.slot]
                for eid in (edge.left,edge.right)}


def required_ids(edge,mode,query):
    if edge.kind=='observed_context_contrast':
        text=query.casefold()
        left=any(c.casefold() in text for c in edge.left_context)
        right=any(c.casefold() in text for c in edge.right_context)
        if left and not right: return (edge.left,)
        if right and not left: return (edge.right,)
        return (edge.left,edge.right)
    if mode=='historical': return (edge.left,)
    # An explicit contrast packet supplies both endpoints, including on current
    # queries. Report contrast does not label either endpoint false.
    return (edge.left,edge.right)


def inject(pool,edges,budget=48):
    if len(set(pool))!=len(pool): raise ValueError('Duplicate candidates')
    endpoints=list(dict.fromkeys(eid for edge in edges for eid in (edge.left,edge.right)))
    missing=[e for e in endpoints if e not in pool]
    count=min(len(missing),max(0,len(pool)-24))
    return [*pool[:-count],*missing[:count]] if count else list(pool)


def select_packet(arm,ranking,edges,mode,query,catalog,pack,*,question_date=''):
    """Finite packet search; lexicographic energy has no tuned coefficients.

    Objective: missing source witnesses, dangling supersession endpoints,
    negative CE-rank utility, tokens. Exact minimization ONLY over the logged
    proposal set (<=5 packets); not a universal minimality theorem. The CE-only
    control sees identical pool, proposals, budget and output-order policy.
    """
    if arm not in ARMS: raise ValueError('Unknown packet policy')
    if not edges:
        selected,packet=pack(ranking)
        return selected,packet,{'bound':False,'reason':'no_grounded_contrast','energy_used':False,'proposals':[]}
    if arm=='naive_chain':
        order=[*catalog.chain(edges[0],question_date),*ranking]
        order=list(dict.fromkeys(e for e in order if e in ranking))
        selected,packet=pack(order)
        return selected,packet,{'bound':True,'reason':'observed_sequence_expansion',
            'energy_used':False,'edges':[edges[0].__dict__],'proposals':[]}
    if arm=='joint':
        selected,packet=pack(ranking)
        return selected,packet,{'bound':False,'reason':'ordinary_retrieval','energy_used':False,'proposals':[]}
    utility={eid:1/(1+rank) for rank,eid in enumerate(ranking)}
    # The highest-ranked admitted relation supplies source-only obligations.
    target=edges[0]; required=set(required_ids(target,mode,query))
    proposals=[]; packets=[]
    for reserved in [(),*(required_ids(e,mode,query) for e in edges)]:
        order=list(dict.fromkeys([*reserved,*ranking])); selected,packet=pack(order)
        ids={s['event_id'] for s in selected}; tokens=pack.token_count(packet)
        witnesses=pack.witness_ids(selected) if hasattr(pack,'witness_ids') else ids
        dangling=sum(e.kind=='schema_supersession' and e.left in ids and e.right not in ids for e in edges)
        relevance=sum(utility.get(e,0.) for e in ids)
        energy=(len(required-witnesses),dangling,-relevance,tokens)
        proposals.append({'reserved_ids':list(reserved),'selected_ids':[s['event_id'] for s in selected],
            'energy':list(energy),'ce_rank_utility':relevance,'tokens':tokens,
            'witness_ids_in_passages':sorted(required & witnesses)})
        packets.append((selected,packet))
    best=min(range(len(proposals)),key=lambda i:(tuple(proposals[i]['energy']),i)) if arm=='compact_energy' else min(
        range(len(proposals)),key=lambda i:(-proposals[i]['ce_rank_utility'],proposals[i]['tokens'],i))
    selected,packet=packets[best]
    return selected,packet,{'bound':True,'reason':'finite_source_witness_packet_selection','energy_used':arm=='compact_energy',
        'edges':[e.__dict__ for e in edges],'required_ids':sorted(required),
        'witnesses_survived':proposals[best]['witness_ids_in_passages'],
        'selected_proposal':best,'proposals':proposals,'objective_scope':'bounded source access; not world-truth or thermodynamics'}
