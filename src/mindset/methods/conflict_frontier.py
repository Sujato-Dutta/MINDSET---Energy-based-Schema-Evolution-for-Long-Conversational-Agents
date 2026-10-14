"""Energy-free, query-scoped raw witness frontier (development only).

An extraction failure is not proof that nothing changed. Opaque later messages
are retained as raw evidence, never promoted into inferred values or truth.
Exact lexical scope is conservative, not a complete semantic relevance test.
"""
from collections import defaultdict
import math
import re
import numpy as np
from mindset.methods.typed_grounding import TypedReportCatalog
from mindset.methods.joint import utc_time
from mindset.memory.energy import unit_vector
from mindset.retrieval.hybrid import BM25Index,ranked_indices,reciprocal_rank_fusion


class LeanEpisodeIndex:
    """Exact old flat access path, with no schema/energy construction."""
    def __init__(self,events,encoder):
        self.ids=[e.event_id for e in events]
        if len(set(self.ids))!=len(self.ids): raise ValueError('Duplicate immutable source IDs')
        self.by_id={eid:i for i,eid in enumerate(self.ids)}
        dense_docs=[f'[{e.timestamp}] {e.speaker}: {e.text}' for e in events]
        self.documents=[f'[{e.timestamp}] | Speaker: {e.speaker}\n{e.text}' for e in events]
        vectors=encoder.encode(dense_docs,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
        self.vectors=np.asarray([unit_vector(v) for v in vectors],dtype=np.float32)
        self.lexical=BM25Index(dense_docs)

    def candidates(self,query,encoder,budget=48):
        q=encoder.encode([query],normalize_embeddings=True,convert_to_numpy=True)[0]
        fused=reciprocal_rank_fusion([ranked_indices(self.vectors@q),ranked_indices(self.lexical.scores(query))])
        return sorted(fused,key=lambda i:(-fused[i],i))[:budget]


def words(text):
    return {w.casefold().strip('.') for w in re.findall(r'[\w.]+',text) if w.strip('.')}


def known_time(value):
    # utc_time's established failure sentinel is zero, not None. Fail closed
    # for that sentinel (including epoch-zero ambiguity) rather than guessing.
    stamp=utc_time(value)
    return stamp if stamp else None


class ConflictFrontierCatalog(TypedReportCatalog):
    def __init__(self,source_rows,events,equivalence,*,policy='latest'):
        if policy!='latest': raise ValueError('This candidate does not use energy policies')
        super().__init__(source_rows,events,equivalence,policy='mdl')
        self.lexical=defaultdict(set)
        self.source_order={e.event_id:i for i,e in enumerate(events)}
        for e in events:
            if e.speaker.casefold()=='user':
                for token in words(e.text): self.lexical[token].add(e.event_id)

    def _advance(self,group):
        # Preserve the source journal without inferring or committing a regime.
        group['mode']=group['reports'][-1]['value']
        self.audit.append({'source_id':group['reports'][-1]['eid'],'policy':'source_journal_no_energy',
                           'meaning':'source order, not inferred world-state validity'})

    def scoped_sources(self,contract):
        names=contract['named_scope']
        if contract['answer_kind']=='frequency':
            if not contract['target_names']: return set()
            scope=set(self.by_id)
            for name in contract['target_names']: scope &= self.lexical[name]
        else:
            scope=set()
            for head in contract['argument_heads']:
                # Exact head and regular plural surface forms only. No aliases,
                # semantic guesses or entity-name-specific exception lists.
                for term in (head,head+'s',head+'es'): scope |= self.lexical[term]
        for name in names: scope &= self.lexical[name]
        return scope

    def frontier(self,query_row,*,opaque=True,contrast=True):
        q=query_row['report_query']; group,reason=self.bind(query_row)
        cert={'binding':reason,'selector':'conflict_frontier','reserved_ids':[],
              'world_truth_certified':False,'energy_used':False,'hard_exclusions':[],
              'committed_report_edge':False,'opaque_tail_ids':[]}
        if group is None or q['mode']!='state_current': return [],cert
        deadline=known_time(query_row.get('question_date'))
        eligible=[r for r in group['reports'] if r['tense']=='nonpast'
                  and known_time(r['timestamp']) is not None
                  and (deadline is None or known_time(r['timestamp'])<=deadline)]
        if not eligible: return [],cert
        eligible.sort(key=lambda r:(known_time(r['timestamp']),self.source_order[r['eid']]))
        latest=eligible[-1]; latest_time=known_time(latest['timestamp'])
        anchors=[latest['eid']]
        if contrast:
            previous=next((r for r in reversed(eligible[:-1]) if r['value']!=latest['value']),None)
            if previous: anchors.append(previous['eid'])
        unknown=[]; known={r['eid'] for r in group['reports']}
        if opaque:
            for eid in self.scoped_sources(q['contract'])-known:
                stamp=known_time(self.by_id[eid].timestamp)
                if stamp is not None and stamp>latest_time and (deadline is None or stamp<=deadline): unknown.append(eid)
        if unknown:
            eid=max(unknown,key=lambda eid:(known_time(self.by_id[eid].timestamp),self.source_order[eid]))
            anchors=[eid,latest['eid']]; cert['opaque_tail_ids']=[eid]
        cert.update(reserved_ids=anchors,bound_reports=len(group['reports']),
                    observed_latest_value=latest['value'],
                    scope='at most two raw witnesses; NOT complete conflict coverage or truth certification')
        return anchors,cert


def select_frontier(pool,scores,witnesses,*,top_k=8,tie_keys=None):
    """Reserve witnesses in membership, but retain common CE output order."""
    if top_k<0 or len(pool)!=len(scores) or len(set(pool))!=len(pool): raise ValueError('Bad pool/budget')
    if any(not math.isfinite(float(s)) for s in scores): raise ValueError('Bad score')
    score=dict(zip(pool,scores,strict=True)); tie=tie_keys or {i:i for i in pool}
    order=sorted(pool,key=lambda i:(-score[i],tie[i]))
    chosen=list(dict.fromkeys(i for i in witnesses if i in score))[:min(2,top_k)]
    chosen+=(i for i in order if i not in chosen)  # Membership, not forced prefix.
    return sorted(chosen[:top_k],key=lambda i:(-score[i],tie[i]))
