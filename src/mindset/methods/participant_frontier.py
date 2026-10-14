"""Same frontier mechanism with explicit metadata-grounded speaker namespaces."""
from collections import defaultdict
from mindset.methods.conflict_frontier import ConflictFrontierCatalog,words
from mindset.memory.query_contracts import compatible


class ParticipantFrontierCatalog(ConflictFrontierCatalog):
    def __init__(self,source_rows,events,equivalence,*,policy='latest'):
        super().__init__(source_rows,events,equivalence,policy=policy)
        self.owned_lexical=defaultdict(set)
        for e in events:
            for token in words(e.text): self.owned_lexical[(e.speaker.casefold(),token)].add(e.event_id)

    def scoped_sources(self,contract):
        owner=contract.get('owner')
        if not owner: return set()
        if contract['answer_kind']=='frequency':
            if not contract['target_names']: return set()
            scope=set(self.by_id)
            for name in contract['target_names']: scope &= self.owned_lexical[(owner,name)]
        else:
            scope=set()
            for head in contract['argument_heads']:
                for term in (head,head+'s',head+'es'): scope |= self.owned_lexical[(owner,term)]
        for name in contract['named_scope']: scope &= self.owned_lexical[(owner,name)]
        return scope

    def bind(self,query_row):
        q=query_row['report_query']; contract=q.get('contract',{}); owner=contract.get('owner')
        if not owner or not q['self_reference'] or q['mode'] not in ('state_current','report_compare','report_change'):
            return None,'no_supported_explicit_query_owner'
        groups=[]
        for g in self.groups:
            if g['key'][0]!=owner or g['key'][1] or g['key'][3]: continue
            if not any(tuple(p['unit_terms'])==g['key'][2]
                and all(self.eq.equivalent(p['probe'],probe) for probe in g['probes']) for p in q['probes']): continue
            safe=True
            for report in g['reports']:
                row=self.source_rows[report['eid']]
                frames=[f for f in row['reports'] if tuple(f['unit_terms'])==g['key'][2]
                    and f['probe']==report['probe'] and f['normalized_value']==report['value']]
                if not frames or not all(compatible(contract,f,row['text']) for f in frames): safe=False; break
            if safe: groups.append(g)
        if len(groups)!=1: return None,'no_unique_speaker_argument_type_binding'
        return groups[0],'unique_explicit_speaker_argument_type_binding'
