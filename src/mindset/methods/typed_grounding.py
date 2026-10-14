"""Isolated argument/type-safe report catalog, not an energy novelty claim."""
from mindset.methods.broad_grounding import BroadReportCatalog
from mindset.memory.query_contracts import compatible


class TypedReportCatalog(BroadReportCatalog):
    def __init__(self,source_rows,events,equivalence,*,policy='mdl'):
        self.source_rows=source_rows
        super().__init__(source_rows,events,equivalence,policy=policy)

    def bind(self,query_row):
        q=query_row['report_query']
        if q['mode'] not in ('state_current','report_compare','report_change') or not q['self_reference']:
            return None,'unsupported_or_historical_query_operator'
        contract=q.get('contract')
        if not contract: return None,'missing_typed_query_contract'
        groups=[]
        for g in self.groups:
            if g['key'][0]!='user' or g['key'][1] or g['key'][3]: continue
            if not any(tuple(p['unit_terms'])==g['key'][2]
                       and all(self.eq.equivalent(p['probe'],probe) for probe in g['probes'])
                       for p in q['probes']): continue
            # Every report in a state sequence must refer to the same queried
            # argument. A subset of matching reports cannot certify the state
            # of a group polluted by a different named topic.
            safe=True
            for report in g['reports']:
                row=self.source_rows[report['eid']]
                frames=[f for f in row['reports'] if tuple(f['unit_terms'])==g['key'][2]
                        and f['probe']==report['probe'] and f['normalized_value']==report['value']]
                if not frames or not all(compatible(contract,f,row['text']) for f in frames):
                    safe=False; break
            if safe: groups.append(g)
        if len(groups)!=1: return None,'no_unique_argument_type_safe_binding'
        return groups[0],'unique_argument_type_safe_source_binding'
