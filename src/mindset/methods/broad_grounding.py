"""Isolated broader report grounding; old frozen candidates remain untouched."""
from mindset.methods.report_grounding import ReportCatalog


class BroadReportCatalog(ReportCatalog):
    def _advance(self,group):
        super()._advance(group)
        self.audit[-1]['predicate_probes']=list(group['probes'])

    def bind(self,query_row):
        q=query_row['report_query']
        if q['mode'] not in ('state_current','report_compare','report_change') or not q['self_reference']:
            return None,'unsupported_or_historical_query_operator'
        groups=[g for g in self.groups if g['key'][0]=='user' and not g['key'][1]
                and not g['key'][3] and any(tuple(p['unit_terms'])==g['key'][2]
                and all(self.eq.equivalent(p['probe'],probe) for probe in g['probes']) for p in q['probes'])]
        if len(groups)!=1: return None,'no_unique_owner_family_predicate_binding'
        return groups[0],'unique_broad_source_predicate_binding'

    def witnesses(self,query_row,*,selector='energy'):
        q=query_row['report_query']
        if q['mode']!='state_current': return super().witnesses(query_row,selector=selector)
        group,reason=self.bind(query_row)
        cert={'binding':reason,'reserved_ids':[],'committed_report_edge':False,'selector':selector,
              'world_truth_certified':False,'hard_exclusions':[]}
        if group is None: return [],cert
        cert['bound_reports']=len(group['reports'])
        if selector=='off': return [],cert
        if selector=='latest' or selector=='chronological':
            eligible=[r for r in group['reports'] if r['tense']=='nonpast']
        elif selector=='energy':
            eligible=[r for r in group['active'] if r['value']==group['mode'] and r['tense']=='nonpast']
        else: raise ValueError('Unknown selector')
        # Past-only descriptions cannot be promoted into current witnesses.
        if not eligible:
            cert['routing_reason']='no_nonpast_support_for_selected_report_distribution'
            return [],cert
        ids=[eligible[-1]['eid']]; cert['reserved_ids']=ids
        cert['committed_report_edge']=bool(selector=='energy' and group['closed'])
        cert['selected_observed_value']=eligible[-1]['value']
        return ids,cert
