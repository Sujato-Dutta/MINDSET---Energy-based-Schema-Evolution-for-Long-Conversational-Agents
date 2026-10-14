"""Development-only grounded report readout; never certifies world truth.

The NLI cache binds value-free predicate paraphrases, not owners or quantities.
Temporal order is REPORT order. A reported past quantity is not current state.
"""
from collections import Counter
from mindset.methods.mdl_state import transition_candidates, choose_transition, mode_value
from mindset.methods.joint import utc_time


class PredicateEquivalence:
    def __init__(self, pairs=()):
        self.labels = {(r['premise'], r['hypothesis']): r['label'] for r in pairs}

    def equivalent(self, a, b):
        return a == b or (self.labels.get((a, b)) == 'entailment'
                          and self.labels.get((b, a)) == 'entailment')


def typed_key(frame):
    return (frame['owner'], tuple(frame['holder_terms']), tuple(frame['unit_terms']), tuple(frame['contexts']))


class ReportCatalog:
    def __init__(self, source_rows, events, equivalence, *, policy='mdl'):
        if policy not in ('mdl', 'two_reports'): raise ValueError('Unknown report policy')
        self.eq = equivalence; self.policy = policy; self.groups = []; self.audit = []
        self.by_id = {e.event_id: e for e in events}
        for event in events:
            row = source_rows[event.event_id]
            if row['text'] != event.text or row['speaker'] != event.speaker:
                raise ValueError('Report projection/source mismatch')
            frames = row['reports']
            # Conflicting reports of a typed predicate in the same episode are
            # not resolved by clause order or future information.
            for frame in frames:
                start, end = frame['value_span']
                if event.text[start:end] != frame['value']: raise ValueError('Bad value span')
                if any(typed_key(f) == typed_key(frame) and self.eq.equivalent(f['probe'], frame['probe'])
                       and f['normalized_value'] != frame['normalized_value'] for f in frames): continue
                groups = [g for g in self.groups if g['key'] == typed_key(frame)
                          and all(self.eq.equivalent(frame['probe'], p) for p in g['probes'])]
                if len(groups) > 1: continue  # Do not merge ambiguous classes.
                if not groups:
                    group = {'key': typed_key(frame), 'probes': [], 'reports': [], 'closed': [], 'active': []}
                    self.groups.append(group)
                else: group = groups[0]
                if frame['probe'] not in group['probes']: group['probes'].append(frame['probe'])
                report = {'eid': event.event_id, 'session': event.session_id, 'timestamp': event.timestamp,
                          'value': frame['normalized_value'], 'probe': frame['probe'], 'tense': frame['tense']}
                if any(r['eid'] == report['eid'] for r in group['reports']): continue
                group['reports'].append(report); group['active'].append(report)
                self._advance(group)

    def _advance(self, group):
        items = group['active']; last = {r['session']: r for r in items}
        votes = [r for r in items if last[r['session']] is r]
        values = [r['value'] for r in votes]
        stamps = [utc_time(r['timestamp']) for r in items]
        ordered = bool(all(stamps) and stamps == sorted(stamps))
        preferred = group.get('mode')
        plans = transition_candidates(values, temporal=ordered, preferred=preferred)
        feasible = []
        for plan in plans:
            if plan['cut'] is not None:
                pos = items.index(votes[plan['cut']])
                if utc_time(items[pos-1]['timestamp']) >= utc_time(items[pos]['timestamp']): continue
                if {r['session'] for r in items[:pos]} & {r['session'] for r in items[pos:]}: continue
            feasible.append(plan)
        selected = choose_transition(feasible)
        if self.policy == 'two_reports':
            selected = feasible[0]
            if len(values) >= 2 and values[-1] == values[-2] and values[-1] != preferred:
                selected = next((p for p in feasible if p['cut'] == len(values)-2), selected)
        if selected['cut'] is not None:
            position = items.index(votes[selected['cut']])
            group['closed'].append(items[:position]); group['active'] = items[position:]
        active = group['active']; latest = {r['session']: r for r in active}
        group['mode'] = mode_value(Counter(r['value'] for r in latest.values()), preferred)
        self.audit.append({'source_id': items[-1]['eid'], 'key': group['key'], 'policy': self.policy,
            'chosen': selected, 'candidates': feasible, 'ordered_report_times': ordered,
            'meaning': 'observed report segmentation, NOT latent state/truth'})

    def bind(self, query_row):
        q = query_row['report_query']
        if q['mode'] not in ('report_compare', 'report_change') or not q['self_reference']:
            return None, 'not_supported_self_report_comparison'
        # Raw role mapping only; no name inference from embeddings/NLI.
        groups = [g for g in self.groups if g['key'][0] == 'user' and not g['key'][1]
                  and not g['key'][3] and any(tuple(p['unit_terms']) == g['key'][2]
                  and all(self.eq.equivalent(p['probe'], probe) for probe in g['probes']) for p in q['probes'])]
        if len(groups) != 1: return None, 'no_unique_typed_predicate_binding'
        return groups[0], 'unique_owner_unit_bidirectional_predicate_entailment'

    def witnesses(self, query_row, *, selector='energy'):
        group, reason = self.bind(query_row)
        certificate = {'binding': reason, 'reserved_ids': [], 'committed_report_edge': False,
                       'selector': selector, 'world_truth_certified': False, 'hard_exclusions': []}
        if group is None: return [], certificate
        reports = group['reports']
        certificate['bound_reports'] = len(reports)
        if selector == 'off': return [], certificate
        if selector == 'latest':
            ids=[reports[-1]['eid']]
            certificate['reserved_ids']=ids
            return ids,certificate
        if selector == 'chronological':
            if query_row['report_query']['mode'] == 'report_compare': pair = [reports[0], reports[-1]]
            else:
                latest = reports[-1]
                earlier = next((r for r in reversed(reports[:-1]) if r['value'] != latest['value']), None)
                pair = [earlier, latest] if earlier else []
        elif selector == 'energy':
            if not group['closed']: return [], certificate
            left = group['closed'][0] if query_row['report_query']['mode'] == 'report_compare' else group['closed'][-1]
            right = group['active']
            # Actual raw provenance of committed report distributions only.
            lm = mode_value(Counter(r['value'] for r in {r['session']: r for r in left}.values()))
            rm = group['mode']
            pair = [next(r for r in left if r['value'] == lm), next(r for r in reversed(right) if r['value'] == rm)]
            certificate['committed_report_edge'] = True
        else: raise ValueError('Unknown selector')
        ids = list(dict.fromkeys(r['eid'] for r in pair))
        if len(ids) != 2: return [], certificate
        certificate['reserved_ids'] = ids
        return ids, certificate


def inject_witnesses(pool, anchors, *, budget=48, direct_keep=24):
    """Same bounded reserve for matched controls; preserve original first24."""
    if len(set(pool)) != len(pool) or len(pool) > budget: raise ValueError('Bad pool')
    missing = [i for i in dict.fromkeys(anchors) if i not in pool]
    count = min(len(missing), max(0, len(pool)-min(direct_keep, len(pool))))
    return pool[:-count] + missing[:count] if count else list(pool)


def select_with_witnesses(pool, scores, witnesses, top_k=8):
    """Reserve at most two raw reports, never exclude other states as false."""
    if len(pool) != len(scores) or len(set(pool)) != len(pool): raise ValueError('Bad scores/pool')
    positions = dict(zip(pool, scores, strict=True))
    order = sorted(pool, key=lambda i: (-positions[i], i))
    reserved = [i for i in dict.fromkeys(witnesses) if i in positions][:min(2, top_k)]
    return (reserved + [i for i in order if i not in reserved])[:top_k]
