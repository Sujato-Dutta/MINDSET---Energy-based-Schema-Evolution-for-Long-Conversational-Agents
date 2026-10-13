"""Isolated v2 development candidate: grounded state routing and compiled readout.

No benchmark-fitted coefficients. Self-report parsing is deliberately bounded
English normalization, NOT a learned semantic parser or biological claim.
Original source text, timestamps, IDs and prior implementations are preserved.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import replace
import re
from time import perf_counter
import numpy as np

from mindset.methods.joint import typed_features, utc_time
from mindset.methods.state_readout import StateAwareMindsetMemory, QueryScope, explicit_period, _stamp
from mindset.memory.energy import unit_vector


def self_report_view(text):
    """Only assertive first-person clauses; canonicalization is parsing-only."""
    clauses = re.split(r'[.!?;\n]+|\band\s+(?=I\b|my\b)', text, flags=re.I)
    accepted = []
    for clause in clauses:
        clause = clause.strip()
        if not re.match(r"^(?:I\b|I'm\b|I've\b|My\b)", clause, re.I): continue
        if re.search(r'\b(?:said|says|told|asked|quoted|used to|formerly|previously)\b', clause, re.I): continue
        if re.search(r"\b(?:don't|do not|am not|no longer|didn't|did not)\b", clause, re.I): continue
        clause = re.sub(r"\bI'm\b", 'I am', clause, flags=re.I)
        clause = re.sub(r"\bI've\b", 'I have', clause, flags=re.I)
        clause = re.sub(r'\bfavourite\b', 'favorite', clause, flags=re.I)
        clause = re.sub(r'\bI\s+(now|currently)\s+(live|work|prefer)\b', r'I \2 currently', clause, flags=re.I)
        # Keep change cues in a separate prefix rather than erasing them.
        changed = bool(re.search(r'\b(?:now|currently|moved|relocated|joined|changed|switched)\b', clause, re.I))
        clause = re.sub(r'\bI live currently\b', 'I currently live', clause, flags=re.I)
        clause = re.sub(r'\bI work currently\b', 'I currently work', clause, flags=re.I)
        clause = re.sub(r'\bI prefer currently\b', 'I prefer', clause, flags=re.I)
        clause = re.sub(r'\bI\s+(?:am residing|reside|am based)\s+in\b', 'I live in', clause, flags=re.I)
        clause = re.sub(r'\bI\s+(?:have\s+)?(?:recently\s+)?(?:moved|relocated)\s+to\b', 'I moved to', clause, flags=re.I)
        clause = re.sub(r'\bI\s+(?:am employed by|am working for|am working at|have joined|joined)\b', 'I work for', clause, flags=re.I)
        clause = re.sub(r'\bmy\s+(?:current\s+)?(?:home|address|residence|location)\s+is\s+(?:in\s+)?', 'I live in ', clause, flags=re.I)
        # Existing extractor supports favorites/relationships, but not arbitrary
        # occupations. Normalize a self-reported role without a job-name list.
        role = re.match(r'I\s+(?:currently\s+)?(?:work as|am working as)\s+(?:a|an)\s+(.+)', clause, re.I)
        if role: clause = 'My occupation is ' + role[1]
        known_role = re.match(r'I am\s+(?:a|an)\s+(teacher|doctor|nurse|engineer|designer|student|chef|lawyer|artist)\b(.*)', clause, re.I)
        if known_role: clause = 'My occupation is ' + known_role[1] + known_role[2]
        accepted.append(('Currently, ' if changed else '') + clause)
    return '. '.join(accepted)


def observable_period(text):
    """Recognize native slash timestamps and day-first dates without clock use."""
    import calendar
    # Reject multiple explicit dates rather than silently selecting the first.
    matches = re.findall(r'\b(\d{4})[-/](\d{2})[-/](\d{2})\b', text)
    if len(matches) > 1: return 0., 0.
    try:
        if matches:
            start = _stamp(*map(int, matches[0])); return start, start + 86400.
        names = '|'.join(calendar.month_name[1:])
        match = re.search(rf'\b(\d{{1,2}})\s+({names})\s+(\d{{4}})\b', text, re.I)
        if match:
            month = next(i for i, name in enumerate(calendar.month_name) if name.casefold() == match[2].casefold())
            start = _stamp(int(match[3]), month, int(match[1])); return start, start + 86400.
    except (ValueError, OverflowError): return 0., 0.
    return explicit_period(text)


class AlignedMindsetMemory(StateAwareMindsetMemory):
    """Frozen scalar transfer; query binding is derived from observed slot names.

The energy decision still executes through the audited Joint controller. A
parsing-view adapter broadens explicit self-report aliases, without storing
generated schema summaries. One transition per raw episode remains intact.
"""
    retrieval_version = 'aligned-compiled-state-v2-development'

    def __init__(self, *args, write_policy='energy', **kwargs):
        if write_policy not in ('energy', 'rule', 'nearest'): raise ValueError('Unknown write policy')
        self.write_policy = write_policy
        self.last_route_certificate = {}
        super().__init__(*args, **kwargs)

    def _plans(self, schema, episode):
        plans = super()._plans(schema, episode)
        if self.write_policy == 'rule':
            s = plans[0]['signals']
            preferred = 'split' if s.contextual_potential else 'supersede' if s.temporal_potential else 'reinforce'
            for p in plans: p['cost'] = float(p['action'] != preferred)
        return plans

    def _ingest_episode(self, episode):
        view = self_report_view(episode.text)
        # Occupation alias is made recognizable to the unchanged Joint parser.
        # Its generic relationship spelling is retained internally for this
        # adapter; the read slot family maps it back to occupation.
        incoming = replace(episode, text=view)
        if self.write_policy == 'nearest':
            incoming = replace(incoming, features=typed_features(incoming.speaker, view))
            topic = f'{episode.speaker.casefold()}::{incoming.features.primary_slot}'
            pool = [self.schemas[sid] for sid in self._by_topic[topic] if self.schemas[sid].status == 'active']
            self.episodes[episode.episode_id] = incoming
            if not pool:
                s = self._create_schema(incoming, action='new')
                self._record_transition(incoming, 'new', None, (s.schema_id,), {}, None)
            else:
                target = max(pool, key=lambda s: (float(s.centroid @ episode.vector), s.schema_id))
                target.support_episode_ids.append(episode.episode_id); target._add_features(incoming.features)
                self._sums[target.schema_id] += episode.vector
                total = self._sums[target.schema_id]; target.centroid = unit_vector(total)
                target.mean_distortion = float(np.clip((1 - np.linalg.norm(total) / len(target.support_episode_ids)) / 2, 0, 1))
                self._latest[target.schema_id] = max(self._latest[target.schema_id], utc_time(episode.timestamp))
                self._record_transition(incoming, 'reinforce', target.schema_id, (target.schema_id,), {}, None)
        else:
            super()._ingest_episode(incoming)
            if self.write_policy == 'rule':
                cert = self._decision_certificates.get(episode.episode_id)
                if cert:
                    record = self.operation_log.all()[-1]
                    margin = 0. if self.config.ablation == 'no_hysteresis' else self.config.switch_margin
                    for p in cert['feasible_candidates']:
                        p['rule_score'] = p['adjusted_energy']
                        p['adjusted_energy'] = p['terms']['total'] + margin * (p['action'] == 'supersede')
                    chosen = next(p for p in cert['feasible_candidates']
                                  if p['schema_id'] == record.target_schema_id and p['action'] == record.operation)
                    cert.update(policy='fixed_rule_control', selected_adjusted_energy=chosen['adjusted_energy'],
                        optimality_gap=max(0., chosen['adjusted_energy'] - min(p['adjusted_energy'] for p in cert['feasible_candidates'])))
        # No parsing view is exposed as source evidence. Features agree with
        # the actual schema slot spelling used by the unchanged write controller.
        self.episodes[episode.episode_id] = replace(episode, features=typed_features(episode.speaker, view))

    def _scope(self, query, question_date=''):
        text = re.sub(r"\bwhat's\b", 'what is', query, flags=re.I)
        owners = tuple(sorted(s.casefold() for s in self._speakers
                              if re.search(r'(?<!\w)' + re.escape(s) + r'(?!\w)', text, re.I)))
        slots = set()
        aliases = {
            'location': r'\b(?:live|lives|lived|living|reside|resides|residence|address|based|location|home city)\b',
            'employer': r'\b(?:employer|company|organization|work for|works for|worked for|work at|works at|worked at|employed)\b',
            'occupation': r'\b(?:occupation|profession|job|role|career|work as|works as|worked as)\b',
            'preference': r'\b(?:prefer|prefers|preferred|preference|preferences)\b',
        }
        observed = set(self._slot_words)
        for family, pattern in aliases.items():
            if re.search(pattern, text, re.I):
                slots.update(s for s in observed if s in (family, 'relationship_' + family))
        for slot, pattern in self._slot_words.items():
            if slot.startswith('favorite_') and not re.search(r'\bfavou?rite\b', text, re.I): continue
            if slot.startswith(('favorite_', 'relationship_')) and pattern.search(text): slots.add(slot)
        binding = 'explicit_named_speaker' if owners else 'unresolved'
        # The native role is an observable identity convention, not a guessed
        # pronoun referent. Never transfer a friend's state to the asking user.
        direct_self = bool(re.search(r'\b(?:do|did|am|can|have)\s+I\b|\bI\s+(?:live|lived|work|worked|prefer|preferred)\b', text, re.I))
        possessive = bool(re.search(r'\bmy\s+(?:current\s+|new\s+|latest\s+)?(?:employer|job|profession|occupation|address|location|preference|favorite\s+\w+)\b', text, re.I))
        relation_self = bool(re.search(r'\b(?:who|what)\s+(?:is|was|are|were)\s+my\b', text, re.I))
        nested = bool(re.search(r"\bmy\s+\w+['’]s\b|\b(?:friend|daughter|son|mother|father)\s+(?:live|lives|work|works)\b", text, re.I))
        roles = {s.casefold() for s in self._speakers}
        if not owners and 'user' in roles and roles <= {'user', 'assistant', 'system', 'tool', 'unknown'} and not nested and (direct_self or possessive or relation_self):
            owners = ('user',); binding = 'first_person_bound_to_literal_user_role'
        start, end = observable_period(text)
        # A range/comparison is not a point state even if one date parses first.
        multi = bool(re.search(r'\b(?:between|compare|versus|before and after)\b', text, re.I))
        event = bool(re.search(r'\b(?:when|how long|since when|changed|change|changes|moved|move|timeline)\b', text, re.I)) or multi
        past = bool(re.search(r'\b(?:previously|formerly|used to|did|was|were|lived|worked|preferred)\b', text, re.I))
        current = bool(re.search(r'\b(?:now|currently|current|still|today|latest|does|do|is|are|am)\b', text, re.I))
        mode = 'event' if event else 'as_of' if start and slots else 'historical' if past else 'current' if current and slots else 'ordinary'
        if mode == 'current' and question_date:
            ds, de = observable_period(question_date)
            if ds: start, end = ds, de
        self._last_scope_binding = binding
        return QueryScope(mode, owners, tuple(sorted(slots)), tuple(sorted(typed_features('', text).contexts)), start, end)

    def _assessment(self, index, scope):
        eid = self._episode_order[index]
        records = [r for r in self._compiled_states[eid]
                   if r[0][1] in scope.owners and r[0][2] in scope.slots]
        if scope.mode in ('ordinary', 'event', 'historical') or not records:
            return ('', ''), False, [], False
        alternatives = []
        for (sid, owner, slot, begin, finish, status, contexts), kind in records:
            reasons = []; known = not scope.contexts or bool(contexts)
            if scope.contexts and contexts and not set(scope.contexts).issubset(contexts): reasons.append('different_context')
            if scope.mode == 'as_of' or (scope.mode == 'current' and scope.period_start):
                known = known and bool(begin)
                if begin and begin >= scope.period_end: reasons.append('future_schema')
                if finish and finish <= scope.period_start: reasons.append('expired_schema')
                if self._observed_times[index] and self._observed_times[index] >= scope.period_end: reasons.append('future_observation')
            elif scope.mode == 'current' and status == 'historical': reasons.append('historical_schema')
            if kind == 'exception': reasons.append('unresolved_exception')
            alternatives.append(((owner, slot), True, reasons, kind == 'support' and not reasons and known))
        return min(alternatives, key=lambda r: (len(r[2]), r[2], r[0]))

    def select_scored(self, query, candidates, scores, top_k=8, question_date=''):
        # Reuse the audited selector, supplying scope through a private hook is
        # avoided: its global query parser would ignore dynamic observed slots.
        from dataclasses import asdict
        from mindset.methods.base import MemoryHit
        scores = np.asarray(scores, dtype=float)
        if len(scores) != len(candidates) or not np.isfinite(scores).all() or len(set(candidates)) != len(candidates):
            raise ValueError('Invalid aligned scores/candidates')
        if top_k < 0: raise ValueError('Negative evidence budget')
        scope = self._scope(query, question_date)
        assessments = {i: self._assessment(i, scope) for i in candidates} if self.state_readout else {}
        witnesses = {r[0] for r in assessments.values() if r[1] and r[3]}
        excluded = {i: r[2] for i, r in assessments.items() if r[1] and r[2] and r[0] in witnesses}
        pos = {i: j for j, i in enumerate(candidates)}
        order = sorted((i for i in candidates if i not in excluded), key=lambda i: (-float(scores[pos[i]]), self._episode_order[i]))[:top_k]
        self.last_readout_certificate = {'scope': asdict(scope), 'identity_binding': self._last_scope_binding,
            'state_readout': self.state_readout, 'candidate_count': len(candidates), 'eligible_count': len(candidates) - len(excluded),
            'grounded_candidates': sum(r[1] for r in assessments.values()),
            'excluded': {self._episode_order[i]: r for i, r in excluded.items()},
            'selected_ids': [self._episode_order[i] for i in order], 'new_fitted_read_scalars': 0,
            'routing': self.last_route_certificate, 'packet_relevance_energy': float(sum(1/(1+np.exp(np.clip(scores[pos[i]], -60, 60))) for i in order))}
        return [MemoryHit(self._episode_order[i], self._documents[i], float(scores[pos[i]]), (self._episode_order[i],),
                          'aligned_raw_state_source', {'retrieval_version': self.retrieval_version}) for i in order]

    def candidate_pool(self, query, question_date=''):
        candidates, memberships = super().candidate_pool(query)
        scope = self._scope(query, question_date)
        anchors = []
        if scope.owners and scope.slots and scope.mode in ('current', 'as_of'):
            for owner in scope.owners:
                for slot in scope.slots:
                    for sid in self._slot_schemas.get((owner, slot), ()):
                        s = self.schemas[sid]
                        contexts = set(s.context_counts)
                        if scope.contexts and contexts and not set(scope.contexts).issubset(contexts): continue
                        begin, finish = self._bounds[sid]
                        dated = scope.mode == 'as_of' or bool(scope.period_start)
                        if not dated and scope.mode == 'current' and s.status != 'active': continue
                        if dated and ((begin and begin >= scope.period_end) or (finish and finish <= scope.period_start)): continue
                        anchors.append(self._index_by_id[s.medoid_episode_id])
        anchors = list(dict.fromkeys(anchors))
        # A small bounded witness reserve; preserve the original direct prefix.
        reserve = min(self.config.nearest_schema_k, len(anchors), max(0, len(candidates) - min(self.config.direct_pool_k, len(candidates))))
        missing = [i for i in anchors if i not in candidates][:reserve]
        if missing: candidates = candidates[:-len(missing)] + missing
        self.last_candidate_pool = tuple(candidates)
        self.last_route_certificate = {'identity_binding': self._last_scope_binding, 'state_anchors': len(anchors),
                                       'injected_witnesses': [self._episode_order[i] for i in missing], 'candidate_budget': self.config.retrieval_candidate_k}
        return candidates, memberships

    def retrieve(self, query, top_k=8, question_date=''):
        if top_k <= 0 or not self._episode_order: return []
        start = perf_counter(); candidates, _ = self.candidate_pool(query, question_date)
        built = perf_counter()
        if self.local_reranker is None: raise RuntimeError('Pinned cross-encoder required')
        scores = self.local_reranker.predict([(query, self._documents[i]) for i in candidates],
            batch_size=self.config.rerank_batch_size, show_progress_bar=False)
        scored = perf_counter(); hits = self.select_scored(query, candidates, scores, top_k, question_date)
        end = perf_counter(); self._query_count += 1; self._query_seconds += end - start
        self._retrieval_profile['candidate_build_seconds'] += built - start
        self._retrieval_profile['cross_encoder_seconds'] += scored - built
        self._retrieval_profile['state_selection_seconds'] += end - scored
        self._retrieval_profile['query_episode_pairs'] += len(candidates)
        return hits

    def _rebuild_retrieval_index(self):
        # Compiled static facts: no repeated schema/episode date parsing per query.
        super()._rebuild_retrieval_index()
        self._compiled_states = defaultdict(list); self._slot_schemas = defaultdict(list); self._slot_words = {}
        self._observed_times = [utc_time(self.episodes[eid].timestamp) for eid in self._episode_order]
        self._bounds = {}
        for s in self.schemas.values():
            owner, slot = s.topic.split('::', 1)
            self._slot_schemas[(owner, slot)].append(s.schema_id)
            label = slot.removeprefix('favorite_').removeprefix('relationship_').replace('_', ' ')
            self._slot_words[slot] = re.compile(r'\b' + re.escape(label) + r'(?:s|es)?\b', re.I)
            begin, finish = utc_time(s.valid_from), utc_time(s.valid_until)
            self._bounds[s.schema_id] = (begin, finish)
            state = (s.schema_id, owner, slot, begin, finish, s.status, frozenset(s.context_counts))
            for eid in s.support_episode_ids: self._compiled_states[eid].append((state, 'support'))
            for eid in s.exception_episode_ids: self._compiled_states[eid].append((state, 'exception'))

    def telemetry(self):
        return {**super().telemetry(), 'write_policy': self.write_policy,
                'readout_optimization': 'compiled_validity_and_provenance; state_off_skips_assessment',
                'parser': 'bounded_assertive_self_report_aliases_not_learned_semantics',
                'scalar_policy': 'unchanged_independent_joint_scalars_transferred_without_retuning'}
