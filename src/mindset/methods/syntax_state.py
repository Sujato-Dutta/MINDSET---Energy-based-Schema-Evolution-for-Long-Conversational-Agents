"""Development syntax-grounded state readout over immutable source episodes."""
from collections import defaultdict
from dataclasses import replace
import re

from mindset.methods.factorized_state import FactorizedMindsetMemory, atomic_features
from mindset.methods.aligned_readout import observable_period
from mindset.memory.energy import EpisodeFeatures, tokenize
from mindset.methods.joint import utc_time


def frame_features(speaker, text, frames):
    grouped = defaultdict(list)
    for frame in frames:
        start, end = frame['value_span']
        if not (0 <= start < end <= len(text)) or text[start:end] != frame['value']:
            raise ValueError('Syntax value is not grounded in the exact raw source')
        if not frame['slot'].startswith('syntax_') or not frame['match_terms']:
            raise ValueError('Invalid source-derived syntax predicate')
        key = (frame['slot'], tuple(sorted(frame['contexts'])))
        grouped[key].append(frame)
    features = []
    for (slot, contexts), rows in grouped.items():
        if len({f['value'].casefold() for f in rows}) != 1: continue
        features.append(EpisodeFeatures((speaker.casefold(),), (), contexts,
            ((slot, rows[0]['value'].casefold()),), tokenize(text),
            any(f['explicit_change'] for f in rows)))
    return features


class SyntaxMindsetMemory(FactorizedMindsetMemory):
    retrieval_version = 'syntax-grounded-source-state-v2-development'

    def __init__(self, *args, source_projections, query_projections, syntax_readout=True, **kwargs):
        self.source_projections = source_projections
        self.query_projections = query_projections
        self.syntax_readout = syntax_readout
        self._syntax_terms = {}
        self._syntax_sources = 0
        super().__init__(*args, **kwargs)

    def ingest(self, events):
        # Validate the complete batch before the inherited raw-source mutation.
        for event in events:
            row = self.source_projections.get(event.event_id)
            if row is None or row['text'] != event.text or row['speaker'] != event.speaker:
                raise ValueError('Missing or stale source projection')
            if row.get('timestamp', event.timestamp) != event.timestamp:
                raise ValueError('Stale source projection timestamp')
            frame_features(event.speaker, event.text, row['frames'])
        return super().ingest(events)

    def _ingest_episode(self, episode):
        projected = self.source_projections.get(episode.episode_id)
        if projected is None or projected['text'] != episode.text or projected['speaker'] != episode.speaker:
            raise ValueError('Missing or stale source projection')
        observations = atomic_features(episode.speaker, episode.text)
        extra = frame_features(episode.speaker, episode.text, projected['frames'])
        if extra: self._syntax_sources += 1
        for frame in projected['frames']:
            slot, terms = frame['slot'], tuple(frame['match_terms'])
            if slot in self._syntax_terms and self._syntax_terms[slot] != terms:
                raise ValueError('Ambiguous syntax predicate encoding')
            self._syntax_terms[slot] = terms
        observations.extend(extra)
        if not observations:
            super()._ingest_episode(episode); return
        seen = set()
        for features in observations:
            key = (features.primary_slot, features.contexts)
            if key in seen: raise ValueError('Duplicate projected owner/predicate/context')
            seen.add(key); self._apply_observation(replace(episode, features=features))
        attributes = tuple(dict.fromkeys(a for f in observations for a in f.attributes))
        self.episodes[episode.episode_id] = replace(episode, features=replace(episode.features, attributes=attributes))

    def _scope(self, query, question_date=''):
        scope = super()._scope(query, question_date)
        if not self.syntax_readout: return scope
        projected = self.query_projections.get(query)
        if projected is None: raise ValueError('Missing frozen query projection')
        terms = set(projected['terms'])
        matched = {slot for slot, label in self._syntax_terms.items() if set(label).issubset(terms)}
        owners = scope.owners
        roles = {s.casefold() for s in self._speakers}
        if not owners and 'user' in roles and roles <= {'user', 'assistant', 'system', 'tool', 'unknown'}:
            if projected['direct_self'] or projected['possessive_self']:
                owners = ('user',)
                self._last_scope_binding = 'dependency_self_bound_to_literal_user_role'
        slots = tuple(sorted(set(scope.slots) | matched))
        mode, start, end = scope.mode, scope.period_start, scope.period_end
        transition = bool(terms & {'change', 'increase', 'decrease', 'reduce', 'switch', 'update', 'difference'})
        latest = bool(re.search(r'\b(?:latest|most recent|most recently)\b', query, re.I))
        if slots and owners and transition: mode = 'transition'
        elif slots and owners and latest and mode != 'event': mode = 'current'
        if mode == 'ordinary' and slots and projected['present_tense']: mode = 'current'
        # Absolute-time point state remains distinct from change/event questions.
        qs, qe = observable_period(query)
        if slots and qs and mode not in ('event', 'transition'): mode, start, end = 'as_of', qs, qe
        if mode == 'current' and question_date:
            ds, de = observable_period(question_date)
            if ds: start, end = ds, de
        return replace(scope, owners=owners, slots=slots, mode=mode, period_start=start, period_end=end,
                       contexts=tuple(projected.get('contexts') or scope.contexts))

    def _assessment(self, index, scope):
        if scope.mode == 'transition': return ('', ''), False, [], False
        return super()._assessment(index, scope)

    def _transition_pair(self, scope, question_date=''):
        if scope.mode != 'transition': return ()
        _, limit = observable_period(question_date) if question_date else (0., 0.)
        if scope.period_end: limit = scope.period_end
        pairs = []
        for owner in scope.owners:
            for slot in scope.slots:
                for sid in self._slot_schemas.get((owner, slot), ()):
                    new = self.schemas[sid]
                    if not new.predecessor_schema_id: continue
                    time = utc_time(new.valid_from)
                    if not time or (limit and time >= limit): continue
                    if scope.period_start and time < scope.period_start: continue
                    if scope.contexts and not set(scope.contexts).issubset(new.context_counts): continue
                    old = self.schemas[new.predecessor_schema_id]
                    pairs.append((time, sid, old, new))
        if not pairs: return ()
        _, _, old, new = max(pairs, key=lambda p: (p[0], p[1]))
        # Actual version-edge witness: source at creation of each schema, not a
        # fabricated textual summary or an arbitrary recent neighboring turn.
        return (old.support_episode_ids[0], new.support_episode_ids[0])

    def candidate_pool(self, query, question_date=''):
        candidates, memberships = super().candidate_pool(query, question_date)
        pair = self._transition_pair(self._scope(query, question_date), question_date)
        missing = [self._index_by_id[eid] for eid in pair if self._index_by_id[eid] not in candidates]
        reserve = max(0, len(candidates) - min(self.config.direct_pool_k, len(candidates)))
        missing = missing[:reserve]
        if missing: candidates = candidates[:-len(missing)] + missing
        self.last_candidate_pool = tuple(candidates)
        self.last_route_certificate['transition_witnesses'] = list(pair)
        self.last_route_certificate['injected_transition_witnesses'] = [self._episode_order[i] for i in missing]
        return candidates, memberships

    def select_scored(self, query, candidates, scores, top_k=8, question_date=''):
        hits = super().select_scored(query, candidates, scores, top_k, question_date)
        pair = self._transition_pair(self._scope(query, question_date), question_date)
        from mindset.methods.base import MemoryHit
        position = {self._episode_order[i]: j for j, i in enumerate(candidates)}
        complete_pair = len(pair) == 2 and all(eid in position for eid in pair)
        if self.state_readout and complete_pair and top_k >= 2:
            chosen = list(pair) + [h.memory_id for h in hits if h.memory_id not in pair]
            chosen = chosen[:top_k]
            chosen.sort(key=lambda eid: (-float(scores[position[eid]]), eid))
            hits = [MemoryHit(eid, self._documents[self._index_by_id[eid]], float(scores[position[eid]]),
                (eid,), 'syntax_raw_transition_witness', {'retrieval_version': self.retrieval_version}) for eid in chosen]
            self.last_readout_certificate['selected_ids'] = chosen
            self.last_readout_certificate['packet_relevance_energy'] = None
        self.last_readout_certificate['transition_pair'] = list(pair)
        self.last_readout_certificate['transition_pair_reserved'] = bool(self.state_readout and complete_pair and top_k >= 2)
        return hits

    def telemetry(self):
        return {**super().telemetry(), 'syntax_readout': self.syntax_readout,
                'syntax_observed_predicates': len(self._syntax_terms), 'syntax_sources': self._syntax_sources,
                'parser_policy': 'frozen pretrained dependency parser + fixed extractive projections; not NLI',
                'calibration_status': 'old scalars transferred, new syntax representation NOT independently recalibrated'}
