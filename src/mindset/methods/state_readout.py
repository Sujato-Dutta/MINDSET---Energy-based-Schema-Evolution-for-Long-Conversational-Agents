"""Experimental state-to-evidence coupling; no benchmark-fitted read weights.

Immutable sources remain accessible. Explicitly scoped state questions can
exclude incompatible sources *from one packet*, never from memory. Unknown
dates, owners, predicates and contexts cause abstention from state filtering.
The write controller and its independent calibration are deliberately unchanged.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import calendar
import re
from time import perf_counter

import numpy as np

from mindset.methods.base import MemoryHit
from mindset.methods.joint import JointMindsetMemory, typed_features, utc_time


@dataclass(frozen=True)
class QueryScope:
    mode: str
    owners: tuple[str, ...]
    slots: tuple[str, ...]
    contexts: tuple[str, ...]
    period_start: float = 0.
    period_end: float = 0.


def _stamp(year, month=1, day=1):
    return datetime(year, month, day, tzinfo=timezone.utc).timestamp()


def explicit_period(query):
    """Unambiguous absolute day/month/year only; no wall-clock date inference."""
    match = re.search(r'\b(\d{4})-(\d{2})-(\d{2})\b', query)
    try:
        if match:
            y, m, d = map(int, match.groups())
            start = _stamp(y, m, d)
            return start, start + 86400.
        names = '|'.join(calendar.month_name[1:])
        match = re.search(rf'\b({names})\s+(?:(\d{{1,2}}),?\s+)?(\d{{4}})\b', query, re.I)
        if match:
            month = next(i for i, n in enumerate(calendar.month_name) if n.casefold() == match[1].casefold())
            year = int(match[3])
            if match[2]:
                start = _stamp(year, month, int(match[2]))
                return start, start + 86400.
            return _stamp(year, month), _stamp(year + (month == 12), month % 12 + 1)
        # Multiple years often describe an interval/comparison, not one state.
        years = re.findall(r'\b(?:19|20)\d{2}\b', query)
        if len(years) == 1:
            year = int(years[0])
            return _stamp(year), _stamp(year + 1)
    except (ValueError, OverflowError):
        pass
    return 0., 0.


def query_scope(query, speakers, question_date=''):
    owners = tuple(sorted(s.casefold() for s in speakers
                          if re.search(r'(?<!\w)' + re.escape(s) + r'(?!\w)', query, re.I)))
    slots = []
    for slot, pattern in (
        ('location', r'\b(?:live|lives|lived|living|reside|resides|based|location)\b'),
        ('employer', r'\b(?:employer|company|work for|works for|worked for|work at|works at|worked at)\b'),
        ('occupation', r'\b(?:occupation|profession|job|work as|works as|worked as)\b'),
        ('preference', r'\b(?:prefer|prefers|preferred)\b'),
    ):
        if re.search(pattern, query, re.I): slots.append(slot)
    favorite = re.search(r'\bfavou?rite\s+([a-z]+)\b', query, re.I)
    if favorite: slots.append('favorite_' + favorite[1].casefold())
    contexts = typed_features('', query).contexts
    start, end = explicit_period(query)
    # Event/multi-state questions must retain both versions, even if they say now.
    event = bool(re.search(r'\b(?:when|how long|since when|before and after|changed|change|changes|moved|move|timeline|previous and current)\b', query, re.I))
    past = bool(re.search(r'\b(?:formerly|previously|used to|historical|did|was|were|lived|worked|preferred)\b', query, re.I))
    current = bool(re.search(r'\b(?:now|currently|current|still|today|latest|does|is|are)\b', query, re.I))
    mode = 'event' if event else 'as_of' if start and slots else 'historical' if past else 'current' if current and slots else 'ordinary'
    if mode == 'current' and question_date:
        # The evaluation supplies this observable date, not a gold category.
        date_start, date_end = explicit_period(question_date)
        if date_start: start, end = date_start, date_end
    return QueryScope(mode, owners, tuple(sorted(set(slots))), tuple(sorted(contexts)), start, end)


class StateReadoutMixin:
    """Constrained greedy packet energy, with auditable conservative gates.

E(A|q) = sum(1-sigmoid(CE(e,q))) over accessible sources. Accessibility
requires owner/predicate grounding and known context/validity compatibility.
Incompatible states are excluded only when a compatible support witness is
present in the same scored pool. This is a constrained optimization heuristic,
not a claim of globally optimal combinatorial inference or new physics.
"""
    retrieval_version = 'state-constrained-raw-provenance-v2-development'

    def __init__(self, *args, state_readout=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_readout = state_readout
        self.last_readout_certificate = {}

    def _rebuild_retrieval_index(self):
        super()._rebuild_retrieval_index()
        self._episode_state = defaultdict(list)
        self._speakers = sorted({e.speaker for e in self.episodes.values()})
        for schema in self.schemas.values():
            for eid in schema.support_episode_ids:
                self._episode_state[eid].append((schema.schema_id, 'support'))
            for eid in schema.exception_episode_ids:
                self._episode_state[eid].append((schema.schema_id, 'exception'))

    def _assessment(self, index, scope):
        episode = self.episodes[self._episode_order[index]]
        # No inferred pronoun identity or cross-speaker attribute transfer.
        relevant = bool(scope.owners and episode.speaker.casefold() in scope.owners
                        and scope.slots and episode.features.primary_slot in scope.slots
                        and scope.mode not in ('ordinary', 'event', 'historical'))
        key = (episode.speaker.casefold(), episode.features.primary_slot)
        if not relevant: return key, False, [], False
        alternatives = []
        for sid, kind in self._episode_state[episode.episode_id]:
            schema = self.schemas[sid]
            reasons = []
            known_scope = not scope.contexts or bool(schema.context_counts)
            if scope.contexts and schema.context_counts and not set(scope.contexts).issubset(schema.context_counts):
                reasons.append('different_context')
            if scope.mode == 'as_of' or (scope.mode == 'current' and scope.period_start):
                begin, finish = utc_time(schema.valid_from), utc_time(schema.valid_until)
                known_scope = known_scope and bool(begin)
                # Intervals overlap the requested DAY/MONTH/YEAR, not a fabricated
                # January-1 point. Unknown timestamps do not justify exclusion.
                if begin and begin >= scope.period_end: reasons.append('future_schema')
                if finish and finish <= scope.period_start: reasons.append('expired_schema')
                observed = utc_time(episode.timestamp)
                if observed and observed >= scope.period_end: reasons.append('future_observation')
            elif scope.mode == 'current' and schema.status == 'historical':
                reasons.append('historical_schema')
            if kind == 'exception': reasons.append('unresolved_exception')
            alternatives.append((reasons, kind == 'support' and not reasons and known_scope))
        if not alternatives: return key, True, [], False
        reasons, witness = min(alternatives, key=lambda row: (len(row[0]), row[0]))
        return key, True, reasons, witness

    def select_scored(self, query, candidates, scores, top_k=8, question_date=''):
        """Shared CE scores allowed for quality only; native retrieve scores anew."""
        scores = np.asarray(scores, dtype=float)
        if len(scores) != len(candidates) or not np.isfinite(scores).all():
            raise ValueError('Invalid state-readout CE scores')
        if len(set(candidates)) != len(candidates): raise ValueError('Duplicate candidates')
        if top_k < 0: raise ValueError('Negative evidence budget')
        scope = query_scope(query, self._speakers, question_date)
        assessments = {i: self._assessment(i, scope) for i in candidates}
        witnesses = {key for key, relevant, reasons, support in assessments.values() if relevant and support}
        excluded = {i: reasons for i, (key, relevant, reasons, _) in assessments.items()
                    if self.state_readout and relevant and reasons and key in witnesses}
        # Stable monotone relevance loss; CE logits are NOT calibrated probabilities.
        losses = 1. / (1. + np.exp(np.clip(scores, -60., 60.)))
        position = {i: j for j, i in enumerate(candidates)}
        feasible = [i for i in candidates if i not in excluded]
        order = sorted(feasible, key=lambda i: (-float(scores[position[i]]), self._episode_order[i]))[:top_k]
        self.last_readout_certificate = {
            'scope': asdict(scope), 'state_readout': self.state_readout,
            'candidate_count': len(candidates), 'eligible_count': len(feasible),
            'grounded_candidates': sum(row[1] for row in assessments.values()),
            'compatible_witness_keys': [list(k) for k in sorted(witnesses)],
            'excluded': {self._episode_order[i]: v for i, v in sorted(excluded.items())},
            'selected_ids': [self._episode_order[i] for i in order],
            'packet_relevance_energy': float(sum(losses[position[i]] for i in order)),
            'optimization': 'minimum additive relevance loss at maximal feasible cardinality <= top_k',
            'new_fitted_read_scalars': 0, 'fallback': 'unscoped_or_unwitnessed_sources_remain_accessible',
        }
        return [MemoryHit(self._episode_order[i], self._documents[i], float(scores[position[i]]),
                          (self._episode_order[i],), 'state_scoped_raw_episode',
                          {'retrieval_version': self.retrieval_version}) for i in order]

    def retrieve(self, query, top_k=8, question_date=''):
        if top_k <= 0 or not self._episode_order: return []
        start = perf_counter()
        candidates, _ = self.candidate_pool(query)
        built = perf_counter()
        if self.local_reranker is None: raise RuntimeError('Pinned cross-encoder required')
        scores = self.local_reranker.predict([(query, self._documents[i]) for i in candidates],
            batch_size=self.config.rerank_batch_size, show_progress_bar=False)
        scored = perf_counter()
        hits = self.select_scored(query, candidates, scores, top_k, question_date)
        end = perf_counter()
        self._query_seconds += end - start; self._query_count += 1
        self._retrieval_profile['candidate_build_seconds'] += built - start
        self._retrieval_profile['cross_encoder_seconds'] += scored - built
        self._retrieval_profile['state_selection_seconds'] += end - scored
        self._retrieval_profile['query_episode_pairs'] += len(candidates)
        return hits

    def telemetry(self):
        return {**super().telemetry(), 'state_readout': self.state_readout,
                'state_readout_version': self.retrieval_version, 'new_fitted_read_scalars': 0,
                'claim_status': 'experimental_state_coupling_not_paper_frozen'}


class StateAwareMindsetMemory(StateReadoutMixin, JointMindsetMemory):
    """Energy-selected write state now constrains scoped raw-evidence packets."""
