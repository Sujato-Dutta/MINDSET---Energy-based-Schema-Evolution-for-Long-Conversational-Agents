"""Controlled-input diagnostic adapters, NOT natural-language paper methods.

Only asserted source facts and requested query operators enter controllers.
Latent world states and noise labels belong to the evaluator, never this file.
"""
from dataclasses import replace
import numpy as np
from mindset.methods.factorized_state import FactorizedMindsetMemory, atomic_features
from mindset.methods.aligned_readout import AlignedMindsetMemory
from mindset.memory.energy import EpisodeFeatures, tokenize, unit_vector
from mindset.memory.energy import contradiction_probability
from mindset.methods.joint import utc_time


class ObservedStateMemory(FactorizedMindsetMemory):
    def __init__(self, *args, observations, scopes, **kwargs):
        self.observations = observations
        self.scopes = scopes
        super().__init__(*args, **kwargs)

    def _features(self, eid):
        o = self.observations[eid]
        return EpisodeFeatures((o['speaker'].casefold(),), (), tuple(o['contexts']),
                               ((o['slot'], o['value'].casefold()),), tokenize(o['text']), o['explicit_change'])

    def _ingest_episode(self, episode):
        o = self.observations[episode.episode_id]
        if o['text'] != episode.text or o['speaker'] != episode.speaker or o['value'] not in episode.text:
            raise ValueError('Controlled observation is not source-grounded')
        features = self._features(episode.episode_id)
        self._apply_observation(replace(episode, features=features))
        self.episodes[episode.episode_id] = replace(episode, features=features)

    def _scope(self, query, question_date=''):
        if query not in self.scopes: raise ValueError('Unknown controlled query operator')
        self._last_scope_binding = 'controlled_observable_query_operator_not_inferred_parser'
        return self.scopes[query]


class ConfirmationReplayMemory(ObservedStateMemory):
    """Candidate: distinguish confirmation time from inferred validity time.

Only matched pending observations after the LAST old-state support are replayed.
No world/noise label is used. Repeated misinformation remains a limitation.
"""
    def __init__(self, *args, contradiction_debt=True, **kwargs):
        self.contradiction_debt = contradiction_debt
        super().__init__(*args, **kwargs)

    def _pending_support(self, schema, episode):
        latest = self._latest[schema.schema_id]
        incoming_time = utc_time(episode.timestamp)
        if not latest or not incoming_time: return []
        return sorted((eid for eid in schema.exception_episode_ids
            if latest < utc_time(self.episodes[eid].timestamp) <= incoming_time
            and self._features(eid).attributes == episode.features.attributes
            and self._features(eid).contexts == episode.features.contexts),
            key=lambda eid: (utc_time(self.episodes[eid].timestamp), eid))

    def _signals(self, schema, episode):
        signals = super()._signals(schema, episode)
        pending = self._pending_support(schema, episode)
        recurrent = any(self.episodes[eid].session_id != episode.session_id for eid in pending)
        old, new = set(schema.context_counts), set(episode.features.contexts)
        different = bool(old and new and old != new)
        later = bool(utc_time(episode.timestamp) and self._latest[schema.schema_id]
                     and utc_time(episode.timestamp) >= self._latest[schema.schema_id])
        temporal = float(signals.contradiction_probability > 0 and not different and later
                         and (episode.features.explicit_change or recurrent))
        return replace(signals, recurrence=float(recurrent), temporal_potential=temporal)

    def _plans(self, schema, episode):
        plans = super()._plans(schema, episode)
        pending = self._pending_support(schema, episode)
        for p in plans:
            if p['mutation'] != 'version' or not p['signals'].recurrence or not pending: continue
            vectors = np.asarray([self.episodes[eid].vector for eid in pending] + [episode.vector])
            center = unit_vector(vectors.sum(axis=0))
            distortion = float(np.mean(np.clip((1 - vectors @ center) / 2, 0, 1)))
            terms = p['terms']
            # Marginal unresolved-incompatibility potential: a version resolves
            # matched deferred observations, while another exception does not.
            # This is a signed energy DELTA, not a negative probability.
            debt = sum(contradiction_probability(schema, self._features(eid)) for eid in pending) if self.contradiction_debt else 0.
            p['terms'] = replace(terms, distortion=distortion, contradiction=-debt,
                total=terms.total + distortion - terms.distortion - debt - terms.contradiction)
            p['cost'] = p['terms'].total + p['switch']
        return plans

    def _apply_observation(self, episode):
        # Compute eligible replay before mutation; target selection is unchanged.
        candidates = {sid: self._pending_support(self.schemas[sid], episode)
            for sid in self._by_topic[f'{episode.speaker.casefold()}::{episode.features.primary_slot}']
            if self.schemas[sid].status == 'active'}
        super()._apply_observation(episode)
        record = self.operation_log.all()[-1]
        if record.operation != 'supersede' or not record.signals.get('recurrence'): return
        pending = candidates.get(record.target_schema_id, [])
        if not pending: return
        old = self.schemas[record.target_schema_id]
        new = self.schemas[record.resulting_schema_ids[-1]]
        start = self.episodes[pending[0]].timestamp
        old.valid_until = start; new.valid_from = start
        for eid in pending:
            if eid not in new.support_episode_ids:
                new.support_episode_ids.append(eid)
                new._add_features(self._features(eid))
        total = sum((self.episodes[eid].vector for eid in new.support_episode_ids),
                    np.zeros_like(episode.vector))
        self._sums[new.schema_id] = total
        new.centroid = unit_vector(total)
        new.mean_distortion = float(np.clip((1 - np.linalg.norm(total) / len(new.support_episode_ids)) / 2, 0, 1))
        key = (episode.episode_id, new.topic, tuple(sorted(episode.features.contexts)))
        self._observation_certificates[key]['confirmation_replay'] = {
            'support_ids': pending, 'decision_time': episode.timestamp, 'inferred_valid_from': start,
            'policy': 'matched_pending_after_last_old_support; no latent labels'}


class TextProjectionMixin:
    """Same control logic, but actual existing text extraction/query parsing."""
    def _features(self, eid):
        o = self.observations[eid]
        parsed = atomic_features(o['speaker'], o['text'])
        if len(parsed) != 1: raise ValueError('Constructed text projection is ambiguous')
        return parsed[0]

    def _ingest_episode(self, episode):
        FactorizedMindsetMemory._ingest_episode(self, episode)

    def _scope(self, query, question_date=''):
        return AlignedMindsetMemory._scope(self, query, question_date)


class LatestJournalMemory(ObservedStateMemory):
    """Strong simple control: latest matching-context assertion, with history."""
    def __init__(self, *args, **kwargs):
        kwargs['write_policy'] = 'nearest'
        super().__init__(*args, **kwargs)

    def _assessment(self, index, scope):
        eid = self._episode_order[index]; e = self.episodes[eid]
        f = e.features; key = (e.speaker.casefold(), f.primary_slot)
        if key[0] not in scope.owners or key[1] not in scope.slots: return key, False, [], False
        if scope.period_end and utc_time(e.timestamp) >= scope.period_end:
            return key, True, ['future_observation'], False
        eligible = []
        for other in self.episodes.values():
            of = other.features
            if other.speaker.casefold() != key[0] or of.primary_slot != key[1]: continue
            if scope.contexts and not set(scope.contexts).issubset(of.contexts): continue
            stamp = utc_time(other.timestamp)
            if not stamp or (scope.period_end and stamp >= scope.period_end): continue
            eligible.append(other)
        if not eligible: return key, True, [], False
        latest = max(eligible, key=lambda r: (utc_time(r.timestamp), r.episode_id))
        if e.episode_id == latest.episode_id: return key, True, [], True
        # Preserve equally valued older evidence; journal is not latest-text-only.
        if f.attributes == latest.features.attributes and f.contexts == latest.features.contexts:
            return key, True, [], True
        return key, True, ['not_latest_matching_context_state'], False


class TextStateMemory(TextProjectionMixin, ObservedStateMemory): pass
class TextReplayMemory(TextProjectionMixin, ConfirmationReplayMemory): pass
class TextLatestJournalMemory(TextProjectionMixin, LatestJournalMemory): pass
