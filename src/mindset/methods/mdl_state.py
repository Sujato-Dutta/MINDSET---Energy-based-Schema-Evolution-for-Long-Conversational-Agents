"""Isolated v2 candidate: finite-sample MDL energy for temporal transitions.

An exact LOCAL comparison of preserve versus every admissible single cut of
the active facet, not globally optimal segmentation of an entire history.
Jeffreys categorical coding and boundary-description costs are standard ideas,
not claimed as newly invented mathematics. No benchmark fitting is performed.
"""
from collections import Counter
from dataclasses import replace
from math import lgamma, log
import numpy as np
from mindset.methods.factorized_state import FactorizedMindsetMemory
from mindset.methods.state_causal_probe import ObservedStateMemory, TextProjectionMixin
from mindset.methods.joint import utc_time
from mindset.memory.energy import MULTI_VALUED_SLOTS, unit_vector


def categorical_code(counts, alphabet_size):
    """Negative log integrated categorical likelihood, Dirichlet(1/2,...).

All compared segments use the SAME observed-prefix alphabet. Alphabet growth
is observed-data adaptation, not lookahead or a calibrated probability of truth.
"""
    counts = tuple(counts)
    if alphabet_size < 1 or len(counts) > alphabet_size or any(c < 0 or int(c) != c for c in counts):
        raise ValueError('Invalid categorical counts/alphabet')
    n = sum(counts)
    return max(0., lgamma(n + alphabet_size / 2) - lgamma(alphabet_size / 2)
               - sum(lgamma(c + .5) - lgamma(.5) for c in counts))


def mode_value(counts, preferred=None):
    if not counts: raise ValueError('Empty state counts')
    maximum = max(counts.values())
    tied = sorted(v for v, c in counts.items() if c == maximum)
    return preferred if preferred in tied else tied[0]


def transition_candidates(values, *, temporal=True, boundary=True, likelihood=True, preferred=None):
    """Enumerate all feasible cuts. Costs are nats, not cosine losses.

One bit per segment codes a geometric prior over segment count. Each cut
position costs log(n-1), an explicit fixed-length boundary description.
This code penalizes transient fragmentation without a tuned mixing weight.
"""
    if not values: raise ValueError('At least one observation is required')
    n = len(values); alphabet = len(set(values)); total = Counter(values)
    data = categorical_code(total.values(), alphabet) if likelihood else 0.
    preserve = {'action': 'reinforce', 'mutation': 'preserve_distribution', 'cut': None,
        'terms': {'likelihood_code_nats': data, 'segment_code_nats': log(2),
                  'boundary_code_nats': 0., 'total': data + log(2)}, 'adjusted_energy': data + log(2)}
    plans = [preserve]
    if temporal and alphabet > 1:
        left = Counter(); right = total.copy()
        for cut in range(1, n):
            value = values[cut - 1]; left[value] += 1; right[value] -= 1
            if not right[value]: del right[value]
            if mode_value(left, preferred) == mode_value(right, preferred): continue
            data = (categorical_code(left.values(), alphabet) + categorical_code(right.values(), alphabet)) if likelihood else 0.
            price = log(n - 1) if boundary else 0.
            cost = data + 2 * log(2) + price
            plans.append({'action': 'supersede', 'mutation': 'version', 'cut': cut,
                'terms': {'likelihood_code_nats': data, 'segment_code_nats': 2 * log(2),
                          'boundary_code_nats': price, 'total': cost}, 'adjusted_energy': cost})
    return plans


def choose_transition(plans):
    return min(plans, key=lambda p: (p['adjusted_energy'], p['action'] != 'reinforce', -(p['cut'] or 0)))


class MDLEnergyMemory(FactorizedMindsetMemory):
    retrieval_version = 'categorical-mdl-active-facet-v2-development'

    def __init__(self, *args, boundary_energy=True, likelihood_energy=True, mdl_policy='energy', **kwargs):
        if mdl_policy not in ('energy', 'two_reports'): raise ValueError('Unknown MDL control')
        self.boundary_energy = boundary_energy; self.likelihood_energy = likelihood_energy
        self.mdl_policy = mdl_policy; self._mdl_facets = {}; self._mdl_posteriors = {}
        super().__init__(*args, **kwargs)

    @staticmethod
    def _votes(items):
        # Multiple turns in one session do not become repeated independent votes.
        # Last asserted value in a session is a declared modeling assumption.
        last = {e.session_id: e for e in items}
        return [e for e in items if last[e.session_id].episode_id == e.episode_id]

    def _fit(self, schema, items, alphabet, preferred=None):
        votes = self._votes(items)
        counts = Counter(e.features.attributes[0][1] for e in votes)
        mode = mode_value(counts, preferred)
        support = [e for e in items if e.features.attributes[0][1] == mode]
        schema.support_episode_ids = [e.episode_id for e in support]
        schema.exception_episode_ids = [e.episode_id for e in items if e.features.attributes[0][1] != mode]
        schema.entity_counts.clear(); schema.role_counts.clear(); schema.context_counts.clear(); schema.attribute_counts.clear()
        for e in support: schema._add_features(e.features)
        total = sum((e.vector for e in support), np.zeros_like(support[0].vector))
        self._sums[schema.schema_id] = total; schema.centroid = unit_vector(total)
        schema.mean_distortion = float(np.clip((1 - np.linalg.norm(total) / len(support)) / 2, 0, 1))
        schema.medoid_episode_id = support[0].episode_id
        self._latest[schema.schema_id] = max(utc_time(e.timestamp) for e in support)
        denominator = len(votes) + alphabet / 2
        self._mdl_posteriors[schema.schema_id] = {'mode': mode, 'session_counts': dict(counts),
            'observed_values': sorted(counts), 'alphabet_size': alphabet, 'sessions': len(votes),
            'posterior_mass': {v: (c + .5) / denominator for v, c in counts.items()},
            'meaning': 'categorical observed-report distribution, NOT truth confidence'}
        return mode

    def _apply_observation(self, episode):
        f = episode.features; slot = f.primary_slot
        if not f.attributes or slot in MULTI_VALUED_SLOTS:
            return super()._apply_observation(episode)
        if len(f.attributes) != 1: raise ValueError('MDL expects atomic facets')
        topic = f'{episode.speaker.casefold()}::{slot}'
        contexts = tuple(sorted(f.contexts)); facet = (topic, contexts)
        key = (episode.episode_id, topic, contexts)
        if key in self._observation_certificates: raise ValueError('Duplicate MDL observation')
        self._observation_counts[episode.episode_id] += 1
        if facet not in self._mdl_facets:
            related = [self.schemas[sid] for (name, _), (sid, _) in self._mdl_facets.items() if name == topic]
            target = min(related, key=lambda s: s.schema_id) if related else None
            action = 'split' if target else 'new'
            schema = self._create_schema(episode, action=action)
            self._mdl_facets[facet] = (schema.schema_id, [episode])
            if target:
                self._split_links[target.schema_id].add(schema.schema_id)
                self._split_links[schema.schema_id].add(target.schema_id)
            self._fit(schema, [episode], 1)
            self._record_transition(episode, action, target.schema_id if target else None,
                (target.schema_id, schema.schema_id) if target else (schema.schema_id,), {}, None)
            self._observation_certificates[key] = {'policy': self.mdl_policy, 'action': action,
                'topic': topic, 'source_id': episode.episode_id, 'feasible_candidates': [],
                'energy_definition': 'Jeffreys categorical code + segment/boundary description length'}
            return
        sid, previous = self._mdl_facets[facet]; old = self.schemas[sid]; items = previous + [episode]
        votes = self._votes(items); values = [e.features.attributes[0][1] for e in votes]
        alphabet = len(set(values)); preferred = self._mdl_posteriors[sid]['mode']
        stamps = [utc_time(e.timestamp) for e in items]
        temporal = bool(all(stamps) and stamps == sorted(stamps))
        plans = transition_candidates(values, temporal=temporal, boundary=self.boundary_energy,
                                      likelihood=self.likelihood_energy, preferred=preferred)
        # Also reject cuts that would give old sources a future version boundary
        # after their successor. Session-deduplication must not reorder time.
        feasible = []
        for p in plans:
            if p['cut'] is not None:
                index = items.index(votes[p['cut']])
                if not index or utc_time(items[index - 1].timestamp) >= utc_time(items[index].timestamp): continue
                if {e.session_id for e in items[:index]} & {e.session_id for e in items[index:]}: continue
            feasible.append(p)
        chosen = choose_transition(feasible)
        if self.mdl_policy == 'two_reports':
            # Strong confirmation rule with the same retrospective cut/provenance.
            chosen = feasible[0]
            if len(values) >= 2 and values[-1] == values[-2] and values[-1] != preferred:
                target_cut = len(values) - 2
                matching = [p for p in feasible if p['cut'] == target_cut]
                if matching: chosen = matching[0]
        action = chosen['action']; cut = chosen['cut']
        if action == 'supersede':
            position = items.index(votes[cut]); left, right = items[:position], items[position:]
            old.status = 'historical'; old.valid_until = right[0].timestamp
            self._fit(old, left, alphabet, preferred)
            new = self._create_schema(right[0], action='supersede', predecessor_schema_id=sid)
            old.successor_schema_ids.append(new.schema_id)
            self._fit(new, right, alphabet)
            self._mdl_facets[facet] = (new.schema_id, right)
            resulting = (sid, new.schema_id); exception = episode.episode_id in new.exception_episode_ids
        else:
            self._fit(old, items, alphabet, preferred)
            self._mdl_facets[facet] = (sid, items); resulting = (sid,)
            exception = episode.episode_id in old.exception_episode_ids
        self._record_transition(episode, action, sid, resulting, {}, None, stored_as_exception=exception)
        audit = [{**p, 'schema_id': sid} for p in feasible]
        self._observation_certificates[key] = {'policy': self.mdl_policy, 'action': action,
            'topic': topic, 'source_id': episode.episode_id, 'feasible_candidates': audit,
            'selected_adjusted_energy': chosen['adjusted_energy'],
            'optimality_gap': chosen['adjusted_energy'] - min(p['adjusted_energy'] for p in feasible),
            'chosen_cut_session': cut, 'decision_time': episode.timestamp,
            'inferred_valid_from': self.schemas[resulting[-1]].valid_from,
            'effective_session_observations': len(votes), 'temporal_order_known': temporal,
            'energy_definition': 'nats; exact active-facet one-cut comparison, NOT global history optimum',
            'boundary_energy': self.boundary_energy, 'likelihood_energy': self.likelihood_energy}

    def telemetry(self):
        return {**super().telemetry(), 'mdl_policy': self.mdl_policy, 'boundary_energy': self.boundary_energy,
            'likelihood_energy': self.likelihood_energy, 'write_energy': 'finite_sample_categorical_MDL',
            'decision_scalars_used': [], 'structural_defaults': 'Dirichlet Jeffreys 1/2; geometric segment prior 1/2',
            'scalar_scope': 'MDL categorical facets only; unsupported/multi-valued facets retain the legacy fallback',
            'global_history_optimality_claim': False, 'no_benchmark_parameter_fit': True,
            'limitations': 'bounded extraction; categorical cardinality; session reliability; committed older boundaries'}


class ObservedMDLEnergyMemory(MDLEnergyMemory, ObservedStateMemory): pass
class TextMDLEnergyMemory(TextProjectionMixin, ObservedMDLEnergyMemory): pass
