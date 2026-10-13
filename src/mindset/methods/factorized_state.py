"""Development-only source-backed, attribute-factorized state controller.

The parser is still bounded English, not learned comprehensive semantics.
Each accepted owner/predicate/context observation gets its own transition, but
all observations retain the SAME immutable raw source ID and episode vector.
The two existing scalars are transferred, not recalibrated for this design.
"""
from collections import Counter, defaultdict
from dataclasses import replace
import math
import re
import numpy as np

from mindset.methods.aligned_readout import AlignedMindsetMemory, self_report_view
from mindset.methods.joint import typed_features, utc_time
from mindset.memory.energy import unit_vector


def atomic_features(speaker, text):
    """Reuse the existing parser clause-wise; do not invent a predicate lexicon.

Multiple values for one predicate/context in a single episode are ambiguous:
they are omitted from the state layer, not resolved by clause order. Sources
remain retrievable. Context is attached to its own clause, never globally.
"""
    grouped = defaultdict(list)
    for clause in re.split(r'[.!?;\n]+|\band\s+(?=I\b|my\b)', text, flags=re.I):
        view = self_report_view(clause)
        if not view: continue
        features = typed_features(speaker, view)
        for slot, value in features.attributes:
            key = (slot, tuple(sorted(features.contexts)))
            grouped[key].append((value, features))
    result = []
    for (slot, _), rows in sorted(grouped.items()):
        if len({value for value, _ in rows}) != 1: continue
        value, features = rows[0]
        result.append(replace(features, attributes=((slot, value),),
                              explicit_change=any(f.explicit_change for _, f in rows)))
    return result


class FactorizedMindsetMemory(AlignedMindsetMemory):
    retrieval_version = 'factorized-source-state-v2-development'

    def __init__(self, *args, atomic=True, local_complexity=True, **kwargs):
        self.atomic = atomic
        self.local_complexity = local_complexity
        self._observation_counts = Counter()
        self._observation_certificates = {}
        super().__init__(*args, **kwargs)

    def _plans(self, schema, episode):
        # Preserve actual-state distortion/damage and all feasibility checks.
        plans = super()._plans(schema, episode)
        if self.config.ablation == 'no_complexity': price = 0.
        elif self.local_complexity:
            # Model size in the SAME owner/predicate block, including versions.
            # Unrelated observations cannot change this local description cost.
            price = self.config.complexity_penalty * math.log1p(len(self._by_topic[schema.topic]))
        else: price = self.config.complexity_penalty * math.log1p(len(self.episodes))
        for plan in plans:
            old = plan['terms']
            terms = replace(old, total=old.distortion + old.contradiction + old.historical_damage
                            + price * old.fragmentation + old.inconsistency)
            plan['terms'] = terms
            plan['cost'] = terms.total + plan['switch']
            plan['complexity_price'] = price
        return plans

    def _ingest_episode(self, episode):
        observations = atomic_features(episode.speaker, episode.text) if self.atomic else []
        if not self.atomic:
            features = typed_features(episode.speaker, self_report_view(episode.text))
            observations = [features]
        if not observations:
            observations = [replace(typed_features(episode.speaker, ''), attributes=())]
        for features in observations:
            incoming = replace(episode, features=features)
            self._apply_observation(incoming)
        # Raw text/vector/ID never replaced; combined features are audit-only.
        attributes = tuple(dict.fromkeys(a for f in observations for a in f.attributes))
        self.episodes[episode.episode_id] = replace(episode, features=replace(
            episode.features, attributes=attributes))

    def _apply_observation(self, episode):
        topic = f'{episode.speaker.casefold()}::{episode.features.primary_slot}'
        key = (episode.episode_id, topic, tuple(sorted(episode.features.contexts)))
        if key in self._observation_certificates: raise ValueError('Duplicate atomic observation')
        pool = [self.schemas[sid] for sid in self._by_topic[topic]
                if self.schemas[sid].status == 'active']
        self._observation_counts[episode.episode_id] += 1
        if not pool:
            schema = self._create_schema(episode, action='new')
            self._record_transition(episode, 'new', None, (schema.schema_id,), {}, None)
            self._observation_certificates[key] = {'policy': self.write_policy, 'action': 'new',
                'topic': topic, 'source_id': episode.episode_id, 'feasible_candidates': []}
            return
        priority = {'reinforce': 0, 'supersede': 1, 'split': 2, 'new': 3}
        nearest = sorted(pool, key=lambda s: (-float(s.centroid @ episode.vector), s.schema_id))[:self.config.nearest_schema_k]
        plans = [p for s in nearest for p in self._plans(s, episode)]
        if self.write_policy == 'nearest':
            target = nearest[0]
            chosen = {'schema': target, 'action': 'reinforce', 'mutation': 'merge',
                      'signals': self._signals(target, episode)}
        elif self.write_policy == 'rule':
            def rule_order(p):
                s = p['signals']
                preferred = 'split' if s.contextual_potential else 'supersede' if s.temporal_potential else 'reinforce'
                return (p['action'] != preferred, -float(p['schema'].centroid @ episode.vector),
                        p['schema'].schema_id, priority[p['action']])
            chosen = min(plans, key=rule_order)
        else:
            chosen = min(plans, key=lambda p: (p['cost'], priority[p['action']], p['schema'].schema_id))
        target, action, signals = chosen['schema'], chosen['action'], chosen['signals']
        exception = chosen['mutation'] == 'exception'
        if action == 'reinforce':
            if exception:
                target.add_exception(episode)
                self._pending[target.schema_id][tuple(sorted(episode.features.attributes))].add(episode.session_id)
            else:
                already_supported = episode.episode_id in target.support_episode_ids
                if episode.episode_id in target.exception_episode_ids:
                    raise ValueError('One source cannot be support and exception in the same facet')
                if not already_supported: target.support_episode_ids.append(episode.episode_id)
                target._add_features(episode.features)
                if not already_supported: self._sums[target.schema_id] += episode.vector
                total = self._sums[target.schema_id]
                target.centroid = unit_vector(total)
                target.mean_distortion = float(np.clip((1 - np.linalg.norm(total) / len(target.support_episode_ids)) / 2, 0, 1))
                self._latest[target.schema_id] = max(self._latest[target.schema_id], utc_time(episode.timestamp))
            resulting = (target.schema_id,)
        elif action == 'supersede':
            target.status = 'historical'; target.valid_until = episode.timestamp
            successor = self._create_schema(episode, action=action, predecessor_schema_id=target.schema_id)
            target.successor_schema_ids.append(successor.schema_id)
            resulting = (target.schema_id, successor.schema_id)
        else:
            created = self._create_schema(episode, action=action)
            if action == 'split':
                self._split_links[target.schema_id].add(created.schema_id)
                self._split_links[created.schema_id].add(target.schema_id)
            resulting = (target.schema_id, created.schema_id)
        energies = {p['action']: p['terms'] for p in plans if p['schema'].schema_id == target.schema_id}
        self._record_transition(episode, action, target.schema_id, resulting, energies, signals,
                                stored_as_exception=exception)
        audit = [{'schema_id': p['schema'].schema_id, 'action': p['action'], 'mutation': p['mutation'],
                  'terms': p['terms'].as_dict(), 'adjusted_energy': p['cost'],
                  'complexity_price': p['complexity_price']} for p in plans]
        selected_cost = chosen.get('cost')
        self._observation_certificates[key] = {'policy': self.write_policy, 'action': action,
            'source_id': episode.episode_id, 'topic': topic, 'feasible_candidates': audit,
            'selected_adjusted_energy': selected_cost,
            'optimality_gap': max(0., selected_cost - min(p['cost'] for p in plans)) if selected_cost is not None else None}

    def validate_invariants(self):
        # This representation intentionally has MANY transitions per raw source.
        ids = set(self.episodes)
        records = Counter(r.episode_id for r in self.operation_log.all())
        if records != self._observation_counts or set(records) != ids:
            raise ValueError('Atomic observation/transition accounting mismatch')
        grounded = set()
        for schema in self.schemas.values():
            provenance = schema.support_episode_ids + schema.exception_episode_ids
            if set(provenance) - ids or len(provenance) != len(set(provenance)):
                raise ValueError('Missing or duplicate raw provenance')
            grounded.update(provenance)
            if schema.medoid_episode_id not in schema.support_episode_ids:
                raise ValueError('Ungrounded medoid')
            if schema.status == 'historical' and (not schema.valid_until or not schema.successor_schema_ids):
                raise ValueError('Historical state lacks successor interval')
            if schema.predecessor_schema_id:
                old = self.schemas.get(schema.predecessor_schema_id)
                if old is None or schema.schema_id not in old.successor_schema_ids:
                    raise ValueError('Broken version provenance')
        if grounded != ids: raise ValueError('Some immutable sources became inaccessible')
        return {'valid': True, 'episode_count': len(ids), 'schema_count': len(self.schemas),
                'transition_count': sum(records.values()), 'grounded_episode_count': len(grounded),
                'multi_observation_episode_count': sum(n > 1 for n in records.values())}

    def snapshot(self, path):
        raise NotImplementedError('Experimental atomic certificates require observation-keyed export; use diagnostic audit')

    def observation_audit(self):
        return [{'source_id': key[0], 'topic': key[1], 'contexts': list(key[2]), **cert}
                for key, cert in self._observation_certificates.items()]

    def telemetry(self):
        return {**super().telemetry(), 'atomic_state': self.atomic, 'local_complexity': self.local_complexity,
                'complexity_definition': 'lambda * log(1 + schemas_in_owner_predicate_block)' if self.local_complexity else 'global episode cost control',
                'observation_count': sum(self._observation_counts.values()),
                'multi_observation_episode_count': sum(n > 1 for n in self._observation_counts.values()),
                'calibration_status': 'unchanged scalar transfer; new representation not independently recalibrated',
                'max_observation_energy_gap': max((c['optimality_gap'] for c in self._observation_certificates.values()
                    if c['policy'] == 'energy' and c.get('optimality_gap') is not None), default=0.)}
