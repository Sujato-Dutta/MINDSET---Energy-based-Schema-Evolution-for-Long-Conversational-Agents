"""Integrated source-accessibility controller, isolated development candidate.

The primary potential counts broken top-k schema routes for a deterministic
sample of previously observed source vectors, plus an inaccessible new source.
The frozen geometric/creation energy breaks ties. This is a routing surrogate,
not future answer loss, semantic truth, or a demonstrated novelty claim.
"""
from dataclasses import replace
from hashlib import sha256
from time import perf_counter
import numpy as np

from mindset.methods.joint_one_price import OnePriceJointMemory
from mindset.methods.joint import typed_features, utc_time
from mindset.memory.energy import unit_vector
from mindset.utils.hashing import stable_hash


def top_roots(scores, names, k):
    """Exact dense ranking with the same schema-ID tie break as retrieval."""
    if not len(names):
        return np.empty((len(scores), 0), dtype=int)
    tie = np.asarray(sorted(range(len(names)), key=lambda i: names[i]))
    return tie[np.argsort(-scores[:, tie], axis=1, kind='stable')[:, :k]]


def projected_roots(scores, before, names, k, column, values, new_name=None):
    """One-column update using top-(k+1); no memory-state cloning."""
    n = len(names)
    if new_name is not None:
        column = n
        names = [*names, new_name]
    if not len(scores):
        return np.empty((0, min(k, len(names))), dtype=int)
    candidates = np.column_stack((before, np.full(len(scores), column)))
    selected_scores = np.column_stack((np.take_along_axis(scores, before, axis=1), values))
    # Remove the old copy of a changed column before adding its actual new score.
    selected_scores[:, :-1] = np.where(before == column, -np.inf, selected_scores[:, :-1])
    tie_rank = {sid: rank for rank, sid in enumerate(sorted(names))}
    ties = np.asarray([tie_rank[sid] for sid in names])[candidates]
    order = np.lexsort((ties, -selected_scores), axis=1)[:, :min(k, len(names))]
    return np.take_along_axis(candidates, order, axis=1)


class AccessibilityMemory(OnePriceJointMemory):
    """Actual write transitions minimizing (route damage, original energy).

    No benchmark queries, answers, or relevance labels enter writes. The
    bottom-hash source reservoir is bounded at 128 for this preregistered pilot.
    All episodes remain immutable and readable through the direct channel.
    """
    retrieval_version = 'source-accessibility-joint-v2-pilot'

    def __init__(self, *args, access_mode='full', probe_budget=128, **kwargs):
        super().__init__(*args, **kwargs)
        if access_mode not in {'full', 'no_history', 'no_current', 'no_access'}:
            raise ValueError('Unknown access ablation')
        if probe_budget < 1:
            raise ValueError('Positive probe budget required')
        self.access_mode = access_mode
        self.probe_budget = probe_budget
        self.access_records = {}
        self._route_ids = []
        self._route_centers = None
        self._route_parent = {}
        self._probe_ids = []
        self._probe_vectors = None
        self._probe_owner = {}
        self._route_scores = None
        self._route_seconds = 0.

    def _component(self, sid):
        while self._route_parent[sid] != sid:
            sid = self._route_parent[sid]
        return sid

    def _prepare_routes(self, episode):
        dim = len(episode.vector)
        if self._route_centers is None:
            self._route_centers = np.empty((0, dim), dtype=np.float32)
            self._probe_vectors = np.empty((0, dim), dtype=np.float32)
            self._route_scores = np.empty((0, 0), dtype=np.float32)
        names = self._route_ids
        # Top-(k+1) suffices to project any one-column change exactly.
        self._before_roots = top_roots(self._route_scores, names, self.config.nearest_schema_k + 1)
        components = np.asarray([self._component(sid) for sid in names])
        self._components = components
        self._before_access = np.asarray([
            self._component(self._probe_owner[eid]) in components[roots[:self.config.nearest_schema_k]]
            for eid, roots in zip(self._probe_ids, self._before_roots, strict=True)], dtype=bool)
        self._incoming_scores = self._route_centers @ episode.vector

    def _risk(self, plan, episode):
        sid = plan['schema'].schema_id
        mutation = plan['mutation']
        new = mutation in {'create', 'version', 'branch'}
        names = self._route_ids
        col = len(names) if new else names.index(sid)
        new_name = ('fes-' + stable_hash((episode.episode_id, plan['action'],
                       sid if mutation == 'version' else None, len(self.schemas)))[:16]) if new else None
        center = unit_vector(self._sums[sid] + episode.vector) if mutation == 'merge' else (
            episode.vector if new else self.schemas[sid].centroid)
        components = list(self._components)
        if new:
            components.append(new_name if mutation == 'create' else self._component(sid))
        owner = components[col]
        roots = projected_roots(self._route_scores, self._before_roots, names,
                                self.config.nearest_schema_k, col,
                                self._probe_vectors @ center, new_name)
        accessible = np.asarray([
            self._component(self._probe_owner[eid]) in np.asarray(components)[row]
            for eid, row in zip(self._probe_ids, roots, strict=True)], dtype=bool)
        history = int(np.count_nonzero(self._before_access & ~accessible))
        current_scores = self._incoming_scores.copy()
        if new:
            current_scores = np.append(current_scores, float(episode.vector @ center))
        else:
            current_scores[col] = float(episode.vector @ center)
        current_roots = top_roots(current_scores[None, :], [*names, new_name] if new else names,
                                 self.config.nearest_schema_k)[0]
        current = int(owner not in np.asarray(components)[current_roots])
        risk = (history if self.access_mode not in {'no_history', 'no_access'} else 0) + (
            current if self.access_mode not in {'no_current', 'no_access'} else 0)
        return dict(history_routes_lost=history, new_source_inaccessible=current, access_risk=risk)

    def _commit(self, episode, plan):
        target = plan['schema']; sid = target.schema_id
        mutation = plan['mutation']; action = plan['action']
        if mutation == 'merge':
            target.support_episode_ids.append(episode.episode_id)
            target._add_features(episode.features)
            self._sums[sid] += episode.vector
            target.centroid = unit_vector(self._sums[sid])
            target.mean_distortion = float(np.clip((1 - np.linalg.norm(self._sums[sid]) /
                                                   len(target.support_episode_ids)) / 2, 0, 1))
            self._latest[sid] = max(self._latest[sid], utc_time(episode.timestamp))
            return (sid,)
        if mutation == 'exception':
            target.add_exception(episode)
            self._pending[sid][tuple(sorted(episode.features.attributes))].add(episode.session_id)
            return (sid,)
        if mutation == 'version':
            target.status = 'historical'; target.valid_until = episode.timestamp
        child = self._create_schema(episode, action=action,
                                    predecessor_schema_id=sid if mutation == 'version' else None)
        if mutation == 'version':
            target.successor_schema_ids.append(child.schema_id)
        if mutation == 'branch':
            self._split_links[sid].add(child.schema_id)
            self._split_links[child.schema_id].add(sid)
        return (sid, child.schema_id)

    def _update_routes(self, episode, resulting, mutation):
        for sid in resulting:
            if sid not in self._route_parent:
                self._route_parent[sid] = sid
                self._route_ids.append(sid)
                self._route_centers = np.vstack((self._route_centers, self.schemas[sid].centroid))
                self._route_scores = np.column_stack((self._route_scores,
                                                       self._probe_vectors @ self.schemas[sid].centroid))
            else:
                col = self._route_ids.index(sid)
                self._route_centers[col] = self.schemas[sid].centroid
                self._route_scores[:, col] = self._probe_vectors @ self.schemas[sid].centroid
        if mutation in {'branch', 'version'}:
            self._route_parent[resulting[-1]] = self._component(resulting[0])
        self._probe_owner[episode.episode_id] = resulting[-1]
        keys = [*self._probe_ids, episode.episode_id]
        selected = sorted(keys, key=lambda eid: (sha256(('mindset-access-v1/' + eid).encode()).hexdigest(), eid))[:self.probe_budget]
        if selected != self._probe_ids:
            old = {eid: i for i, eid in enumerate(self._probe_ids)}
            self._probe_vectors = np.asarray([self.episodes[eid].vector for eid in selected], dtype=np.float32)
            self._route_scores = np.asarray([
                self._route_scores[old[eid]] if eid in old else self._route_centers @ episode.vector
                for eid in selected], dtype=np.float32)
            self._probe_ids = selected

    def _ingest_episode(self, episode):
        start = perf_counter()
        episode = replace(episode, features=typed_features(episode.speaker, episode.text))
        self.episodes[episode.episode_id] = episode
        self._prepare_routes(episode)
        topic = f'{episode.speaker.casefold()}::{episode.features.primary_slot}'
        pool = [self.schemas[sid] for sid in self._by_topic[topic] if self.schemas[sid].status == 'active']
        if not pool:
            schema = self._create_schema(episode, action='new')
            self._record_transition(episode, 'new', None, (schema.schema_id,), {}, None)
            self._update_routes(episode, (schema.schema_id,), 'create')
            self._route_seconds += perf_counter() - start
            return
        nearest = sorted(pool, key=lambda s: (-float(s.centroid @ episode.vector), s.schema_id))[:self.config.nearest_schema_k]
        plans = [p for schema in nearest for p in self._plans(schema, episode)]
        priority = {'reinforce': 0, 'supersede': 1, 'split': 2, 'new': 3}
        base_key = lambda p: (p['cost'], priority[p['action']], p['schema'].schema_id)
        base = min(plans, key=base_key)
        risks = [self._risk(p, episode) for p in plans]
        best = min(range(len(plans)), key=lambda i: (risks[i]['access_risk'], *base_key(plans[i])))
        chosen = plans[best]; resulting = self._commit(episode, chosen)
        self.last_features = chosen['signals']
        energies = {p['action']: p['terms'] for p in plans if p['schema'].schema_id == chosen['schema'].schema_id}
        self._record_transition(episode, chosen['action'], chosen['schema'].schema_id, resulting,
                                energies, chosen['signals'], stored_as_exception=chosen['mutation'] == 'exception')
        audit = [dict(schema_id=p['schema'].schema_id, action=p['action'], mutation=p['mutation'],
                      base_energy=float(p['cost']), **risk) for p, risk in zip(plans, risks, strict=True)]
        self.last_joint_candidates = tuple(audit)
        self.access_records[episode.episode_id] = dict(selected=best, candidates=audit,
            changed_from_geometric_minimum=chosen is not base, probe_count=len(self._probe_ids))
        self._decision_certificates[episode.episode_id] = dict(selected_schema_id=chosen['schema'].schema_id,
            selected_mutation=chosen['mutation'], selected_adjusted_energy=chosen['cost'], optimality_gap=0.,
            access_objective=[risks[best]['access_risk'], chosen['cost']],
            policy='lexicographic_source_accessibility_not_answer_loss')
        self._update_routes(episode, resulting, chosen['mutation'])
        self._route_seconds += perf_counter() - start

    def telemetry(self):
        return {**super().telemetry(), 'controller': 'source_accessibility_joint_candidate',
            'access_mode': self.access_mode, 'probe_budget': self.probe_budget,
            'route_controller_seconds': self._route_seconds,
            'route_interventions': sum(r['changed_from_geometric_minimum'] for r in self.access_records.values()),
            'probe_definition': 'bottom-hash observed episode vectors; not independent future questions',
            'objective': 'lexicographic broken prior routes + inaccessible new source; geometric tie break',
            'limitations': 'top-k schema proxy; direct raw retrieval may already rescue a lost route; novelty unverified'}
