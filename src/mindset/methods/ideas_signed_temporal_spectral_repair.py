"""Isolated development pilots 7--10; no benchmark labels enter these policies.

These are falsifiable mechanism candidates, not established novel methods.
Signed/version/compiler claims are about explicit observed self-reports, not
world truth. Spectral partitions are geometric and never certify contexts.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from time import perf_counter
import copy
import math
import re
import numpy as np

from mindset.methods.joint import JointMindsetMemory, typed_features, utc_time
from mindset.methods.base import MemoryHit
from mindset.memory.energy import EnergyTerms, unit_vector


SINGLE_SLOTS = frozenset({'location', 'employer', 'occupation'})


@dataclass(frozen=True)
class ObservedFact:
    owner: str
    slot: str
    value: str
    context: tuple[str, ...]
    timestamp: float
    episode_id: str
    explicit_change: bool


@dataclass(frozen=True)
class TemporalProgram:
    operator: str
    owner: str
    slot: str
    time: float | None = None
    end: float | None = None
    context: str | None = None


def observed_facts(episodes):
    """Conservative single-valued, affirmative first-person source facts.

    Ignore quoted speech, negations, hypothetical/future/past-only statements,
    and compound values. Extraction incompleteness is an explicit limitation.
    """
    out = []
    for episode in episodes:
        text = episode.text
        if re.search(r'["“”]|\b(?:not|never|if|would|will|might|used to|formerly|previously|worked)\b', text, re.I):
            continue
        features = typed_features(episode.speaker, text)
        stamp = utc_time(episode.timestamp)
        if not stamp:
            continue
        for slot, values in features.attribute_map().items():
            if slot not in SINGLE_SLOTS or len(set(values)) != 1:
                continue
            value = values[0]
            if re.search(r'\b(?:and|or|but|because|when|while)\b', value):
                continue
            # Context is typed separately, not part of the attribute value.
            for context in features.contexts:
                value = re.sub(r'\s+' + re.escape(context) + r'\s*$', '', value, flags=re.I)
            if value:
                out.append(ObservedFact(episode.speaker.casefold(), slot, value,
                    tuple(sorted(features.contexts)), stamp, episode.episode_id,
                    features.explicit_change))
    return sorted(out, key=lambda f: (f.timestamp, f.episode_id, f.slot))


def certified_edges(facts):
    """Only changed, explicitly ordered same-owner/slot/context source edges."""
    last = {}; edges = []
    for fact in facts:
        key = (fact.owner, fact.slot, fact.context)
        previous = last.get(key)
        if previous and previous.value != fact.value and fact.explicit_change and fact.timestamp > previous.timestamp:
            edges.append((previous, fact))
        last[key] = fact
    return edges


def compile_temporal(query, owners):
    """Narrow executable grammar; no aliases, pronoun resolution or label use."""
    lower = query.casefold()
    found = [o for o in owners if re.search(r'(?<!\w)' + re.escape(o) + r'(?!\w)', lower)]
    if len(found) != 1:
        return None, 'unresolved_or_ambiguous_owner'
    patterns = {
        'location': r'\b(?:live|lives|living|reside|resides|residing|residence)\b',
        'employer': r'\b(?:work|works|working|employer|employed)\b',
        'occupation': r'\b(?:occupation|profession|job title)\b',
    }
    slots = [slot for slot, pattern in patterns.items() if re.search(pattern, lower)]
    if len(slots) != 1:
        return None, 'unresolved_or_ambiguous_slot'
    owner, slot = found[0], slots[0]
    escaped = re.escape(owner)
    direct_owner = {
        'location': (r'(?<!\w)' + escaped + r'\s+(?:(?:currently|now)\s+)?(?:live|lives|living|reside|resides|residing)\b',
                     r'(?<!\w)' + escaped + r"['’]s\s+(?:(?:current|previous)\s+)?residence\b"),
        'employer': (r'(?<!\w)' + escaped + r'\s+(?:(?:currently|now)\s+)?(?:work|works|working|employed)\b',
                     r'(?<!\w)' + escaped + r"['’]s\s+(?:(?:current|previous)\s+)?employer\b"),
        'occupation': (r'(?<!\w)' + escaped + r"['’]s\s+(?:(?:current|previous)\s+)?(?:occupation|profession|job title)\b",),
    }
    if not any(re.search(pattern, lower) for pattern in direct_owner[slot]):
        return None, 'owner_not_direct_subject_or_property'
    # ISO dates only: no local-clock relative dates or guessed year/month.
    dates = re.findall(r'\b\d{4}-\d{2}-\d{2}\b', lower)
    parsed = []
    for date in dates:
        try:
            parsed.append(datetime.strptime(date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            return None, 'invalid_date'
    if re.search(r'\b(?:during|between|throughout)\b', lower):
        if len(parsed) != 2 or parsed[0] >= parsed[1]:
            return None, 'interval_requires_two_ordered_iso_dates'
        return TemporalProgram('valid_during', owner, slot, parsed[0], parsed[1]), 'compiled'
    if parsed:
        if len(parsed) != 1:
            return None, 'ambiguous_date'
        return TemporalProgram('value_at', owner, slot, parsed[0]), 'compiled'
    if re.search(r'\b(?:changes|changed|change history)\b', lower):
        return TemporalProgram('changes', owner, slot), 'compiled'
    if re.search(r'\b(?:previous|previously|before|formerly|used to|earlier)\b', lower):
        return TemporalProgram('previous', owner, slot), 'compiled'
    contexts = re.findall(r'\b(?:at (?:home|work|school|university)|on (?:weekdays|weekends))\b', lower)
    if contexts:
        if len(set(contexts)) != 1:
            return None, 'ambiguous_context'
        return TemporalProgram('context_value', owner, slot, context=contexts[0]), 'compiled'
    if re.search(r'\b(?:current|currently|now|latest|does|is)\b', lower):
        return TemporalProgram('current', owner, slot), 'compiled'
    return None, 'unsupported_temporal_grammar'


def execute_temporal(program, facts):
    if program is None:
        return [], None, 'no_program'
    scoped = [f for f in facts if (f.owner, f.slot) == (program.owner, program.slot)]
    if program.context is not None:
        scoped = [f for f in scoped if program.context in f.context]
    elif any(f.context for f in scoped):
        # Unknown context cannot collapse multiple contextual worlds.
        return [], None, 'context_unspecified'
    if not scoped:
        return [], None, 'no_grounded_facts'
    scoped.sort(key=lambda f: (f.timestamp, f.episode_id))
    by_time = defaultdict(set)
    for fact in scoped:
        by_time[fact.timestamp].add(fact.value)
    if any(len(values) > 1 for values in by_time.values()):
        return [], None, 'simultaneous_conflicting_reports'
    edge_set = {(a.episode_id, b.episode_id) for a, b in certified_edges(scoped)}
    regimes = []
    for fact in scoped:
        if not regimes or regimes[-1].value != fact.value:
            if regimes and (scoped[scoped.index(fact)-1].episode_id, fact.episode_id) not in edge_set:
                return [], None, 'uncertified_value_change'
            regimes.append(fact)
    if program.operator in ('current', 'context_value'):
        chosen = [scoped[-1]]
    elif program.operator == 'previous':
        if len(regimes) < 2:
            return [], None, 'no_certified_previous_regime'
        chosen = [regimes[-2], regimes[-1]]
    elif program.operator == 'changes':
        if len(regimes) < 2:
            return [], None, 'no_certified_changes'
        chosen = regimes
    elif program.operator == 'value_at':
        eligible = [f for f in scoped if f.timestamp <= program.time]
        if not eligible:
            return [], None, 'no_report_at_or_before_requested_time'
        chosen = [eligible[-1]]
    elif program.operator == 'valid_during':
        eligible = [f for f in scoped if f.timestamp <= program.time]
        if not eligible:
            return [], None, 'interval_start_unobserved'
        initial = eligible[-1]
        changes = [f for f in scoped if program.time < f.timestamp < program.end and f.value != initial.value]
        if changes:
            return [], None, 'not_constant_during_interval'
        chosen = [initial]
    else:
        return [], None, 'unknown_operator'
    if program.operator == 'previous':
        answer = chosen[0].value
    elif program.operator == 'changes':
        answer = ' -> '.join(f.value for f in chosen)
    else:
        answer = chosen[-1].value
    return list(dict.fromkeys(f.episode_id for f in chosen)), answer, 'observed_report_algebra_not_world_truth'


def signed_selection(pool, scores, stale_ids, *, top_k=8):
    """Lexicographic anti-support: prefer non-stale, never delete history."""
    if len(pool) != len(scores) or len(set(pool)) != len(pool):
        raise ValueError('Malformed candidate pool')
    if not all(math.isfinite(float(s)) for s in scores):
        raise ValueError('Nonfinite candidate scores')
    ranking = sorted(zip(pool, scores), key=lambda row: (row[0] in stale_ids, -float(row[1]), row[0]))
    return [identifier for identifier, _ in ranking[:max(0, top_k)]]


class _IdeaRetrieval(JointMindsetMemory):
    """One common CE pass over <=48 raw sources for retrieval-only pilots."""
    deterministic_answer = None

    def _rank(self, query, top_k, *, stale=(), reserve=()):
        start = perf_counter()
        candidates, memberships = self.candidate_pool(query)
        anchors = [self._index_by_id[eid] for eid in reserve if eid in self._index_by_id]
        budget = self.config.retrieval_candidate_k
        anchors = list(dict.fromkeys(anchors))[:min(top_k, budget)]
        candidates = anchors + [i for i in candidates if i not in anchors]
        candidates = candidates[:budget]
        self.last_candidate_pool = tuple(candidates)
        built = perf_counter()
        if self.local_reranker is None:
            raise RuntimeError('Pinned local cross-encoder required')
        scores = np.asarray(self.local_reranker.predict([(query, self._documents[i]) for i in candidates],
            batch_size=self.config.rerank_batch_size, show_progress_bar=False), dtype=np.float32)
        if len(scores) != len(candidates) or not np.isfinite(scores).all():
            raise ValueError('Malformed local cross-encoder scores')
        identifiers = [self._episode_order[i] for i in candidates]
        ranking = signed_selection(identifiers, scores, set(stale), top_k=len(identifiers))
        chosen = list(dict.fromkeys(eid for eid in reserve if eid in identifiers))[:top_k]
        chosen += [eid for eid in ranking if eid not in chosen][:max(0,top_k-len(chosen))]
        score = dict(zip(identifiers, scores, strict=True))
        # Reserved source membership, but common CE order within signed tier.
        chosen = sorted(set(chosen), key=lambda eid: (eid in set(stale), -float(score[eid]), eid))
        hits = [MemoryHit(eid, self._documents[self._index_by_id[eid]], float(score[eid]), (eid,),
            'idea_raw_provenance_episode', {'schema_ids': sorted(memberships[self._index_by_id[eid]]),
            'retrieval_version': self.retrieval_version}) for eid in chosen]
        end = perf_counter(); self._query_seconds += end-start; self._query_count += 1
        self._retrieval_profile['candidate_build_seconds'] += built-start
        self._retrieval_profile['cross_encoder_seconds'] += end-built
        self._retrieval_profile['query_episode_pairs'] += len(candidates)
        return hits


class SignedRetrievalMemory(_IdeaRetrieval):
    retrieval_version = 'pilot07-certified-signed-raw-retrieval'

    def retrieve(self, query, top_k=8):
        self.deterministic_answer = None
        program, reason = compile_temporal(query, {e.speaker.casefold() for e in self.episodes.values()})
        stale = set(); edge_count = 0
        if program is not None and program.operator in ('current', 'context_value'):
            facts = observed_facts(self.episodes.values())
            for old, new in certified_edges(facts):
                if (new.owner, new.slot) != (program.owner, program.slot):
                    continue
                if program.context is None and new.context:
                    continue
                if program.context is not None and program.context not in new.context:
                    continue
                stale.add(old.episode_id); edge_count += 1
        self.last_idea_certificate = {'idea': 7, 'compile_reason': reason,
            'anti_support_ids': sorted(stale), 'certified_edge_count': edge_count,
            'constraint_scope': 'observed_explicit_report_versions_only', 'history_deleted': False}
        if top_k <= 0 or not self._episode_order:
            return []
        return self._rank(query, top_k, stale=stale)


class TemporalCompilerMemory(_IdeaRetrieval):
    retrieval_version = 'pilot08-typed-temporal-algebra-raw-witnesses'

    def retrieve(self, query, top_k=8):
        program, reason = compile_temporal(query, {e.speaker.casefold() for e in self.episodes.values()})
        witnesses, answer, execution = execute_temporal(program, observed_facts(self.episodes.values()))
        # Raw evidence experiment always still uses the common final generator.
        self.deterministic_answer = answer
        if len(witnesses) > top_k:
            witnesses = []; self.deterministic_answer = None; execution = 'witness_budget_exceeded'
        self.last_idea_certificate = {'idea': 8, 'compile_reason': reason,
            'operator': program.operator if program else None, 'execution': execution,
            'witness_ids': witnesses, 'deterministic_answer_available': self.deterministic_answer is not None,
            'lookup_is_separate_diagnostic': True}
        if top_k <= 0 or not self._episode_order:
            return []
        return self._rank(query, top_k, reserve=witnesses)


@dataclass(frozen=True)
class SpectralConfig:
    max_nodes: int = 32
    neighbours: int = 4
    minimum_cosine: float = .50
    bandwidth: float = .20
    max_lambda2: float = .15
    minimum_gap: float = .10
    minimum_component: int = 2

    def __post_init__(self):
        if min(self.max_nodes, self.neighbours, self.minimum_component) < 1 or self.bandwidth <= 0:
            raise ValueError('Invalid fixed spectral operational settings')


def spectral_partition(vectors, config=SpectralConfig()):
    """Normalized-Laplacian Fiedler bifurcation on a bounded mutual-kNN graph."""
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Invalid spectral vectors')
    n = len(x)
    if n > config.max_nodes:
        raise ValueError('Spectral local node budget exceeded')
    if n < 2:
        return {'bifurcation': False, 'newest_disconnected': False, 'labels': [0]*n, 'eigenvalues': []}
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)
    cosine = np.clip(x @ x.T, -1, 1); np.fill_diagonal(cosine, -np.inf)
    neighbours = [set(sorted(range(n), key=lambda j: (-cosine[i,j], j))[:min(config.neighbours, n-1)]) for i in range(n)]
    graph = np.zeros((n,n), dtype=np.float64)
    for i in range(n):
        for j in neighbours[i]:
            if i in neighbours[j] and cosine[i,j] >= config.minimum_cosine:
                graph[i,j] = graph[j,i] = math.exp(-(1-cosine[i,j])/config.bandwidth)
    degree = graph.sum(axis=1)
    inv = np.divide(1., np.sqrt(degree), out=np.zeros(n), where=degree > 0)
    laplacian = np.diag((degree > 0).astype(float)) - inv[:,None]*graph*inv[None,:]
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    # Isolated nodes are operational NEW, not semantic/contextual clusters.
    if degree[-1] == 0:
        return {'bifurcation': False, 'newest_disconnected': True, 'labels': [0]*(n-1)+[1],
            'eigenvalues': eigenvalues.tolist(), 'degrees': degree.tolist()}
    if n < 2*config.minimum_component or np.any(degree == 0):
        return {'bifurcation': False, 'newest_disconnected': False, 'labels': [0]*n,
            'eigenvalues': eigenvalues.tolist(), 'degrees': degree.tolist()}
    fiedler = eigenvectors[:,1]
    labels = (fiedler > 0).astype(int)
    # The null eigenspace is non-unique on disconnected graphs. Recover exact
    # graph components rather than depending on an arbitrary eigensolver basis.
    remaining = set(range(n)); components = []
    while remaining:
        stack = [min(remaining)]; component = set()
        while stack:
            node = stack.pop()
            if node in component: continue
            component.add(node)
            stack.extend(int(j) for j in np.flatnonzero(graph[node]) if int(j) not in component)
        remaining -= component; components.append(component)
    if len(components) == 2:
        labels = np.asarray([int(i in components[1]) for i in range(n)])
    # Canonical sign orientation eliminates arbitrary eigenvector sign.
    if labels[0]: labels = 1-labels
    gap = float(eigenvalues[2]-eigenvalues[1])
    bifurcation = (min(np.bincount(labels, minlength=2)) >= config.minimum_component
        and eigenvalues[1] <= config.max_lambda2 and gap >= config.minimum_gap)
    return {'bifurcation': bool(bifurcation), 'newest_disconnected': False,
        'labels': labels.tolist(), 'eigenvalues': eigenvalues.tolist(), 'eigengap': gap,
        'degrees': degree.tolist(), 'semantic_validity_certified': False}


class SpectralBifurcationMemory(JointMindsetMemory):
    retrieval_version = 'pilot09-bounded-spectral-bifurcation'

    def __init__(self, *args, spectral_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.spectral_config = spectral_config or SpectralConfig()
        self.spectral_audit = []; self.last_idea_certificate = {'idea': 9, 'graph_bifurcation_candidates': 0}

    def _plans(self, schema, episode):
        plans = super()._plans(schema, episode)
        ids = schema.support_episode_ids[-(self.spectral_config.max_nodes-1):] + [episode.episode_id]
        result = spectral_partition([self.episodes[eid].vector for eid in ids], self.spectral_config)
        self.spectral_audit.append({'episode_id': episode.episode_id, 'schema_id': schema.schema_id,
            'source_ids': ids, **result})
        if result['newest_disconnected']:
            signals = self._signals(schema, episode)
            terms = EnergyTerms(0,0,0,1,0,0)
            return [{'schema': schema, 'signals': signals, 'action': 'new', 'mutation': 'create',
                'terms': terms, 'switch': 0., 'cost': 0., 'geometric_rule': 'disconnected_observation'}]
        if result['bifurcation']:
            signals = self._signals(schema, episode)
            temporal = signals.temporal_potential > 0
            operation = 'supersede' if temporal else 'split'
            terms = EnergyTerms(0,0,0,1,0,0)
            return [{'schema': schema, 'signals': signals, 'action': operation,
                'mutation': 'version' if temporal else 'branch', 'terms': terms, 'switch': 0., 'cost': 0.,
                'geometric_rule': 'spectral_bifurcation', 'partition_ids': ids, 'partition': result['labels']}]
        # Coherent graph preserves typed contradiction exceptions, not blind merge.
        return [p for p in plans if p['action'] == 'reinforce'] or plans

    def _ingest_episode(self, episode):
        super()._ingest_episode(episode)
        record = self.operation_log.all()[-1]
        if record.operation == 'split':
            partition = next((row for row in reversed(self.spectral_audit)
                if row['episode_id'] == episode.episode_id and row['schema_id'] == record.target_schema_id
                and row['bifurcation']), None)
            if partition:
                old = self.schemas[record.target_schema_id]
                new = self.schemas[record.resulting_schema_ids[-1]]
                incoming_label = partition['labels'][-1]
                moved = [eid for eid, label in zip(partition['source_ids'], partition['labels'], strict=True)
                    if label == incoming_label and eid in old.support_episode_ids]
                old.support_episode_ids = [eid for eid in old.support_episode_ids if eid not in moved]
                new.support_episode_ids = list(dict.fromkeys(moved + new.support_episode_ids))
                for schema in (old,new):
                    vectors = np.asarray([self.episodes[eid].vector for eid in schema.support_episode_ids])
                    total = vectors.sum(axis=0); self._sums[schema.schema_id] = total
                    schema.centroid = unit_vector(total)
                    schema.entity_counts.clear(); schema.role_counts.clear(); schema.context_counts.clear()
                    schema.attribute_counts.clear()
                    for eid in schema.support_episode_ids: schema._add_features(self.episodes[eid].features)
                    schema.medoid_episode_id = schema.support_episode_ids[int(np.argmax(vectors@schema.centroid))]
                    schema.mean_distortion = float(np.clip((1-np.linalg.norm(total)/len(vectors))/2,0,1))
                    self._latest[schema.schema_id] = max(utc_time(self.episodes[eid].timestamp)
                        for eid in schema.support_episode_ids)
                partition['reassigned_ids'] = moved
        self.last_idea_certificate = {'idea': 9,
            'graph_bifurcation_candidates': sum(r['bifurcation'] for r in self.spectral_audit),
            'graph_partition_reassignments': sum(bool(r.get('reassigned_ids')) for r in self.spectral_audit),
            'disconnected_candidates': sum(r['newest_disconnected'] for r in self.spectral_audit),
            'semantic_validity_certified': False}

    def telemetry(self):
        from dataclasses import asdict
        return {**super().telemetry(), 'spectral_operational_settings': asdict(self.spectral_config),
            'geometry_is_not_context_or_truth_certificate': True,
            'spectral_summary': dict(self.last_idea_certificate)}


def source_consistency(memory, scope=None):
    """Source-observation loss, not answer loss: typed conflicts + inaccessible IDs."""
    wanted = set(scope) if scope is not None else set(memory.episodes)
    accessible = set(); conflicts = 0
    for schema in memory.schemas.values():
        accessible.update(schema.all_provenance())
        groups = defaultdict(set)
        for eid in schema.all_provenance():
            episode = memory.episodes[eid]
            if eid not in wanted: continue
            f = typed_features(episode.speaker, episode.text)
            for slot, value in f.attributes:
                if slot in SINGLE_SLOTS:
                    groups[(episode.speaker.casefold(), slot, tuple(sorted(f.contexts)))].add(value)
        conflicts += sum(max(0,len(values)-1) for values in groups.values())
    return conflicts + len(wanted-accessible)


def repair_plan_key(plan):
    """Match the Joint controller's deterministic action priority."""
    priority = {'reinforce': 0, 'supersede': 1, 'split': 2, 'new': 3}
    cost = float(plan['cost'])
    if not math.isfinite(cost): raise ValueError('Nonfinite repair transition cost')
    return cost, priority[plan['action']], str(plan.get('schema_id')), plan['mutation']


def replay_source_greedy(state, episodes):
    """Replay every write through the shared copy-on-write state simulator.

    A fresh initial counterfactual does not mean its *other* schemas are fresh:
    the simulator intentionally shares untouched structures. Direct ingestion
    on a child would mutate those shared schemas and corrupt sibling histories.
    """
    from mindset.methods.idea_transition_tools import feasible_plans, apply_plan
    for episode in episodes:
        plans = feasible_plans(state,episode)
        if not plans: raise ValueError('Repair replay has no feasible action')
        apply_plan(state,episode,min(plans,key=repair_plan_key))
    return state


class CounterfactualRepairMemory(JointMindsetMemory):
    """Bounded test-time structural replay, scored only by source consistency.

    Repair does not claim to know whether an answer is correct. Replay is
    restricted to the last eight transitions and two alternatives per origin.
    It is query-local: the ingestion baseline remains unchanged across queries.
    """
    retrieval_version = 'pilot10-source-consistency-counterfactual-replay'

    def __init__(self,*args,repair_window=8,repair_alternatives=2,**kwargs):
        super().__init__(*args,**kwargs)
        if repair_window <= 0 or repair_alternatives <= 0: raise ValueError('Invalid repair budgets')
        self.repair_window = repair_window; self.repair_alternatives = repair_alternatives
        self.repair_trace = []; self.last_idea_certificate = {'idea': 10, 'triggered': False}

    def retrieve(self,query,top_k=8):
        hits = super().retrieve(query,top_k)
        scope = {eid for hit in hits for eid in hit.source_event_ids}
        baseline = source_consistency(self, scope)
        trigger = not hits or baseline > 0
        self.last_idea_certificate = {'idea': 10, 'triggered': trigger, 'source_loss_before': baseline,
            'oracle_used': False, 'rollback_scope': self.repair_window, 'query_local': True}
        if not trigger or len(self._episode_order) < 2: return hits
        repair_started = perf_counter()
        # Shared exact state simulator, supplied by the bounded pilot harness.
        from mindset.methods.idea_transition_tools import feasible_plans, apply_plan, state_for_plan
        start = max(0,len(self._episode_order)-self.repair_window)
        sequence = [self.episodes[eid] for eid in self._episode_order]
        prefix = JointMindsetMemory(self.config, encoder=self.encoder, local_reranker=self.local_reranker)
        replay_source_greedy(prefix,sequence[:start])
        best = None; evaluations = []; baseline_global = source_consistency(self)
        for offset, origin in enumerate(sequence[start:]):
            choices = feasible_plans(prefix,origin)
            chosen = min(choices,key=repair_plan_key)
            alternatives = [p for p in choices if (p['action'],p.get('schema_id'),p['mutation']) !=
                (chosen['action'],chosen.get('schema_id'),chosen['mutation'])][:self.repair_alternatives]
            for alternative in alternatives:
                candidate = state_for_plan(prefix,origin,alternative)
                replay_source_greedy(candidate,sequence[start+offset+1:])
                loss = source_consistency(candidate)
                row = {'origin_id': origin.episode_id, 'counterfactual_action': alternative['action'],
                    'source_loss': loss, 'influence': baseline_global-loss}
                evaluations.append(row)
                if loss < baseline_global and (best is None or loss < best[0]): best = (loss,candidate,row)
            prefix = apply_plan(prefix,origin,chosen)
        self.last_idea_certificate.update(counterfactuals=evaluations, selected_repair=best[2] if best else None)
        if best is None:
            elapsed = perf_counter()-repair_started
            self._query_seconds += elapsed; self._retrieval_profile['repair_seconds'] += elapsed
            self.last_idea_certificate['repair_seconds'] = elapsed
            self.repair_trace.append(copy.deepcopy(self.last_idea_certificate))
            return hits
        repaired = best[1]
        # Retrieve from repaired provenance, but preserve immutable original rows
        # and do not commit query-order-dependent state changes.
        repaired._episode_order = list(self._episode_order)
        repaired._source_events = dict(self._source_events)
        # Index rebuilding writes extractive medoid IDs on *every* schema,
        # including untouched schemas shared by the copy-on-write simulator.
        repaired.schemas = {sid: copy.deepcopy(schema) for sid,schema in repaired.schemas.items()}
        repaired._rebuild_retrieval_index()
        result = repaired.retrieve(query,top_k)
        elapsed = perf_counter()-repair_started
        self._query_seconds += elapsed; self._retrieval_profile['repair_seconds'] += elapsed
        self._retrieval_profile['query_episode_pairs'] += repaired._retrieval_profile['query_episode_pairs']
        self._retrieval_profile['counterfactual_replay_count'] += len(evaluations)
        self.last_candidate_pool = repaired.last_candidate_pool
        self.last_idea_certificate['source_loss_after'] = best[0]
        self.last_idea_certificate['repair_seconds'] = elapsed
        self.repair_trace.append(copy.deepcopy(self.last_idea_certificate))
        return result
