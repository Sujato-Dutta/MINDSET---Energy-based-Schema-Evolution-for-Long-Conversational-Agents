"""Branch-safe, source-only transitions for the isolated ten-idea pilot.

Frozen Joint controller is unchanged. Plans enumerate its feasible actual
mutations; explicit create/defer proposals may be requested by a prototype.
Model handles and immutable episodes are shared, never deep-copied. A branch
detaches its mutable target before changing it. This is not a global optimizer.
"""
from collections import defaultdict
from copy import copy, deepcopy
from dataclasses import replace, asdict
from collections.abc import Mapping
import numpy as np
from mindset.methods.joint import JointMindsetMemory, typed_features, utc_time
from mindset.methods.base import ConversationEvent
from mindset.memory.energy import EnergyTerms, TransitionLog, unit_vector

STATE_FIELDS = ('episodes', 'schemas', '_episode_order', '_source_events',
    '_sums', '_latest', '_by_topic', '_pending', '_split_links', 'operation_log',
    '_decision_certificates', 'last_features', 'last_joint_candidates')

class _ObservedPrefix(Mapping):
    """Read-only single-source overlay; O(1) len for creation-price scoring."""
    def __init__(self, parent, incoming):
        self.parent=parent; self.incoming=incoming
    def __len__(self):
        return len(self.parent)+int(self.incoming.episode_id not in self.parent)
    def __iter__(self):
        yield from self.parent
        if self.incoming.episode_id not in self.parent: yield self.incoming.episode_id
    def __getitem__(self, key):
        if key==self.incoming.episode_id: return self.incoming
        return self.parent[key]

def _branch(memory):
    # Plain Joint object avoids invoking prototype ingestion recursively.
    result = object.__new__(JointMindsetMemory)
    result.__dict__ = memory.__dict__.copy()
    result.schemas = memory.schemas.copy()
    result.episodes = memory.episodes.copy()
    result._episode_order = list(memory._episode_order)
    result._source_events = memory._source_events.copy()
    result._sums = memory._sums.copy()
    result._latest = memory._latest.copy()
    result._by_topic = defaultdict(list, memory._by_topic)
    result._pending = defaultdict(lambda: defaultdict(set),
        {sid: defaultdict(set, values) for sid, values in memory._pending.items()})
    result._split_links = defaultdict(set, memory._split_links)
    result.operation_log = TransitionLog()
    result.operation_log._records = list(memory.operation_log.all())
    result._decision_certificates = memory._decision_certificates.copy()
    return result

def _register(memory, episode):
    episode = replace(episode, features=typed_features(episode.speaker, episode.text))
    memory.episodes[episode.episode_id] = episode
    if episode.episode_id not in memory._source_events:
        memory._source_events[episode.episode_id] = ConversationEvent(
            episode.episode_id, episode.conversation_id, episode.session_id,
            episode.timestamp, episode.speaker, episode.text, episode.metadata)
    if episode.episode_id not in memory._episode_order:
        memory._episode_order.append(episode.episode_id)
    return episode

def feasible_plans(memory, episode):
    # Scoring is read-only: avoid copying a complete history and transition log
    # simply to ask for feasible costs. Detach pending dicts because the frozen
    # signal extractor uses defaultdict lookups even while only reading.
    episode=replace(episode,features=typed_features(episode.speaker,episode.text))
    view=object.__new__(JointMindsetMemory)
    view.__dict__=memory.__dict__.copy()
    view.episodes=_ObservedPrefix(memory.episodes,episode)
    view._pending=defaultdict(lambda:defaultdict(set),memory._pending)
    topic = f'{episode.speaker.casefold()}::{episode.features.primary_slot}'
    pool = [view.schemas[sid] for sid in view._by_topic.get(topic, ())
            if view.schemas[sid].status == 'active']
    if not pool:
        return [dict(schema_id=None, action='new', mutation='create', cost=0.)]
    nearest = sorted(pool, key=lambda s: (-float(s.centroid @ episode.vector), s.schema_id))
    output = []
    for schema in nearest[:view.config.nearest_schema_k]:
        view._pending[schema.schema_id]=defaultdict(set,memory._pending.get(schema.schema_id,{}))
        for plan in JointMindsetMemory._plans(view, schema, episode):
            output.append(dict(schema_id=schema.schema_id, action=plan['action'],
                mutation=plan['mutation'], cost=float(plan['cost']),
                terms=plan['terms'].as_dict(), signals=asdict(plan['signals']),
                switching_cost=plan['switch']))
    return output

def state_for_plan(memory, episode, plan):
    """Return a fresh state with precisely the requested mutation recorded."""
    state = _branch(memory)
    episode = _register(state, episode)
    sid = plan.get('schema_id')
    action, mutation = plan['action'], plan['mutation']
    if (action, mutation) not in {('new','create'), ('reinforce','merge'),
            ('reinforce','exception'), ('supersede','version'), ('split','branch')}:
        raise ValueError('Unknown actual transition mutation')
    target = state.schemas.get(sid)
    if mutation != 'create' and target is None:
        raise ValueError('Transition target is missing')
    if target is not None:
        target = deepcopy(target)
        state.schemas[sid] = target
        state._sums[sid] = state._sums[sid].copy()
        state._split_links[sid] = set(state._split_links.get(sid, ()))
        state._pending[sid] = defaultdict(set,
            {key: set(value) for key,value in state._pending.get(sid, {}).items()})
    topic = f'{episode.speaker.casefold()}::{episode.features.primary_slot}'
    state._by_topic[topic] = list(state._by_topic.get(topic, ()))
    signals = JointMindsetMemory._signals(state, target, episode) if target else None
    if mutation == 'merge':
        target.support_episode_ids.append(episode.episode_id)
        target._add_features(episode.features)
        state._sums[sid] += episode.vector
        total = state._sums[sid]
        target.centroid = unit_vector(total)
        target.mean_distortion = float(np.clip((1-np.linalg.norm(total)/len(target.support_episode_ids))/2,0,1))
        state._latest[sid] = max(state._latest[sid], utc_time(episode.timestamp))
        resulting = (sid,)
    elif mutation == 'exception':
        target.add_exception(episode)
        state._pending[sid][tuple(sorted(episode.features.attributes))].add(episode.session_id)
        resulting = (sid,)
    else:
        if mutation == 'version':
            target.status = 'historical'
            target.valid_until = episode.timestamp
        created = state._create_schema(episode, action=action,
            predecessor_schema_id=sid if mutation == 'version' else None)
        if mutation == 'version':
            target.successor_schema_ids.append(created.schema_id)
        if mutation == 'branch':
            state._split_links[sid].add(created.schema_id)
            state._split_links[created.schema_id] = {sid}
        resulting = (sid, created.schema_id) if sid else (created.schema_id,)
    terms = plan.get('terms')
    energies = {action: EnergyTerms(**terms)} if terms else {}
    state._record_transition(episode, action, sid, resulting, energies, signals,
                             stored_as_exception=mutation == 'exception')
    state.last_features = signals
    state.last_joint_candidates = (dict(plan),)
    state._decision_certificates[episode.episode_id] = {
        'selected_adjusted_energy': float(plan['cost']), 'selected_mutation':mutation,
        'selected_schema_id':sid, 'optimality_gap':0.,
        'policy':'explicit_counterfactual_pilot_proposal_not_global_optimum'}
    return state

def adopt_state(memory, state):
    """Adopt structural state without replacing prototype/model bookkeeping."""
    for field in STATE_FIELDS:
        setattr(memory, field, getattr(state, field))
    return memory

def apply_plan(memory, episode, plan):
    return adopt_state(memory, state_for_plan(memory, episode, plan))
