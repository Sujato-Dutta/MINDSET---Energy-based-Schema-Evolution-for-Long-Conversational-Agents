"""Pilot 1: bounded reversible transition search, development only.

The objective is the sum of the existing actual-state transition costs, not
future QA accuracy. A beam does not guarantee global optimality. Keeping a
separately completed greedy incumbent only guarantees non-worse objective
than greedy from the exact same frozen prefix and source window.
"""
from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
from time import perf_counter
import math

from mindset.methods.joint import JointMindsetMemory, typed_features


PRIORITY = {'reinforce': 0, 'supersede': 1, 'split': 2, 'new': 3}


def plan_key(plan):
    value = float(plan['cost'])
    if not math.isfinite(value):
        raise ValueError('Nonfinite window-transition objective')
    return value, PRIORITY[plan['action']], str(plan.get('schema_id')), plan['mutation']


def path_key(path):
    return path['cost'], tuple(plan_key(p)[1:] for p in path['plans'])


def search_window(prefix, episodes, feasible, branch, *, beam_width=4):
    """Fresh-branch bounded search with a separately retained greedy path.

    ``branch`` must return an independent state without mutating its input.
    Model objects and immutable source rows may be shared by the simulator.
    """
    if beam_width <= 0:
        raise ValueError('Beam width must be positive')
    empty = {'state': prefix, 'cost': 0., 'plans': [], 'states': []}
    greedy = empty
    for episode in episodes:
        plans = feasible(greedy['state'], episode)
        if not plans:
            raise ValueError('Window episode has no feasible action')
        selected = min(plans, key=plan_key)
        state = branch(greedy['state'], episode, selected)
        greedy = {'state': state, 'cost': greedy['cost'] + float(selected['cost']),
            'plans': greedy['plans'] + [selected], 'states': greedy['states'] + [state]}
    beam = [empty]; expanded = 0
    for episode in episodes:
        children = []
        for parent in beam:
            plans = feasible(parent['state'], episode)
            if not plans:
                raise ValueError('Window episode has no feasible action')
            for plan in plans:
                plan_key(plan)  # Validate before applying it.
                state = branch(parent['state'], episode, plan)
                children.append({'state': state, 'cost': parent['cost'] + float(plan['cost']),
                    'plans': parent['plans'] + [plan], 'states': parent['states'] + [state]})
                expanded += 1
        beam = sorted(children, key=path_key)[:beam_width]
    selected = min([greedy, *beam], key=path_key)
    if selected['cost'] > greedy['cost'] + 1e-12:
        raise AssertionError('Greedy-incumbent guarantee violated')
    return selected, {'greedy_cost': greedy['cost'], 'selected_cost': selected['cost'],
        'expanded_branches': expanded, 'beam_width': beam_width,
        'greedy_incumbent_retained': True, 'global_optimality_claimed': False}


class ReversibleWindowMemory(JointMindsetMemory):
    """Reoptimize only the most recent W sources; older transitions are frozen."""
    retrieval_version = 'pilot01-reversible-window-actual-state-cost'

    def __init__(self, *args, window_size=3, beam_width=4, **kwargs):
        super().__init__(*args, **kwargs)
        if window_size <= 0 or beam_width <= 0:
            raise ValueError('Window and beam budgets must be positive')
        self.window_size = window_size; self.beam_width = beam_width
        self._rwse_prefix = JointMindsetMemory(self.config, encoder=self.encoder,
            local_reranker=self.local_reranker, precomputed_embeddings=self.precomputed_embeddings)
        self._rwse_window = []
        self._rwse_selected_states = []
        self._rwse_previous_plans = {}
        self.window_audit = []
        self.last_idea_certificate = {'idea': 1, 'window_size': window_size, 'beam_width': beam_width}

    def _ingest_episode(self, episode):
        from mindset.methods.idea_transition_tools import feasible_plans, state_for_plan, adopt_state
        start = perf_counter()
        episode = replace(episode, features=typed_features(episode.speaker, episode.text))
        self.episodes[episode.episode_id] = episode
        if len(self._rwse_window) == self.window_size:
            # The oldest action selected on the preceding step is now immutable.
            # The new episode never reaches into the already frozen prefix.
            self._rwse_prefix = self._rwse_selected_states[0]
            self._rwse_window = self._rwse_window[1:]
        self._rwse_window.append(episode)
        winner, audit = search_window(self._rwse_prefix, self._rwse_window,
            feasible_plans, state_for_plan, beam_width=self.beam_width)
        changed = []
        signatures = {}
        for source, plan in zip(self._rwse_window, winner['plans'], strict=True):
            signature = (plan['action'], plan.get('schema_id'), plan['mutation'])
            signatures[source.episode_id] = signature
            prior = self._rwse_previous_plans.get(source.episode_id)
            if prior is not None and prior != signature:
                changed.append({'episode_id': source.episode_id,
                    'previous': list(prior), 'selected': list(signature)})
        # Shared utility copies only structural state; model references and this
        # subclass's rolling-window bookkeeping must remain intact.
        adopt_state(self, winner['state'])
        self._rwse_selected_states = winner['states']
        self._rwse_previous_plans = signatures
        self.last_idea_certificate = {'idea': 1, 'window_ids': [e.episode_id for e in self._rwse_window],
            'frozen_prefix_transition_count': len(self._rwse_prefix.operation_log.all()),
            'action_changes': changed, 'search_seconds': perf_counter()-start, **audit}
        self.window_audit.append(dict(self.last_idea_certificate))

    def _rebuild_retrieval_index(self):
        # The common index builder updates medoids in every schema. Detach
        # before those writes so retained frozen-prefix/window snapshots stay
        # immutable even when a schema was untouched by the selected path.
        self.schemas = {sid:deepcopy(schema) for sid,schema in self.schemas.items()}
        super()._rebuild_retrieval_index()

    def telemetry(self):
        return {**super().telemetry(), 'reversible_window': {'window_size': self.window_size,
            'beam_width': self.beam_width, 'window_search_count': len(self.window_audit),
            'reconsidered_transition_count': sum(len(row['action_changes']) for row in self.window_audit),
            'expanded_branches': sum(row['expanded_branches'] for row in self.window_audit),
            'search_seconds': sum(row['search_seconds'] for row in self.window_audit),
            'max_selected_minus_greedy': max((row['selected_cost']-row['greedy_cost']
                for row in self.window_audit), default=0.),
            'objective_is_not_answer_accuracy': True}}
