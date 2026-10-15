"""Experimental joint source-interpretation / state-transition energy.

This is an isolated research candidate, not the paper's frozen controller.
Literal provenance is a hard constraint. Semantic bindings remain hypotheses.
The scorer is fitted separately on independent synthetic read outcomes, never
on benchmark answers. Neither linear risk regression nor finite enumeration
is claimed as a new general-purpose learning/optimization algorithm.
"""
from dataclasses import dataclass, replace
from collections import Counter
import math
import numpy as np


FEATURES = (
    'state_write', 'new_variable', 'same_value', 'value_conflict',
    'known_context_match', 'explicit_context_change', 'unresolved_context',
    'source_later', 'backdated', 'repeat_support', 'retained_versions',
    'negated_write', 'modal_write', 'past_write', 'eventive_write',
    'explicit_change_write', 'unresolved_holder_write', 'discard_same_value',
    'discard_change', 'context_rebinding',
)
GLOBAL_FEATURES = frozenset((
    'new_variable', 'same_value', 'value_conflict', 'known_context_match',
    'explicit_context_change', 'source_later', 'backdated', 'repeat_support',
    'retained_versions', 'discard_same_value', 'discard_change',
    'context_rebinding',
))


@dataclass(frozen=True)
class Source:
    event_id: str
    text: str
    observation_order: int


@dataclass(frozen=True)
class Binding:
    """A candidate interpretation, not a verified fact or resolved coreference."""
    event_id: str
    holder: str
    variable: str
    value: str
    value_start: int
    value_end: int
    context: str = ''
    order: int = 0
    negated: bool = False
    modal: bool = False
    past: bool = False
    eventive: bool = False
    explicit_change: bool = False
    unresolved_holder: bool = False

    @property
    def key(self):
        return (self.holder, self.variable, self.context)

    def validate(self, source):
        if (self.event_id != source.event_id or not self.holder or not self.variable
                or not self.value or not 0 <= self.value_start < self.value_end <= len(source.text)
                or source.text[self.value_start:self.value_end] != self.value):
            raise ValueError('Binding is not an exact nonempty source span')
        if not isinstance(self.order, int):
            raise ValueError('An explicit sortable observation/factual order is required')


@dataclass(frozen=True)
class Version:
    binding: Binding
    sources: tuple[str, ...]


@dataclass(frozen=True)
class Plan:
    binding: Binding
    action: str
    features: tuple[float, ...]
    energy: float


class GroundedStateEnergy:
    """Jointly enumerate admitted bindings and their actual post-state writes.

    Every version is retained. Backdated writes cannot displace a later current
    version. A no-state-write interpretation retains its raw source separately.
    Unknown contexts may bind to an existing context OR remain unspecified;
    this is the explicit lattice, not silently certified disambiguation.
    """
    def __init__(self, model=None, mode='joint'):
        if mode not in {'joint', 'local', 'hard_grounding', 'rules'}:
            raise ValueError('Unknown matched control')
        self.mode = mode
        self.model = model
        if mode != 'rules':
            if not model or tuple(model.get('features', ())) != FEATURES:
                raise ValueError('A separately fitted feature-compatible model is required')
            self.weights = np.asarray(model['weights'], dtype=float)
            self.mean = np.asarray(model['mean'], dtype=float)
            self.scale = np.asarray(model['scale'], dtype=float)
            if any(v.shape != (len(FEATURES),) for v in (self.weights, self.mean, self.scale)):
                raise ValueError('Invalid model dimensions')
            if not np.isfinite(np.r_[self.weights, self.mean, self.scale]).all() or np.any(self.scale <= 0):
                raise ValueError('Invalid learned energy')
        self.sources = {}
        self.versions = {}
        self.exceptions = []
        self.log = []
        self.value_counts = Counter()

    def _features(self, b, action, rebound):
        versions = self.versions.get(b.key, ())
        latest = max(versions, key=lambda v: (v.binding.order, v.binding.event_id), default=None)
        same = latest is not None and latest.binding.value.casefold() == b.value.casefold()
        conflict = latest is not None and not same
        other_context = any(k[:2] == b.key[:2] and k[2] != b.context for k in self.versions)
        write = action != 'exception'
        d = dict(state_write=write, new_variable=write and not versions,
            same_value=write and same, value_conflict=write and conflict,
            known_context_match=write and bool(versions),
            explicit_context_change=write and bool(b.context) and other_context,
            unresolved_context=write and not b.context,
            source_later=write and latest is not None and b.order > latest.binding.order,
            backdated=write and latest is not None and b.order < latest.binding.order,
            repeat_support=write * math.log1p(self.value_counts[(b.key, b.value.casefold())]),
            retained_versions=write * math.log1p(len(versions)),
            negated_write=write and b.negated, modal_write=write and b.modal,
            past_write=write and b.past, eventive_write=write and b.eventive,
            explicit_change_write=write and b.explicit_change,
            unresolved_holder_write=write and b.unresolved_holder,
            discard_same_value=not write and same,
            discard_change=not write and conflict and b.explicit_change,
            context_rebinding=write and rebound)
        return tuple(float(d[k]) for k in FEATURES)

    def candidates(self, source, bindings):
        if source.event_id in self.sources:
            raise ValueError('An immutable source may be ingested only once')
        if self.mode == 'hard_grounding':
            bindings = bindings[:1]
        if not bindings:
            return []
        plans = []; seen = set()
        for original in bindings:
            original.validate(source)
            contexts = [original.context]
            if not original.context and self.mode != 'hard_grounding':
                contexts += sorted({k[2] for k in self.versions if k[:2] == original.key[:2] and k[2]})
            for context in contexts:
                b = replace(original, context=context)
                versions = self.versions.get(b.key, ())
                same = any(v.binding.value.casefold() == b.value.casefold() for v in versions)
                other = any(k[:2] == b.key[:2] and k[2] != b.context for k in self.versions)
                action = 'reinforce' if same else ('supersede' if versions else ('split' if other else 'new'))
                for a in (action, 'exception'):
                    key = (b, a)
                    if key in seen:
                        continue
                    seen.add(key)
                    features = self._features(b, a, context != original.context)
                    x = np.asarray(features)
                    if self.mode == 'local':
                        # A matched learned control, not a separately refitted model.
                        x = np.asarray([0. if k in GLOBAL_FEATURES else v for k, v in zip(FEATURES, x)])
                    cost = float(((x - self.mean) / self.scale) @ self.weights) if self.mode != 'rules' else 0.
                    plans.append(Plan(b, a, features, cost))
        return plans

    def choose(self, plans):
        if not plans:
            return None
        if self.mode == 'rules':
            # Strong same-grounder policy: reject nonassertions/unresolved holders,
            # preserve explicit contexts, otherwise use the most recently used
            # context for the same holder/property. No fitted coefficients.
            b = plans[0].binding
            wanted = 'exception' if b.negated or b.modal or b.eventive or b.unresolved_holder else 'write'
            known = [(max(v.binding.order for v in vs), k[2]) for k, vs in self.versions.items()
                     if k[:2] == b.key[:2]]
            context = b.context or (max(known)[1] if known else '')
            eligible = [p for p in plans if p.binding.holder == b.holder and p.binding.variable == b.variable
                        and p.binding.context == context and (p.action == 'exception') == (wanted == 'exception')]
            return (eligible or plans)[0]
        return min(enumerate(plans), key=lambda pair: (pair[1].energy,
                   pair[1].action != 'exception', pair[0]))[1]

    def commit(self, source, plan):
        if source.event_id in self.sources:
            raise ValueError('Source overwrite is forbidden')
        self.sources[source.event_id] = source
        if plan is None:
            self.log.append(dict(event_id=source.event_id, action='unbound'))
            return
        b = plan.binding; b.validate(source)
        if plan.action == 'exception':
            self.exceptions.append(b)
        else:
            versions = self.versions.setdefault(b.key, [])
            # Same-value reports retain their own observation time. Collapsing
            # them into the earliest version would lose historical read accuracy.
            versions.append(Version(b, (source.event_id,)))
            versions.sort(key=lambda v: (v.binding.order, v.binding.event_id))
        self.value_counts[(b.key, b.value.casefold())] += 1
        self.log.append(dict(event_id=source.event_id, action=plan.action,
            key=list(b.key), value=b.value, energy=plan.energy, features=list(plan.features)))

    def ingest(self, source, bindings):
        plans = self.candidates(source, bindings)
        chosen = self.choose(plans)
        self.commit(source, chosen)
        return chosen, plans

    def read(self, holder, variable, context='', before=None):
        versions = self.versions.get((holder, variable, context), ())
        eligible = [v for v in versions if before is None or v.binding.order <= before]
        return max(eligible, key=lambda v: (v.binding.order, v.binding.event_id), default=None)

    def fork(self):
        clone = GroundedStateEnergy(self.model, self.mode)
        clone.sources = self.sources.copy()
        clone.versions = {k: list(v) for k, v in self.versions.items()}
        clone.exceptions = list(self.exceptions)
        clone.log = list(self.log)
        clone.value_counts = self.value_counts.copy()
        return clone

    def invariant_report(self):
        for versions in self.versions.values():
            for version in versions:
                version.binding.validate(self.sources[version.binding.event_id])
        return dict(immutable_sources=len(self.sources), versions=sum(map(len, self.versions.values())),
                    exceptions=len(self.exceptions), decisions=len(self.log),
                    semantic_correctness_verified=False)


def probe_loss(memory, probes):
    """Evaluator-only exact synthetic read loss; never provided at inference."""
    errors = []
    for p in probes:
        v = memory.read(p['holder'], p['variable'], p.get('context', ''), p.get('before'))
        observed = None if v is None else v.binding.value.casefold()
        expected = None if p['value'] is None else p['value'].casefold()
        errors.append(observed != expected)
    return float(np.mean(errors)) if errors else 0.


def fit_risk(records, ridge=1., local=False):
    """Train-only ridge fit to counterfactual post-state read loss.

    Fixed regularization is an optimization default, not selected on LoCoMo.
    Store it explicitly; no assertion that this removes every design choice.
    """
    if not records or any(r.get('split') != 'train' for r in records):
        raise ValueError('Only independently generated train records may fit risk')
    x = np.asarray([r['features'] for r in records], dtype=float)
    y = np.asarray([r['loss'] for r in records], dtype=float)
    if x.shape[1] != len(FEATURES) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid training data')
    if local:
        x[:, [i for i, k in enumerate(FEATURES) if k in GLOBAL_FEATURES]] = 0.
    mean = x.mean(0); scale = x.std(0); scale[scale < 1e-8] = 1.
    z = (x - mean) / scale
    weights = np.linalg.solve(z.T @ z + ridge * np.eye(x.shape[1]), z.T @ (y - y.mean()))
    return dict(features=list(FEATURES), weights=weights.tolist(), mean=mean.tolist(),
                scale=scale.tolist(), intercept=float(y.mean()), ridge=ridge,
                local_features_only=local,
                objective='train-only squared loss predicting counterfactual post-state probe error',
                benchmark_reference_use=False, risk_is_not_a_calibrated_probability=True)
