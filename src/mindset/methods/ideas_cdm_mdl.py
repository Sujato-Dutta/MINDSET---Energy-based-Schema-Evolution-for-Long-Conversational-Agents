"""Isolated pilot ideas 2 and 3: deferred action sets and description lengths.

These are experimental controllers, not paper defaults. Split-conformal action
coverage is conditional on exchangeability with the independently generated
calibration histories; canonical synthetic templates cannot establish that
condition on LoCoMo/LongMemEval. MDL describes the extracted observation stream,
not the original language itself. Original raw source storage is common to all
candidate states and is never deleted or replaced by these codes.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import replace
from math import ceil, exp, isfinite, lgamma, log, log2

import numpy as np

from mindset.methods.base import ConversationEvent
from mindset.methods.joint import JointMindsetMemory, typed_features, utc_time
from mindset.memory.energy import EnergyEpisode
from mindset.utils.hashing import stable_hash


def split_conformal_quantile(scores, alpha=.1):
    """Conservative finite-sample split-conformal quantile, with +infinity.

    Rank ceil((n+1)(1-alpha)); importantly this is NOT np.quantile(scores,
    1-alpha), and the missing n+1-th observation is +infinity.
    """
    values = sorted(float(x) for x in scores)
    if not values or not 0 < alpha < 1:
        raise ValueError("Nonempty calibration scores and 0<alpha<1 required")
    if any(not isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError("Action nonconformity must be finite and in [0,1]")
    rank = ceil((len(values) + 1) * (1 - alpha))
    return float("inf") if rank > len(values) else values[rank - 1]


def action_nonconformity(plans):
    """1-softmax(-minimum feasible energy per action), temperature fixed at 1.

    This preserves the frozen energy scores, but calibrates ACTION uncertainty;
    it does not calibrate which target schema is correct within an action.
    """
    costs = {}
    for plan in plans:
        cost = float(plan["cost"])
        if not isfinite(cost):
            raise ValueError("Nonfinite feasible transition cost")
        action = plan["action"]
        costs[action] = min(costs.get(action, float("inf")), cost)
    if not costs:
        raise ValueError("At least one feasible action required")
    minimum = min(costs.values())
    mass = {a: exp(-(c - minimum)) for a, c in costs.items()}
    total = sum(mass.values())
    return {a: 1 - p / total for a, p in mass.items()}


def conformal_actions(plans, calibration):
    scores = action_nonconformity(plans)
    threshold = calibration.get("quantile")
    threshold = float("inf") if threshold is None else float(threshold)
    # Empty sets remain abstentions, not a forced confident argmin decision.
    return tuple(sorted(a for a, score in scores.items() if score <= threshold))


def synthetic_cdm_corpus(seed=8347, histories=96):
    """IID synthetic mixture, one prespecified action observation per history.

    Labels come from declared source trajectories, not the energy controller.
    New speaker/ID prefixes separate these observations from prior calibration.
    Reused canonical template families are disclosed, not presented as OOD.
    """
    if histories < 1:
        raise ValueError("At least one independent history required")
    rng = np.random.default_rng(seed)
    cities = ("Valparaíso", "Trieste", "Poznan", "Tartu", "Dunedin", "Agadir")
    families = ("repeat", "update", "coexist", "noise", "confirmation", "new_topic")
    result = []
    for i in range(histories):
        family = families[int(rng.integers(len(families)))]
        left, right = rng.choice(cities, 2, replace=False)
        if family == "repeat":
            texts = [f"I live in {left}."] * 3
            labels = ["new", "reinforce", "reinforce"]
        elif family == "update":
            texts = [f"I live in {left}.", f"I moved to {right}.", f"I live in {right}."]
            labels = ["new", "supersede", "reinforce"]
        elif family == "coexist":
            texts = ["I prefer tea at home.", "I prefer coffee at work.", "I prefer tea at home."]
            labels = ["new", "split", "reinforce"]
        elif family == "noise":
            texts = [f"I live in {left}.", f"I live in {right}.", f"I live in {left}."]
            labels = ["new", "reinforce", "reinforce"]
        elif family == "confirmation":
            texts = [f"I live in {left}.", f"I live in {right}.", f"I live in {right}."]
            labels = ["new", "reinforce", "supersede"]
        else:
            texts = ["An observatory measures starlight.", "A recipe combines rice and lentils.",
                     "A recipe combines rice and lentils."]
            labels = ["new", "new", "reinforce"]
        identifier = f"pilot-cdm-{seed}-{i:04d}"
        # Sample nuisance names from the same distribution, rather than encoding
        # the history index into the sentence-encoder input.
        speaker = f"PilotCDM{int(rng.integers(1_000_000_000))}"
        events = [ConversationEvent(f"{identifier}-e{j}", identifier, f"{identifier}-s{j}",
                                   f"2033-03-{j+1:02d}", speaker, text)
                  for j, text in enumerate(texts)]
        result.append({"id": identifier, "family": family, "events": events, "actions": labels,
                       "calibration_index": 2 if family == "confirmation" else 1})
    return result


def calibrate_cdm(encoder, config, *, seed=8347, histories=96, alpha=.1):
    """Calibrate only on independent synthetic histories, never QA questions.

    Caller must execute on DGX Slurm and persist the returned complete artifact.
    Fixed action scores are produced by the already-frozen Joint controller.
    """
    from mindset.methods.idea_transition_tools import feasible_plans
    from mindset.methods.mindset import _render_event
    corpus = synthetic_cdm_corpus(seed, histories)
    events = [e for h in corpus for e in h["events"]]
    vectors = encoder.encode([_render_event(e) for e in events], batch_size=64,
        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    lookup = {e.event_id: np.asarray(v, dtype=np.float32) for e, v in zip(events, vectors, strict=True)}
    records = []
    for h in corpus:
        index = h["calibration_index"]
        memory = JointMindsetMemory(config, precomputed_embeddings=lookup)
        memory.ingest(h["events"][:index])
        event = h["events"][index]
        episode = EnergyEpisode(event.event_id, event.conversation_id, event.session_id,
            event.timestamp, event.speaker, event.text, lookup[event.event_id],
            typed_features(event.speaker, event.text), index, {})
        # Match ingest's pre-decision observed-prefix episode count exactly;
        # omitting the incoming immutable source changes its complexity price.
        memory.episodes[episode.episode_id] = episode
        plans = feasible_plans(memory, episode)
        scores = action_nonconformity(plans)
        label = h["actions"][index]
        records.append({"history_id": h["id"], "family": h["family"], "true_action": label,
            "score": scores.get(label, 1.), "true_action_feasible": label in scores,
            "feasible_action_nonconformities": scores,
            "events": [dict(event_id=e.event_id, conversation_id=e.conversation_id,
                session_id=e.session_id, timestamp=e.timestamp, speaker=e.speaker, text=e.text)
                for e in h["events"]], "calibration_index": index})
    threshold = split_conformal_quantile([r["score"] for r in records], alpha)
    return {"method": "pilot-conformal-deferred-actions-v1", "alpha": alpha, "seed": seed,
        "n_independent_histories": len(records), "rank": ceil((len(records)+1)*(1-alpha)),
        "quantile": threshold if isfinite(threshold) else None,
        "quantile_infinite": not isfinite(threshold), "records": records,
        "records_sha256": stable_hash(records), "label_source": "independent_synthetic_trajectories",
        "score_definition": "one_minus_softmax_negative_frozen_action_energy_temperature_one",
        "guarantee_scope": "marginal action coverage only under exchangeability with calibration mixture",
        "natural_benchmark_coverage_claim": False,
        "limitations": "canonical templates reused; synthetic noise/update semantics assumed; target-schema uncertainty uncalibrated"}


def elias_gamma_nats(value):
    """Prefix-free universal positive-integer gamma code length in nats."""
    if value < 1 or int(value) != value:
        raise ValueError("Gamma coding requires a positive integer")
    return (2 * int(log2(value)) + 1) * log(2)


def kt_nats(counts, alphabet):
    """Exact Dirichlet-half sequential categorical observation code."""
    values = list(counts)
    if alphabet < 1 or len(values) > alphabet or any(c < 0 or int(c) != c for c in values):
        raise ValueError("Invalid categorical observations")
    return max(0., lgamma(sum(values) + alphabet/2) - lgamma(alphabet/2)
               - sum(lgamma(c+.5)-lgamma(.5) for c in values))


def observation_channels(episode):
    # Attribute values and contexts are categorical observations, not truths.
    # Empty is a symbol, so all episodes are decoded without absent-field holes.
    return {"tokens": tuple(episode.features.tokens) or ("<empty>",),
            "attributes": tuple(f"{k}={v}" for k, v in episode.features.attributes) or ("<none>",),
            "contexts": tuple(episode.features.contexts) or ("<unknown>",),
            "roles": tuple(episode.features.roles) or ("<unknown>",)}


class _DescriptionCache:
    """Bounded exact sufficient statistics, shared safely by ephemeral branches.

    No vectors, models or complete historical branch states are copied. Frozen
    feature records are retained for mutation checking; rows are recomputed if
    their extracted observations or timestamps change. Cache keys describe the
    state, never just schema IDs, so sibling branches cannot contaminate costs.
    """
    channels = ("tokens", "attributes", "contexts", "roles")

    def __init__(self, max_statistics=64, max_lengths=256):
        self.sources = {}
        self.prefix = frozenset()
        self.symbols = {channel: set() for channel in self.channels}
        self.dates = set()
        self.statistics = OrderedDict()
        self.lengths = OrderedDict()
        self.max_statistics = max_statistics
        self.max_lengths = max_lengths
        self.hits = Counter()

    @staticmethod
    def _remember(table, key, value, maximum):
        table[key] = value
        table.move_to_end(key)
        while len(table) > maximum:
            table.popitem(last=False)
        return value

    def context(self, memory, assigned):
        changed = set()
        for eid in assigned:
            episode = memory.episodes[eid]
            old = self.sources.get(eid)
            if (old is None or old[0] != episode.features or old[1] != episode.timestamp):
                version = old[3] + 1 if old is not None else 0
                self.sources[eid] = (episode.features, episode.timestamp, observation_channels(episode), version)
                changed.add(eid)
        current = frozenset(assigned)
        # Exact incremental dictionary growth applies only to source additions.
        # Modified/removed sources rebuild the observed-prefix alphabet instead.
        if self.prefix <= current and not (changed & self.prefix):
            added = current - self.prefix
            for eid in sorted(added):
                record = self.sources[eid]
                for channel in self.channels:
                    self.symbols[channel].update(record[2][channel])
                self.dates.add(record[1])
            self.hits["incremental_prefix"] += 1
        else:
            self.symbols = {channel: set() for channel in self.channels}
            self.dates = set()
            for eid in sorted(current):
                record = self.sources[eid]
                for channel in self.channels:
                    self.symbols[channel].update(record[2][channel])
                self.dates.add(record[1])
            self.hits["rebuilt_prefix"] += 1
        self.prefix = current
        return {c: len(self.symbols[c]) for c in self.channels}, self.dates

    def schema_fingerprint(self, schema, links):
        support = tuple(schema.support_episode_ids)
        exceptions = tuple(schema.exception_episode_ids)
        return (schema.schema_id, support, exceptions, schema.status, schema.predecessor_schema_id,
                tuple(schema.successor_schema_ids), schema.valid_from, schema.valid_until, tuple(links),
                tuple(self.sources[eid][3] for eid in (*support, *exceptions)))

    def schema_residual(self, schema, links, alphabet):
        fingerprint = self.schema_fingerprint(schema, links)
        key = (fingerprint, tuple(alphabet[c] for c in self.channels))
        if key in self.lengths:
            self.lengths.move_to_end(key)
            self.hits["schema_length"] += 1
            return self.lengths[key]
        stats = self.statistics.get(fingerprint)
        if stats is None:
            stats = {}
            for channel in self.channels:
                counts = Counter()
                support_lengths = exception_lengths = 0.
                exception_symbols = 0
                for eid in schema.support_episode_ids:
                    values = self.sources[eid][2][channel]
                    support_lengths += elias_gamma_nats(len(values))
                    counts.update(values)
                for eid in schema.exception_episode_ids:
                    values = self.sources[eid][2][channel]
                    exception_lengths += elias_gamma_nats(len(values))
                    exception_symbols += len(values)
                stats[channel] = (counts, support_lengths, exception_lengths, exception_symbols)
            self._remember(self.statistics, fingerprint, stats, self.max_statistics)
            self.hits["schema_statistics_miss"] += 1
        else:
            self.statistics.move_to_end(fingerprint)
            self.hits["schema_statistics"] += 1
        residual = 0.
        for channel in self.channels:
            counts, support_lengths, exception_lengths, exception_symbols = stats[channel]
            width = alphabet[channel]
            residual += support_lengths + kt_nats(counts.values(), width)
            residual += exception_lengths + exception_symbols * log(width)
        return self._remember(self.lengths, key, residual, self.max_lengths)


def description_length(memory, *, use_cache=True):
    """Whole observed state coding, in nats, with no energy-weight mixture.

    Codes: schema count/status/predecessor/time/link pointers, source-to-schema assignments,
    per-source exception flag, channel lengths, KT support symbols, and uniform
    exception symbols. Dictionaries/raw source strings are invariant across
    feasible states at this prefix and therefore omitted as common constants.
    This models EXTRACTED categorical observations, not semantic truth or a
    compressor of language. Shared observed-prefix alphabets prevent unfairly
    cheaper alphabets for differently partitioned candidates.
    """
    schemas = sorted(memory.schemas.values(), key=lambda s: s.schema_id)
    if not schemas:
        return {"structure_nats": elias_gamma_nats(1), "residual_nats": 0., "total": elias_gamma_nats(1)}
    memberships = Counter(eid for s in schemas for eid in (*s.support_episode_ids, *s.exception_episode_ids))
    if any(count != 1 for count in memberships.values()):
        raise ValueError("MDL assignment code requires exactly one schema per source")
    assigned = set(memberships)
    if use_cache:
        cache = getattr(memory, "_mdl_description_cache", None)
        if cache is None:
            cache = _DescriptionCache()
            memory._mdl_description_cache = cache
    else:
        cache = _DescriptionCache()
    alphabet, dates = cache.context(memory, assigned)
    k, n = len(schemas), len(assigned)
    structure = elias_gamma_nats(k) + k * log(2) + n * log(k) + n * log(2)
    residual = 0.
    for schema in schemas:
        # A null pointer is explicitly encoded; historical/active bit above.
        structure += log(k + 1)
        if schema.predecessor_schema_id:
            if schema.predecessor_schema_id not in memory.schemas:
                raise ValueError("Dangling schema predecessor")
        if schema.valid_from not in dates or (schema.valid_until is not None and schema.valid_until not in dates):
            raise ValueError("Validity times must point to observed source times")
        structure += 2 * log(len(dates) + 1)
        links = sorted(memory._split_links.get(schema.schema_id, ()))
        if any(link not in memory.schemas for link in links):
            raise ValueError("Dangling schema context link")
        structure += elias_gamma_nats(len(links) + 1) + len(links) * log(k)
        structure += elias_gamma_nats(len(schema.support_episode_ids)+1)
        residual += cache.schema_residual(schema, links, alphabet)
    return {"structure_nats": structure, "residual_nats": residual, "total": structure + residual}


class ConformalDeferredMemory(JointMindsetMemory):
    """Commit singleton calibrated ACTION sets; otherwise preserve shadow state.

    A shadow retains the raw episode and all plausible action/target descriptors,
    while the actual active schema remains unchanged except an exception link.
    Later compatible observed sources can resolve its action; original sources
    and older validity boundaries are not retrospectively rewritten as truth.
    """
    retrieval_version = "pilot-cdm-frozen-joint-access-path-v1"

    def __init__(self, *args, calibration, **kwargs):
        if calibration.get("label_source") != "independent_synthetic_trajectories":
            raise ValueError("CDM requires independent synthetic action calibration")
        if calibration.get("records_sha256") != stable_hash(calibration.get("records", [])):
            raise ValueError("Conformal calibration records changed")
        records = calibration["records"]
        if (len(records) != calibration.get("n_independent_histories")
            or len({r["history_id"] for r in records}) != len(records)):
            raise ValueError("Conformal observations must be one per independent history")
        expected = split_conformal_quantile([r["score"] for r in records], calibration["alpha"])
        supplied = float("inf") if calibration.get("quantile") is None else float(calibration["quantile"])
        if expected != supplied:
            raise ValueError("Conformal threshold does not equal the finite-sample calibration quantile")
        self.calibration = calibration
        self.shadow_states = {}
        self.cdm_counts = Counter()
        super().__init__(*args, **kwargs)

    def _ingest_episode(self, episode):
        from mindset.methods.idea_transition_tools import feasible_plans, apply_plan
        episode = replace(episode, features=typed_features(episode.speaker, episode.text))
        self.episodes[episode.episode_id] = episode
        plans = feasible_plans(self, episode)
        action_set = conformal_actions(plans, self.calibration)
        if len(action_set) == 1:
            chosen = min((p for p in plans if p["action"] == action_set[0]),
                         key=lambda p: (p["cost"], p.get("schema_id") or ""))
            resolved = []
            for eid, shadow in self.shadow_states.items():
                old = self.episodes[eid]
                if (shadow["resolved_by"] is None and old.speaker == episode.speaker
                    and old.features.attributes == episode.features.attributes
                    and old.features.contexts == episode.features.contexts
                    and old.session_id != episode.session_id
                    and utc_time(old.timestamp) != 0
                    and utc_time(episode.timestamp) > utc_time(old.timestamp)
                    and chosen["action"] in shadow["plausible_actions"]):
                    shadow["resolved_by"] = episode.episode_id
                    shadow["resolved_action"] = chosen["action"]
                    resolved.append(eid)
            self.cdm_counts["committed"] += 1
        else:
            preserves = [p for p in plans if p["action"] == "reinforce"]
            if not preserves:
                # First source has only create, and should be a singleton. An
                # empty conformal set must still keep the immutable source.
                chosen = min(plans, key=lambda p: (p["cost"], p.get("schema_id") or ""))
            else:
                chosen = dict(min(preserves, key=lambda p: (p["cost"], p.get("schema_id") or "")))
                chosen["mutation"] = "exception"
            self.shadow_states[episode.episode_id] = {"plausible_actions": list(action_set),
                "plans": [{k: p[k] for k in ("schema_id", "action", "mutation", "cost")}
                          for p in plans if p["action"] in action_set],
                "resolved_by": None, "resolved_action": None}
            self.cdm_counts["deferred"] += 1
            resolved = []
        apply_plan(self, episode, chosen)
        self._decision_certificates[episode.episode_id] = {"policy": "conformal_deferred_action",
            "action_prediction_set": list(action_set), "deferred": len(action_set) != 1,
            "resolved_shadows": resolved, "nonconformity": action_nonconformity(plans),
            "calibration_records_sha256": self.calibration["records_sha256"],
            "optimality_gap": max(0., chosen["cost"] - min(p["cost"] for p in plans)),
            "target_schema_guarantee": False, "natural_coverage_guarantee": False}

    def telemetry(self):
        return {**super().telemetry(), "controller": "pilot_cdm", "cdm_counts": dict(self.cdm_counts),
            "unresolved_shadow_count": sum(s["resolved_by"] is None for s in self.shadow_states.values()),
            "nominal_alpha": self.calibration["alpha"], "calibration_histories": self.calibration["n_independent_histories"],
            "natural_coverage_guarantee": False, "parameter_status": "independently_synthetic_calibrated_action_sets"}


class MDLDescriptionMemory(JointMindsetMemory):
    """Choose exact minimum whole-state coding among locally feasible edits."""
    retrieval_version = "pilot-mdl-frozen-joint-access-path-v1"

    def __init__(self, *args, **kwargs):
        self._mdl_description_cache = _DescriptionCache()
        super().__init__(*args, **kwargs)

    def _ingest_episode(self, episode):
        from mindset.methods.idea_transition_tools import feasible_plans, apply_plan, state_for_plan
        episode = replace(episode, features=typed_features(episode.speaker, episode.text))
        self.episodes[episode.episode_id] = episode
        plans = feasible_plans(self, episode)
        coded = [(p, description_length(state_for_plan(self, episode, p))) for p in plans]
        chosen, terms = min(coded, key=lambda item: (item[1]["total"],
            item[0]["action"] != "reinforce", item[0].get("schema_id") or ""))
        apply_plan(self, episode, chosen)
        self._decision_certificates[episode.episode_id] = {"policy": "whole_state_categorical_MDL",
            "selected_code_nats": terms, "feasible_candidates": [
                {"action": p["action"], "mutation": p["mutation"], "schema_id": p["schema_id"],
                 "code_nats": cost} for p, cost in coded],
            "optimality_gap": max(0., chosen["cost"] - min(p["cost"] for p in plans)),
            "coding_choices": "Elias integer codes, uniform structural pointers, Jeffreys-half support and uniform exception symbols",
            "semantic_truth_code_claim": False, "global_history_optimality_claim": False}

    def telemetry(self):
        return {**super().telemetry(), "controller": "pilot_mdl_description", "fitted_energy_weights": [],
            "decision_scalar_count": 0,
            "unused_base_decision_scalars": ["complexity_penalty", "switch_margin"],
            "modelled_representation": "extracted token/context/role/attribute observations",
            "base_energy_used_for_selection": False, "global_history_optimality_claim": False,
            "parameter_status": "fixed_declared_coding_choices_not_benchmark_fitted",
            "exact_code_cache_counts": dict(self._mdl_description_cache.hits),
            "exact_cache_statistics_entries": len(self._mdl_description_cache.statistics),
            "exact_cache_length_entries": len(self._mdl_description_cache.lengths)}
