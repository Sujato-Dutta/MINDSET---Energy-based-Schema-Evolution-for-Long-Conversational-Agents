"""Isolated pilot ideas 4--6, never a paper-default controller.

All obligations and certificates are conditional on the existing lexical
extractor. They establish source/representation properties, not world truth.
Future-query regret below is a source-accessibility surrogate, not QA loss.
No benchmark references, evidence labels, or model answers enter these classes.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
import re
from time import perf_counter

import numpy as np

from mindset.memory.energy import MULTI_VALUED_SLOTS, unit_vector
from mindset.methods.base import MemoryHit
from mindset.methods.joint import JointMindsetMemory, typed_features, utc_time
from mindset.methods.mindset import _render_episode_compact


@dataclass(frozen=True)
class SourceProbe:
    owner: str
    slot: str
    route: str
    context: tuple[str, ...]
    source_ids: tuple[str, ...]


def value_map(episode):
    return episode.features.attribute_map()


def same_key(left, right):
    return (left.speaker.casefold() == right.speaker.casefold()
            and left.features.primary_slot == right.features.primary_slot)


def same_value(left, right):
    slot = right.features.primary_slot
    return value_map(left).get(slot) == value_map(right).get(slot)


def ordered(left, right):
    a, b = utc_time(left.timestamp), utc_time(right.timestamp)
    return bool(a and b and a < b)


def source_obligations(memory, episode):
    """Cheap probes generated from observations, never benchmark questions.

    A current/previous pair is warranted only by a same-owner/slot explicit
    later change. Unknown context is not treated as known same context.
    Context probes are warranted only by explicitly distinct contexts.
    """
    slot = episode.features.primary_slot
    if slot == "discourse" or slot in MULTI_VALUED_SLOTS:
        return ()
    prior = [e for e in memory.episodes.values()
             if e.episode_id != episode.episode_id and same_key(e, episode)
             and ordered(e, episode)]
    if not prior:
        return ()
    old = max(prior, key=lambda e: (utc_time(e.timestamp), e.ordinal, e.episode_id))
    old_context = tuple(sorted(old.features.contexts))
    new_context = tuple(sorted(episode.features.contexts))
    if same_value(old, episode):
        return ()
    owner = episode.speaker.casefold()
    if old_context and new_context and old_context != new_context:
        return (SourceProbe(owner, slot, "contextual", old_context, (old.episode_id,)),
                SourceProbe(owner, slot, "contextual", new_context, (episode.episode_id,)))
    if episode.features.explicit_change and not (
            old_context and new_context and old_context != new_context):
        return (SourceProbe(owner, slot, "current", new_context, (episode.episode_id,)),
                SourceProbe(owner, slot, "historical", old_context, (old.episode_id,)),
                SourceProbe(owner, slot, "changes", new_context,
                            (old.episode_id, episode.episode_id)))
    return ()


def accessible_sources(memory, owner, slot, route, context=()):
    schemas = [s for s in memory.schemas.values()
               if s.topic == f"{owner}::{slot}"]
    if route == "current":
        schemas = [s for s in schemas if s.status == "active"]
    elif route == "historical":
        schemas = [s for s in schemas if s.status == "historical"]
    if context:
        schemas = [s for s in schemas if tuple(sorted(s.context_counts)) == context]
    # Exceptions remain source-retrievable for generic/change probes, but they
    # do not constitute a current/historical structural assertion.
    ids = set()
    for schema in schemas:
        ids.update(schema.support_episode_ids)
        if route == "changes":
            ids.update(schema.exception_episode_ids)
    return ids


def probe_regret(memory, probes):
    missing = ambiguity = 0.
    for probe in probes:
        available = accessible_sources(memory, probe.owner, probe.slot,
                                       probe.route, probe.context)
        missing += len(set(probe.source_ids) - available) / max(1, len(probe.source_ids))
        if probe.route in {"current", "contextual"}:
            values = {value_map(memory.episodes[e]).get(probe.slot, ()) for e in available}
            ambiguity += max(0, len(values) - 1)
    return (missing + ambiguity) / max(1, len(probes))


def bind_query(memory, query):
    """Very conservative local grammar; unsupported queries are a no-op."""
    owners = sorted({e.speaker.casefold() for e in memory.episodes.values()})
    named = [o for o in owners if re.search(r"(?<!\w)" + re.escape(o) + r"(?!\w)", query.casefold())]
    if len(named) != 1:
        return None
    q = query.casefold()
    # This pilot has no ownership parser or value-at-time executor. Do not
    # silently reinterpret a person's relative/property or a dated historical
    # query as that person's latest/previous state.
    owner = re.escape(named[0])
    if re.search(r"(?<!\w)" + owner + r"\s*[’']s\b", q):
        return None
    if not re.search(r"\b(?:does|did|is|was|has|had|can|will|would|could|should)\s+"
                     + owner + r"(?!\w)", q):
        return None
    if re.search(r"\b(?:\d{4}|\d{1,2}[-/]\d{1,2}|january|february|march|april|may|june|"
                 r"july|august|september|october|november|december|yesterday|tomorrow|"
                 r"last\s+(?:week|month|year)|\d+\s+(?:days?|weeks?|months?|years?)\s+ago)\b", q):
        return None
    slots = sorted({e.features.primary_slot for e in memory.episodes.values()
                    if e.speaker.casefold() == named[0] and e.features.primary_slot != "discourse"})
    matches = []
    for slot in slots:
        words = slot.replace("favorite_", "favorite ").replace("relationship_", "").replace("_", " ")
        if slot == "location":
            yes = bool(re.search(r"\b(live|lives|lived|living|reside|resides|location|moved)\b", q))
        elif slot == "employer":
            yes = bool(re.search(r"\b(employer|work|works|worked|working|company)\b", q))
        elif slot == "occupation":
            yes = bool(re.search(r"\b(occupation|profession|job)\b", q))
        else:
            yes = words in q
        if yes:
            matches.append(slot)
    if len(matches) != 1:
        return None
    if re.search(r"\b(previous|previously|formerly|before|used to|earlier)\b", q):
        route = "historical"
    elif re.search(r"\b(now|currently|current|latest|today)\b", q):
        route = "current"
    elif re.search(r"\b(changes|changed|change|when)\b", q):
        route = "changes"
    else:
        return None
    contexts = {tuple(sorted(e.features.contexts)) for e in memory.episodes.values()
                if e.speaker.casefold() == named[0] and e.features.primary_slot == matches[0]
                and e.features.contexts}
    found = [c for c in contexts if all(text in q for text in c)]
    if len(found) > 1:
        return None
    return named[0], matches[0], route, found[0] if found else ()


def verify_plan(memory, episode, plan):
    """Conditional source certificate, with separate context/order premises."""
    sid = plan.get("schema_id")
    source_matches = typed_features(episode.speaker, episode.text) == episode.features
    if sid is None:
        return {"valid": plan["action"] == "new" and source_matches,
                "premises": {"new_root": True, "source_extraction_matches": source_matches},
                "source_ids": [episode.episode_id], "scope": "extracted_representation_only"}
    schema = memory.schemas[sid]
    support = [memory.episodes[e] for e in schema.support_episode_ids]
    source_matches = source_matches and all(typed_features(e.speaker, e.text) == e.features for e in support)
    known = bool(support and episode.features.attributes)
    same = known and all(same_key(e, episode) for e in support)
    slot = episode.features.primary_slot
    conflict = bool(same and slot not in MULTI_VALUED_SLOTS
                    and any(not same_value(e, episode) for e in support))
    old_contexts = {tuple(sorted(e.features.contexts)) for e in support}
    new_context = tuple(sorted(episode.features.contexts))
    explicit_same = bool(new_context and old_contexts == {new_context})
    # Unknown contexts are accepted only as unknown, not as explicit equality.
    no_context_conflict = not (new_context and any(c and c != new_context for c in old_contexts))
    later = bool(support and all(ordered(e, episode) for e in support))
    independent_confirmations = len({memory.episodes[e].session_id for e in schema.exception_episode_ids
                                    if same_key(memory.episodes[e], episode)
                                    and same_value(memory.episodes[e], episode)} - {episode.session_id}) > 0
    premises = {"source_extraction_matches": source_matches,
                "same_owner_slot": bool(same), "conflicting_value": conflict,
                "strict_temporal_order": later, "explicit_same_context": explicit_same,
                "no_observed_context_conflict": no_context_conflict,
                "explicit_change_or_confirmation": bool(episode.features.explicit_change or independent_confirmations),
                "explicit_distinct_context": bool(new_context and old_contexts
                    and all(c and c != new_context for c in old_contexts))}
    mutation = plan["mutation"]
    if mutation == "merge":
        valid = bool(same and not conflict and no_context_conflict)
    elif mutation == "version":
        valid = bool(same and conflict and later and no_context_conflict
                     and premises["explicit_change_or_confirmation"])
    elif mutation == "branch":
        valid = bool(same and conflict and premises["explicit_distinct_context"])
    elif mutation == "exception":
        valid = bool(same)
    elif mutation == "create":
        valid = True
    else:
        valid = False
    return {"valid": bool(valid and source_matches), "premises": premises,
            "source_ids": [*schema.support_episode_ids, episode.episode_id],
            "scope": "extracted_representation_only"}


def _refresh(memory, schema):
    rows = [memory.episodes[e] for e in schema.support_episode_ids]
    if not rows:
        raise ValueError("Programs may not empty supporting provenance")
    total = np.asarray([e.vector for e in rows]).sum(axis=0)
    memory._sums[schema.schema_id] = total
    schema.centroid = unit_vector(total)
    schema.entity_counts = Counter()
    schema.role_counts = Counter()
    schema.context_counts = Counter()
    schema.attribute_counts = {}
    for episode in rows:
        schema._add_features(episode.features)
    schema.medoid_episode_id = min(rows, key=lambda e: (-float(e.vector @ schema.centroid), e.episode_id)).episode_id
    schema.mean_distortion = float(np.clip((1 - np.linalg.norm(total) / len(rows)) / 2, 0, 1))
    memory._latest[schema.schema_id] = max(utc_time(e.timestamp) for e in rows)


def representation_loss(memory, owner, slot):
    """Count contradictory within-state values and matching deferred reports."""
    loss = 0
    active_signatures = Counter()
    for schema in memory.schemas.values():
        if schema.topic != f"{owner}::{slot}":
            continue
        values = {value_map(memory.episodes[e]).get(slot, ()) for e in schema.support_episode_ids}
        if slot not in MULTI_VALUED_SLOTS:
            loss += max(0, len(values) - 1)
        if schema.status == "active":
            active_signatures[(tuple(sorted(schema.context_counts)), tuple(sorted(values)))] += 1
            deferred_sessions = {}
            for eid in schema.exception_episode_ids:
                row = memory.episodes[eid]
                signature = value_map(row).get(slot, ())
                if signature not in values:
                    deferred_sessions.setdefault(signature, set()).add(row.session_id)
            # Independently repeated observed reports should not remain forever
            # deferred when a valid compositional successor exists. This is a
            # representation obligation, NOT evidence the report is true.
            loss += sum(len(sessions) >= 2 for sessions in deferred_sessions.values())
        if schema.status == "historical":
            successors = [memory.schemas[s] for s in schema.successor_schema_ids]
            successor_values = {v for s in successors for e in s.support_episode_ids
                                for v in value_map(memory.episodes[e]).get(slot, ())}
            loss += sum(bool(set(value_map(memory.episodes[e]).get(slot, ())) & successor_values)
                        for e in schema.exception_episode_ids)
    return loss + sum(max(0, count - 1) for count in active_signatures.values())


def extend_program(memory, episode, plan):
    """Bounded non-atomic edit: atomic transition, reassign, compatible merge.

    Only exceptions can move; old support remains accessible in predecessor.
    Compatible schemas are archived with a successor link, never erased.
    """
    from mindset.methods.idea_transition_tools import state_for_plan
    state = state_for_plan(memory, episode, plan)
    steps = [plan["mutation"]]
    sid = plan.get("schema_id")
    record = state.operation_log.all()[-1]
    if plan["mutation"] in {"version", "branch"} and sid:
        target = state.schemas[record.resulting_schema_ids[-1]]
        old = state.schemas[sid]
        moved = [e for e in old.exception_episode_ids
                 if same_key(state.episodes[e], episode) and same_value(state.episodes[e], episode)
                 and tuple(sorted(state.episodes[e].features.contexts)) == tuple(sorted(episode.features.contexts))]
        for eid in moved:
            old.exception_episode_ids.remove(eid)
            if eid not in target.all_provenance():
                target.support_episode_ids.append(eid)
        if moved:
            steps.append({"reassign": moved, "from": sid, "to": target.schema_id})
            _refresh(state, target)
    else:
        target = state.schemas[record.resulting_schema_ids[-1]]
    if target.status == "active":
        values = {tuple(sorted(value_map(state.episodes[e]).items())) for e in target.support_episode_ids}
        others = [s for s in state.schemas.values() if s.schema_id != target.schema_id
                  and s.status == "active" and s.topic == target.topic
                  and set(s.context_counts) == set(target.context_counts)
                  and all(utc_time(state.episodes[e].timestamp) and utc_time(episode.timestamp)
                          and utc_time(state.episodes[e].timestamp) <= utc_time(episode.timestamp)
                          for e in s.support_episode_ids)
                  and {tuple(sorted(value_map(state.episodes[e]).items())) for e in s.support_episode_ids} == values]
        # One extra merge operation bounds the program, independent of history.
        if others:
            donor = deepcopy(sorted(others, key=lambda s: s.schema_id)[0])
            state.schemas[donor.schema_id] = donor
            target.support_episode_ids = list(dict.fromkeys((*target.support_episode_ids, *donor.support_episode_ids)))
            target.exception_episode_ids = list(dict.fromkeys((*target.exception_episode_ids, *donor.exception_episode_ids)))
            donor.status = "historical"
            donor.valid_until = episode.timestamp
            donor.successor_schema_ids.append(target.schema_id)
            steps.append({"merge": donor.schema_id, "into": target.schema_id})
            _refresh(state, target)
    return state, steps


class _PilotBase(JointMindsetMemory):
    idea_number = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idea_records = {}
        self._idea_query_bindings = 0
        self._idea_noops = 0
        self._idea_query_seconds = 0.
        self._idea_anchor_ids = ()
        self.last_idea_certificate = {}

    def _fallback(self, episode, reason):
        super()._ingest_episode(episode)
        self.idea_records[episode.episode_id] = {"supported": False, "reason": reason}
        self._idea_noops += 1

    def _adopt(self, state, episode, audit):
        from mindset.methods.idea_transition_tools import adopt_state
        records = dict(self.idea_records)
        records[episode.episode_id] = audit
        adopt_state(self, state)
        self.idea_records = records

    def retrieve(self, query, top_k=8):
        hits = super().retrieve(query, top_k)
        started = perf_counter()
        try:
            return self._source_obligation_hits(query, hits, top_k)
        finally:
            elapsed = perf_counter() - started
            self._query_seconds += elapsed
            self._idea_query_seconds += elapsed

    def candidate_pool(self, query):
        candidates, memberships = super().candidate_pool(query)
        binding = bind_query(self, query)
        self._idea_anchor_ids = ()
        self.last_idea_certificate = {"idea_number": self.idea_number,
            "binding_supported": binding is not None, "source_anchor_ids": [],
            "anchor_count": 0, "oracle_used": False}
        if binding is not None:
            owner, slot, route, context = binding
            sources = accessible_sources(self, owner, slot, route, context)
            rows = sorted((self.episodes[e] for e in sources),
                          key=lambda e: (-utc_time(e.timestamp), -e.ordinal, e.episode_id))
            anchors = [self._index_by_id[e.episode_id] for e in rows[:min(2, self.config.retrieval_candidate_k)]]
            protected = set(anchors)
            for anchor in anchors:
                if anchor in candidates:
                    continue
                if len(candidates) < self.config.retrieval_candidate_k:
                    candidates.append(anchor)
                else:
                    # Replace only the latest replaceable tail, preserving the
                    # direct prefix whenever the operational budget permits it.
                    replacement = next((i for i in range(len(candidates)-1, -1, -1)
                                        if candidates[i] not in protected), None)
                    if replacement is not None:
                        candidates[replacement] = anchor
            self._idea_anchor_ids = tuple(self._episode_order[i] for i in anchors if i in candidates)
            self.last_idea_certificate.update({"owner": owner, "slot": slot,
                "route": route, "context": list(context),
                "source_anchor_ids": list(self._idea_anchor_ids),
                "anchor_count": len(self._idea_anchor_ids)})
        self.last_candidate_pool = tuple(candidates)
        self.last_idea_certificate["candidate_count"] = len(candidates)
        return candidates, memberships

    def _source_obligation_hits(self, query, hits, top_k):
        if not self._idea_anchor_ids or top_k <= 0:
            return hits
        scored = {self._episode_order[i] for i in self.last_candidate_pool}
        sources = [e for e in self._idea_anchor_ids if e in scored]
        # Reserve at most two raw episodes; no schemas or proof text go to LLM.
        rows = [self.episodes[e] for e in sources[:min(2, top_k)]]
        reserved = [MemoryHit(e.episode_id, _render_episode_compact(e), 0., (e.episode_id,),
                    f"idea_{self.idea_number}_source_obligation", {"route": self.last_idea_certificate["route"]}) for e in rows]
        keep = {h.memory_id for h in reserved}
        self._idea_query_bindings += 1
        return [*reserved, *(h for h in hits if h.memory_id not in keep)][:top_k]

    def telemetry(self):
        return {**super().telemetry(), "idea_number": self.idea_number,
                "controller": {4: "source_probe_regret_pilot", 5: "bounded_compositional_edit_pilot",
                               6: "conditional_source_certificate_pilot"}[self.idea_number],
                "idea_supported_episodes": sum(r.get("supported", False) for r in self.idea_records.values()),
                "idea_noop_episodes": self._idea_noops,
                "idea_bound_queries": self._idea_query_bindings,
                "idea_extra_query_seconds": self._idea_query_seconds,
                "claim_status": "pilot_only_not_future_QA_guarantee",
                "idea_records": self.idea_records}


class FutureQueryRegretMemory(_PilotBase):
    """Idea 4: minimize structural probe loss before atomic action commitment."""
    idea_number = 4

    def _ingest_episode(self, episode):
        from mindset.methods.idea_transition_tools import feasible_plans, state_for_plan
        episode = replace(episode, features=typed_features(episode.speaker, episode.text))
        self.episodes[episode.episode_id] = episode
        probes = source_obligations(self, episode)
        if not probes:
            return self._fallback(episode, "no_source_derived_change_obligation")
        options = []
        for plan in feasible_plans(self, episode):
            state = state_for_plan(self, episode, plan)
            regret = probe_regret(state, probes)
            options.append((regret, plan["cost"], str(plan.get("schema_id")), plan["mutation"], state, plan))
        best = min(options, key=lambda x: x[:4])
        self._adopt(best[4], episode, {"supported": True, "selected": best[5],
                    "probes": [vars(p) for p in probes], "surrogate_regret": best[0],
                    "candidates": [{"plan": row[5], "surrogate_regret": row[0]} for row in options],
                    "limitation": "source_access_surrogate_not_real_future_answer_regret"})


class CompositionalEditMemory(_PilotBase):
    """Idea 5: search atomic and bounded multi-step representation edits."""
    idea_number = 5

    def _ingest_episode(self, episode):
        from mindset.methods.idea_transition_tools import feasible_plans, state_for_plan
        episode = replace(episode, features=typed_features(episode.speaker, episode.text))
        self.episodes[episode.episode_id] = episode
        if episode.features.primary_slot == "discourse":
            return self._fallback(episode, "untyped_source")
        options = []
        for plan in feasible_plans(self, episode):
            certificate = verify_plan(self, episode, plan)
            if not certificate["valid"]:
                continue
            atomic = state_for_plan(self, episode, plan)
            expanded, steps = extend_program(self, episode, plan)
            for state, program in ((atomic, [plan["mutation"]]), (expanded, steps)):
                score = representation_loss(state, episode.speaker.casefold(), episode.features.primary_slot)
                # Lexicographic minimum: constraint loss, then program length,
                # then original frozen energy; no benchmark coefficient fit.
                options.append((score, len(program), plan["cost"], str(plan.get("schema_id")), state, program, plan))
        if not options:
            return self._fallback(episode, "no_valid_edit_program")
        best = min(options, key=lambda x: x[:4])
        self._adopt(best[4], episode, {"supported": True, "selected": best[6],
                    "program": best[5], "program_length": best[1], "representation_loss": best[0],
                    "candidates": [{"plan": o[6], "program": o[5], "loss": o[0]} for o in options],
                    "limitation": "bounded_program_search_not_global_minimum"})


class ProofCarryingMemory(_PilotBase):
    """Idea 6: admit structural changes only under explicit source premises."""
    idea_number = 6

    def _ingest_episode(self, episode):
        from mindset.methods.idea_transition_tools import feasible_plans, state_for_plan
        episode = replace(episode, features=typed_features(episode.speaker, episode.text))
        self.episodes[episode.episode_id] = episode
        if episode.features.primary_slot == "discourse":
            return self._fallback(episode, "untyped_no_certificate_scope")
        options = [(plan, verify_plan(self, episode, plan)) for plan in feasible_plans(self, episode)]
        valid = [(p, c) for p, c in options if c["valid"]]
        if not valid:
            raise ValueError("Typed source has no admissible create/defer transition")
        plan, certificate = min(valid, key=lambda row: (row[0]["cost"], row[0]["mutation"], str(row[0].get("schema_id"))))
        state = state_for_plan(self, episode, plan)
        self._adopt(state, episode, {"supported": True, "selected": plan,
                    "certificate": certificate, "rejected": [{"plan": p, "certificate": c} for p, c in options if not c["valid"]],
                    "deferred": plan["mutation"] == "exception"})


def make_memory(idea, config, **kwargs):
    classes = {4: FutureQueryRegretMemory, 5: CompositionalEditMemory, 6: ProofCarryingMemory}
    return classes[idea](config, **kwargs)
