from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from mindset.memory.energy import (
    ActionName,
    EnergyEpisode,
    EnergySchema,
    TransitionLog,
    TransitionRecord,
    compute_action_energies,
    cosine,
    extract_episode_features,
    timestamp_value,
    tokenize,
    transition_signals,
    unit_vector,
)
from mindset.methods.base import ConversationEvent, MemoryHit
from mindset.retrieval.hybrid import (
    BM25Index,
    maximal_marginal_relevance,
    normalize_scores,
    ranked_indices,
    reciprocal_rank_fusion,
)
from mindset.retrieval.local_rerank import (
    LocalCandidate,
    RetrievalWeights,
    rank_candidates,
    select_candidates,
    utility,
)
from mindset.utils.hashing import stable_hash


AblationName = Literal[
    "full",
    "nearest_schema_merge",
    "no_contextual_term",
    "no_temporal_term",
    "no_hysteresis",
    "no_fragmentation_penalty",
    "no_provenance_expansion",
]


@dataclass(frozen=True)
class MindsetConfig:
    ablation: AblationName = "full"
    retrieval_mode: str = "cross_encoder_only"
    nearest_schema_k: int = 6
    retrieval_candidate_k: int = 48
    schema_retrieval_k: int = 12
    direct_pool_k: int = 24
    schema_pool_k: int = 24
    version_schema_k: int = 4
    continuity_anchor_k: int = 16
    continuity_radius: int = 2
    minimum_schema_size: int = 2
    new_schema_threshold: float = 0.43
    reinforce_threshold: float = 0.58
    switch_margin: float = 0.18
    stay_margin: float = 0.06
    fragmentation_eta: float = 0.75
    distortion_weight: float = 1.0
    contradiction_weight: float = 1.25
    historical_damage_weight: float = 0.75
    fragmentation_weight: float = 0.22
    inconsistency_weight: float = 1.0
    direct_dense_weight: float = 0.75
    direct_bm25_weight: float = 0.25
    direct_rank_weight: float = 0.20
    schema_rank_weight: float = 0.05
    continuity_rank_weight: float = 0.40
    rrf_weight: float = 0.05
    mmr_relevance_weight: float = 0.99
    rrf_constant: int = 60
    cross_encoder_weight: float = 0.5625
    local_dense_weight: float = 0.20
    local_bm25_weight: float = 0.0
    local_schema_weight: float = 0.0625
    local_entity_weight: float = 0.0375
    local_temporal_weight: float = 0.1375
    local_validity_weight: float = 0.0
    coverage_redundancy_penalty: float = 0.15
    coverage_new_entity_bonus: float = 0.08
    coverage_new_time_bonus: float = 0.04

    def __post_init__(self) -> None:
        if self.nearest_schema_k <= 0 or self.schema_retrieval_k <= 0:
            raise ValueError("schema candidate counts must be positive")
        if self.continuity_anchor_k <= 0 or self.continuity_radius <= 0:
            raise ValueError("continuity parameters must be positive")
        if self.retrieval_candidate_k <= 0:
            raise ValueError("retrieval_candidate_k must be positive")
        if self.retrieval_mode not in {
            "legacy", "cross_encoder_only", "hybrid", "coverage", "query_adaptive"
        }:
            raise ValueError("unknown retrieval_mode")
        if min(self.direct_pool_k, self.schema_pool_k, self.version_schema_k) <= 0:
            raise ValueError("local candidate-pool sizes must be positive")
        if self.minimum_schema_size <= 0:
            raise ValueError("minimum_schema_size must be positive")
        if not 0.0 <= self.new_schema_threshold <= 1.0:
            raise ValueError("new_schema_threshold must lie in [0, 1]")
        if self.switch_margin < self.stay_margin or self.stay_margin < 0:
            raise ValueError("hysteresis requires switch_margin >= stay_margin >= 0")
        if not 0.0 <= self.mmr_relevance_weight <= 1.0:
            raise ValueError("mmr_relevance_weight must lie in [0, 1]")
        self.retrieval_weights()

    def energy_weights(self) -> dict[str, float]:
        return {
            "distortion": self.distortion_weight,
            "contradiction": self.contradiction_weight,
            "historical_damage": self.historical_damage_weight,
            "fragmentation": (
                0.0 if self.ablation == "no_fragmentation_penalty"
                else self.fragmentation_weight
            ),
            "inconsistency": self.inconsistency_weight,
        }

    def retrieval_weights(self) -> RetrievalWeights:
        return RetrievalWeights(
            cross_encoder=self.cross_encoder_weight,
            dense=self.local_dense_weight,
            bm25=self.local_bm25_weight,
            schema=self.local_schema_weight,
            entity=self.local_entity_weight,
            temporal=self.local_temporal_weight,
            validity=self.local_validity_weight,
        )


class MindsetMemory:
    """Free-energy dynamics over extractive, provenance-preserving schemas."""

    method_name = "mindset"
    method_version = "free-energy-memory-v1"
    retrieval_version = "schema-union-local-cross-encoder-v1"

    def __init__(
        self,
        config: MindsetConfig | None = None,
        *,
        encoder: object | None = None,
        local_reranker: object | None = None,
        precomputed_embeddings: Mapping[str, np.ndarray] | None = None,
        transition_controller: object | None = None,
    ) -> None:
        self.config = config or MindsetConfig()
        self.transition_controller = transition_controller
        self.encoder = encoder
        self.local_reranker = local_reranker
        self.precomputed_embeddings = dict(precomputed_embeddings or {})
        self.episodes: dict[str, EnergyEpisode] = {}
        self.schemas: dict[str, EnergySchema] = {}
        self.operation_log = TransitionLog()
        self._source_events: dict[str, ConversationEvent] = {}
        self._episode_order: list[str] = []
        self._episode_vectors = np.empty((0, 0), dtype=np.float32)
        self._bm25 = BM25Index(())
        self._ingest_seconds = 0.0
        self._query_seconds = 0.0
        self._query_count = 0

    def ingest(self, events: Sequence[ConversationEvent]) -> None:
        if not events:
            return
        duplicates = [event.event_id for event in events if event.event_id in self.episodes]
        duplicates.extend(i for i,n in Counter(e.event_id for e in events).items() if n>1)
        if duplicates:
            raise ValueError(f"Episodes are immutable; duplicate IDs: {duplicates}")
        started = perf_counter()
        rendered = [_render_event(event) for event in events]
        if all(event.event_id in self.precomputed_embeddings for event in events):
            vectors = np.stack(
                [self.precomputed_embeddings[event.event_id] for event in events]
            ).astype(np.float32)
        else:
            vectors = self._encode(rendered)
        for event, vector in zip(events, vectors, strict=True):
            features = extract_episode_features(event.speaker, event.text)
            episode = EnergyEpisode(
                episode_id=event.event_id,
                conversation_id=event.conversation_id,
                session_id=event.session_id,
                timestamp=event.timestamp,
                speaker=event.speaker,
                text=event.text,
                vector=unit_vector(vector),
                features=features,
                ordinal=len(self._episode_order),
                metadata=dict(event.metadata),
            )
            self.episodes[episode.episode_id] = episode
            self._source_events[episode.episode_id] = event
            self._episode_order.append(episode.episode_id)
            self._ingest_episode(episode)
        self._rebuild_retrieval_index()
        self._ingest_seconds += perf_counter() - started

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.encoder is not None:
            return np.asarray(
                self.encoder.encode(
                    list(texts),
                    batch_size=64,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
        vectors = np.zeros((len(texts), 512), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % vectors.shape[1]
                sign = 1.0 if digest[4] & 1 else -1.0
                vectors[row, index] += sign
            vectors[row] = unit_vector(vectors[row])
        return vectors

    def _ingest_episode(self, episode: EnergyEpisode) -> None:
        active = [schema for schema in self.schemas.values() if schema.status == "active"]
        if not active:
            schema = self._create_schema(episode, action="new")
            self._record_transition(episode, "new", None, (schema.schema_id,), {}, None)
            return

        candidates = self._nearest_schemas(active, episode)
        evaluated = [
            (schema, transition_signals(schema, episode, self.episodes))
            for schema in candidates
        ]
        target, signals = max(
            evaluated,
            key=lambda item: (item[1].compatibility, item[1].dense_similarity, item[0].schema_id),
        )
        tiny = sum(
            len(schema.support_episode_ids) < self.config.minimum_schema_size
            for schema in active
        )
        energies = compute_action_energies(
            target,
            episode,
            self.episodes,
            signals,
            active_schema_count=len(active),
            episode_count=len(self.episodes),
            tiny_schema_count=tiny,
            weights=self.config.energy_weights(),
            fragmentation_eta=self.config.fragmentation_eta,
            minimum_schema_size=self.config.minimum_schema_size,
            use_temporal=self.config.ablation != "no_temporal_term",
            use_contextual=self.config.ablation != "no_contextual_term",
        )
        operation, hysteresis_applied = self._choose_transition(signals, energies)
        stored_as_exception = False
        resulting: tuple[str, ...]
        if operation == "reinforce":
            if signals.contradiction_probability >= 0.45:
                target.add_exception(episode)
                stored_as_exception = True
            else:
                target.add_support(episode, self.episodes)
            resulting = (target.schema_id,)
        elif operation == "supersede":
            target.status = "historical"
            target.valid_until = episode.timestamp
            successor = self._create_schema(
                episode,
                action="supersede",
                predecessor_schema_id=target.schema_id,
            )
            target.successor_schema_ids.append(successor.schema_id)
            resulting = (target.schema_id, successor.schema_id)
        elif operation == "split":
            split = self._create_schema(episode, action="split")
            resulting = (target.schema_id, split.schema_id)
        else:
            created = self._create_schema(episode, action="new")
            resulting = (created.schema_id,)
        self._record_transition(
            episode,
            operation,
            target.schema_id,
            resulting,
            energies,
            signals,
            hysteresis_applied=hysteresis_applied,
            stored_as_exception=stored_as_exception,
        )

    def _nearest_schemas(
        self,
        active: Sequence[EnergySchema],
        episode: EnergyEpisode,
    ) -> list[EnergySchema]:
        topic = f"{episode.speaker.casefold()}::{episode.features.primary_slot}"
        same_topic = [schema for schema in active if schema.topic == topic]
        pool = same_topic or list(active)
        # The local controller retrieves only a bounded nearest-state set. The
        # recent cap prevents unbounded quadratic writes in highly fragmented
        # discourse; the energy computation still sees the nearest local states.
        cap = max(64, self.config.nearest_schema_k * 8)
        if len(pool) > cap:
            pool = pool[-cap:]
        matrix = np.stack([schema.centroid for schema in pool])
        scores = matrix @ episode.vector
        indices = np.argsort(-scores, kind="stable")[: self.config.nearest_schema_k]
        return [pool[int(index)] for index in indices]

    def _choose_transition(self, signals, energies) -> tuple[ActionName, bool]:
        if self.transition_controller is not None:
            return self.transition_controller(signals, self.config)
        if self.config.ablation == "nearest_schema_merge":
            return "reinforce", False
        if (
            signals.compatibility < self.config.new_schema_threshold
            and not signals.topic_match
            and signals.dense_similarity < 0.35
        ):
            return "new", False

        feasible: set[ActionName] = {"reinforce", "new"}
        if signals.contradiction_probability >= 0.35:
            feasible.add("supersede")
        if signals.contextual_potential >= 0.12:
            feasible.add("split")
        if signals.compatibility >= self.config.new_schema_threshold:
            feasible.discard("new")
        if (
            signals.contradiction_probability < 0.15
            and signals.contextual_potential < 0.12
            and signals.compatibility >= self.config.reinforce_threshold
        ):
            return "reinforce", False
        if (
            self.config.ablation != "no_hysteresis"
            and signals.contradiction_probability >= 0.60
            and not signals.recurrence
            and not signals.explicit_change
        ):
            return "reinforce", True

        current = energies["reinforce"].total
        alternatives = [action for action in feasible if action != "reinforce"]
        if not alternatives:
            return "reinforce", False
        best = min(alternatives, key=lambda action: (energies[action].total, action))
        if self.config.ablation == "no_hysteresis":
            return min(feasible, key=lambda action: (energies[action].total, action)), False

        margin = self.config.switch_margin
        if signals.recurrence:
            margin *= 1.0 - 0.6 * signals.recurrence
        if signals.temporal_potential > signals.contextual_potential and signals.temporal_potential > 0.45:
            margin *= 0.55
        if energies[best].total < current - margin:
            return best, True
        if current <= energies[best].total + self.config.stay_margin:
            return "reinforce", True
        return best, True

    def _create_schema(
        self,
        episode: EnergyEpisode,
        *,
        action: ActionName,
        predecessor_schema_id: str | None = None,
    ) -> EnergySchema:
        identifier = "fes-" + stable_hash(
            (episode.episode_id, action, predecessor_schema_id, len(self.schemas))
        )[:16]
        topic = f"{episode.speaker.casefold()}::{episode.features.primary_slot}"
        schema = EnergySchema(
            schema_id=identifier,
            topic=topic,
            centroid=episode.vector.copy(),
            medoid_episode_id=episode.episode_id,
            support_episode_ids=[episode.episode_id],
            exception_episode_ids=[],
            valid_from=episode.timestamp,
            predecessor_schema_id=predecessor_schema_id,
        )
        schema._add_features(episode.features)
        self.schemas[identifier] = schema
        return schema

    def _record_transition(
        self,
        episode: EnergyEpisode,
        operation: ActionName,
        target_schema_id: str | None,
        resulting_schema_ids: tuple[str, ...],
        energies: Mapping[ActionName, Any],
        signals: Any,
        *,
        hysteresis_applied: bool = False,
        stored_as_exception: bool = False,
    ) -> None:
        self.operation_log.append(
            TransitionRecord(
                episode_id=episode.episode_id,
                operation=operation,
                target_schema_id=target_schema_id,
                resulting_schema_ids=resulting_schema_ids,
                candidate_energies={
                    action: float(terms.total) for action, terms in energies.items()
                },
                energy_terms={
                    action: terms.as_dict() for action, terms in energies.items()
                },
                signals=(asdict(signals) if signals is not None else {}),
                hysteresis_applied=hysteresis_applied,
                stored_as_exception=stored_as_exception,
            )
        )

    def _rebuild_retrieval_index(self) -> None:
        if not self._episode_order:
            self._episode_vectors = np.empty((0, 0), dtype=np.float32)
            self._bm25 = BM25Index(())
            return
        self._episode_vectors = np.stack(
            [self.episodes[identifier].vector for identifier in self._episode_order]
        ).astype(np.float32)
        self._bm25 = BM25Index(
            [_render_episode(self.episodes[identifier]) for identifier in self._episode_order]
        )
        # Immutable within a build: avoid rebuilding these maps for every query.
        self._index_by_id = {
            identifier: index for index, identifier in enumerate(self._episode_order)
        }
        self._schema_provenance = {
            schema.schema_id: schema.all_provenance() for schema in self.schemas.values()
        }

    def retrieve(self, query: str, top_k: int = 10) -> list[MemoryHit]:
        if self.config.retrieval_mode == "legacy":
            return self._retrieve_legacy(query, top_k)
        candidates = self.retrieve_candidate_pool(query)
        return self.select_candidate_pool(
            query,
            candidates,
            mode=self.config.retrieval_mode,
            top_k=top_k,
        )

    def retrieve_candidate_pool(self, query: str) -> list[LocalCandidate]:
        """Build the raw-episode union and score every pair entirely locally."""
        if not self._episode_order:
            return []
        if self.local_reranker is None:
            raise RuntimeError(
                "MINDSET local reranking requires a CPU cross-encoder; "
                "use retrieval_mode='legacy' only for legacy diagnostics"
            )
        expanded_query = _expand_query(query)
        query_vector = unit_vector(self._encode([expanded_query])[0])
        dense_raw = self._episode_vectors @ query_vector
        bm25_raw = self._bm25.scores(expanded_query)
        dense_global = normalize_scores(dense_raw)
        bm25_global = normalize_scores(bm25_raw)
        direct_score = 0.5 * dense_global + 0.5 * bm25_global
        direct_rank = ranked_indices(
            direct_score,
            min(self.config.direct_pool_k, len(self._episode_order)),
        )

        route = _query_route(query)
        query_tokens = set(tokenize(expanded_query))
        schema_rows: list[tuple[EnergySchema, float]] = []
        schema_scores_by_episode: dict[int, float] = {}
        schema_ids_by_episode: dict[int, list[str]] = {}
        index_by_id = self._index_by_id
        if self.config.ablation != "no_provenance_expansion":
            for schema in self.schemas.values():
                medoid = self.episodes[schema.medoid_episode_id]
                medoid_tokens = set(medoid.features.tokens)
                lexical = len(query_tokens & medoid_tokens) / max(
                    1, len(query_tokens | medoid_tokens)
                )
                entity = _entity_query_score(
                    query_tokens, schema.entity_counts, schema.role_counts
                )
                temporal = _schema_temporal_score(schema, route)
                score = (
                    0.58 * max(0.0, cosine(query_vector, schema.centroid))
                    + 0.18 * lexical
                    + 0.14 * entity
                    + 0.10 * temporal
                )
                schema_rows.append((schema, float(score)))
            schema_rows.sort(key=lambda item: (-item[1], item[0].schema_id))
            for schema, score in schema_rows:
                for identifier in self._schema_provenance[schema.schema_id]:
                    index = index_by_id[identifier]
                    if score > schema_scores_by_episode.get(index, -1.0):
                        schema_scores_by_episode[index] = score
                    schema_ids_by_episode.setdefault(index, []).append(schema.schema_id)

        schema_rank = sorted(
            schema_scores_by_episode,
            key=lambda index: (
                -(
                    0.70 * schema_scores_by_episode[index]
                    + 0.20 * float(dense_global[index])
                    + 0.10 * float(bm25_global[index])
                ),
                index,
            ),
        )[: self.config.schema_pool_k]

        version_indices: list[int] = []
        for status in ("active", "historical"):
            relevant = [item for item in schema_rows if item[0].status == status]
            for schema, _ in relevant[: self.config.version_schema_k]:
                version_indices.append(index_by_id[schema.medoid_episode_id])
        candidate_indices = list(
            dict.fromkeys((*direct_rank, *schema_rank, *version_indices))
        )
        if not candidate_indices:
            return []

        raw_components: dict[str, np.ndarray] = {
            "dense": np.asarray([dense_raw[index] for index in candidate_indices]),
            "bm25": np.asarray([bm25_raw[index] for index in candidate_indices]),
            "schema": np.asarray(
                [schema_scores_by_episode.get(index, 0.0) for index in candidate_indices]
            ),
            "entity": np.asarray(
                [
                    _episode_entity_overlap(query_tokens, self.episodes[self._episode_order[index]])
                    for index in candidate_indices
                ]
            ),
            "temporal": np.asarray(
                [
                    _episode_query_temporal_score(
                        query,
                        self.episodes[self._episode_order[index]],
                        len(self._episode_order),
                    )
                    for index in candidate_indices
                ]
            ),
            "validity": np.asarray(
                [
                    _episode_validity_score(
                        route,
                        tuple(
                            sorted(
                                {
                                    self.schemas[schema_id].status
                                    for schema_id in schema_ids_by_episode.get(index, ())
                                }
                            )
                        ),
                    )
                    for index in candidate_indices
                ]
            ),
        }
        documents = [
            _render_episode_compact(self.episodes[self._episode_order[index]])
            for index in candidate_indices
        ]
        raw_components["cross_encoder"] = np.asarray(
            self.local_reranker.predict(
                [(query, document) for document in documents],
                batch_size=32,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        normalized = {
            name: normalize_scores(values) for name, values in raw_components.items()
        }
        candidates: list[LocalCandidate] = []
        for position, index in enumerate(candidate_indices):
            identifier = self._episode_order[index]
            episode = self.episodes[identifier]
            statuses = tuple(
                sorted(
                    {
                        self.schemas[schema_id].status
                        for schema_id in schema_ids_by_episode.get(index, ())
                    }
                )
            )
            candidates.append(
                LocalCandidate(
                    episode_index=index,
                    episode_id=identifier,
                    text=documents[position],
                    vector=episode.vector,
                    timestamp_value=timestamp_value(episode.timestamp),
                    ordinal=episode.ordinal,
                    entities=frozenset(_coverage_entities(episode)),
                    time_bucket=episode.session_id,
                    statuses=statuses,
                    components={
                        name: float(values[position])
                        for name, values in normalized.items()
                    },
                    schema_ids=tuple(
                        dict.fromkeys(schema_ids_by_episode.get(index, ()))
                    ),
                )
            )
        return candidates

    def select_candidate_pool(
        self,
        query: str,
        candidates: Sequence[LocalCandidate],
        *,
        mode: str,
        top_k: int,
        weights: RetrievalWeights | None = None,
        redundancy_penalty: float | None = None,
        new_entity_bonus: float | None = None,
        new_time_bonus: float | None = None,
    ) -> list[MemoryHit]:
        if mode == "legacy":
            return self._retrieve_legacy(query, top_k)
        selected = select_candidates(
            query,
            candidates,
            mode=mode,
            weights=weights or self.config.retrieval_weights(),
            top_k=top_k,
            redundancy_penalty=(
                self.config.coverage_redundancy_penalty
                if redundancy_penalty is None else redundancy_penalty
            ),
            new_entity_bonus=(
                self.config.coverage_new_entity_bonus
                if new_entity_bonus is None else new_entity_bonus
            ),
            new_time_bonus=(
                self.config.coverage_new_time_bonus
                if new_time_bonus is None else new_time_bonus
            ),
        )
        active_weights = weights or self.config.retrieval_weights()
        return [
            MemoryHit(
                memory_id=item.episode_id,
                text=item.text,
                score=(
                    float(item.components["cross_encoder"])
                    if mode == "cross_encoder_only" else utility(item, active_weights)
                ),
                source_event_ids=(item.episode_id,),
                memory_type="locally_reranked_raw_episode",
                metadata={
                    "components": dict(item.components),
                    "schema_statuses": item.statuses,
                    "retrieval_mode": mode,
                    "retrieval_version": self.retrieval_version,
                    "reranker_model": "cross-encoder/ms-marco-MiniLM-L6-v2",
                },
            )
            for item in selected
        ]

    def ranked_candidate_pool(
        self,
        candidates: Sequence[LocalCandidate],
        *,
        mode: str,
        weights: RetrievalWeights | None = None,
    ) -> list[LocalCandidate]:
        return rank_candidates(
            candidates,
            mode=mode,
            weights=weights or self.config.retrieval_weights(),
        )

    def _retrieve_legacy(self, query: str, top_k: int = 10) -> list[MemoryHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self._episode_order:
            return []
        started = perf_counter()
        expanded_query = _expand_query(query)
        query_vector = unit_vector(self._encode([expanded_query])[0])
        dense = normalize_scores(self._episode_vectors @ query_vector)
        bm25 = normalize_scores(self._bm25.scores(expanded_query))
        direct_scores = (
            self.config.direct_dense_weight * dense
            + self.config.direct_bm25_weight * bm25
        )
        route = _query_route(query)
        memberships = self._schema_memberships()
        direct_scores = direct_scores + self._state_bias(route, memberships)
        direct_ranking = ranked_indices(
            direct_scores,
            min(self.config.retrieval_candidate_k, len(self._episode_order)),
        )
        continuity_rank, continuity_scores, continuity_anchors = self._continuity_channel(
            direct_ranking, direct_scores
        )

        schema_scores_by_episode: dict[int, float] = {}
        schema_rank: list[int] = []
        schema_metadata: dict[int, list[str]] = {}
        if self.config.ablation != "no_provenance_expansion":
            eligible = self._eligible_schemas(route)
            scored_schemas: list[tuple[EnergySchema, float]] = []
            query_tokens = set(tokenize(expanded_query))
            for schema in eligible:
                medoid = self.episodes[schema.medoid_episode_id]
                medoid_tokens = set(medoid.features.tokens)
                lexical = len(query_tokens & medoid_tokens) / max(1, len(query_tokens | medoid_tokens))
                entity = _entity_query_score(query_tokens, schema.entity_counts, schema.role_counts)
                temporal = _schema_temporal_score(schema, route)
                score = (
                    0.58 * max(0.0, cosine(query_vector, schema.centroid))
                    + 0.18 * lexical
                    + 0.14 * entity
                    + 0.10 * temporal
                )
                scored_schemas.append((schema, score))
            scored_schemas.sort(key=lambda item: (-item[1], item[0].schema_id))
            scored_schemas = scored_schemas[: self.config.schema_retrieval_k]
            if scored_schemas:
                schema_values = normalize_scores(
                    np.asarray([score for _, score in scored_schemas], dtype=np.float32)
                )
                index_by_id = {identifier: index for index, identifier in enumerate(self._episode_order)}
                for (schema, _), normalized_schema_score in zip(
                    scored_schemas, schema_values, strict=True
                ):
                    for identifier in schema.all_provenance():
                        index = index_by_id[identifier]
                        episode = self.episodes[identifier]
                        temporal = _episode_temporal_score(episode, route, len(self._episode_order))
                        entity = _episode_entity_score(set(tokenize(query)), episode)
                        score = (
                            0.38 * dense[index]
                            + 0.24 * bm25[index]
                            + 0.23 * float(normalized_schema_score)
                            + 0.10 * temporal
                            + 0.05 * entity
                        )
                        if score > schema_scores_by_episode.get(index, -1.0):
                            schema_scores_by_episode[index] = float(score)
                        schema_metadata.setdefault(index, []).append(schema.schema_id)
                schema_rank = sorted(
                    schema_scores_by_episode,
                    key=lambda index: (-schema_scores_by_episode[index], index),
                )[: self.config.retrieval_candidate_k]

        fused = reciprocal_rank_fusion(
            [ranking for ranking in (direct_ranking, schema_rank, continuity_rank) if ranking],
            rank_constant=self.config.rrf_constant,
        )
        candidate_indices = list(
            dict.fromkeys((*direct_ranking, *schema_rank, *continuity_rank))
        )
        fused_array = np.asarray([fused.get(index, 0.0) for index in candidate_indices])
        fused_normalized = normalize_scores(fused_array)
        direct_reciprocal = {
            index: 1.0 / rank for rank, index in enumerate(direct_ranking, start=1)
        }
        schema_reciprocal = {
            index: 1.0 / rank for rank, index in enumerate(schema_rank, start=1)
        }
        continuity_reciprocal = {
            index: 1.0 / rank for rank, index in enumerate(continuity_rank, start=1)
        }
        relevance = {
            index: float(
                self.config.direct_dense_weight * dense[index]
                + self.config.direct_bm25_weight * bm25[index]
                + self.config.direct_rank_weight * direct_reciprocal.get(index, 0.0)
                + self.config.schema_rank_weight * schema_reciprocal.get(index, 0.0)
                + self.config.continuity_rank_weight
                * continuity_reciprocal.get(index, 0.0)
                + self.config.rrf_weight * fused_normalized[position]
            )
            for position, index in enumerate(candidate_indices)
        }
        selected = maximal_marginal_relevance(
            candidate_indices,
            relevance,
            self._episode_vectors,
            top_k=min(top_k, len(candidate_indices)),
            relevance_weight=self.config.mmr_relevance_weight,
        )
        direct_positions = {index: rank for rank, index in enumerate(direct_ranking, start=1)}
        schema_positions = {index: rank for rank, index in enumerate(schema_rank, start=1)}
        continuity_positions = {
            index: rank for rank, index in enumerate(continuity_rank, start=1)
        }
        hits: list[MemoryHit] = []
        for index in selected:
            identifier = self._episode_order[index]
            episode = self.episodes[identifier]
            schema_ids = tuple(dict.fromkeys(schema_metadata.get(index, ())))
            hits.append(
                MemoryHit(
                    memory_id=identifier,
                    text=_render_episode(episode),
                    score=relevance[index],
                    source_event_ids=(identifier,),
                    memory_type="energy_provenance_episode" if schema_ids else "episodic",
                    metadata={
                        "dense_score": float(dense[index]),
                        "bm25_score": float(bm25[index]),
                        "direct_rank": direct_positions.get(index),
                        "schema_rank": schema_positions.get(index),
                        "continuity_rank": continuity_positions.get(index),
                        "continuity_anchor": (
                            self._episode_order[continuity_anchors[index]]
                            if index in continuity_anchors else None
                        ),
                        "schema_ids": schema_ids,
                        "schema_statuses": sorted(
                            {self.schemas[item].status for item in schema_ids}
                        ),
                        "retrieval_version": self.retrieval_version,
                    },
                )
            )
        self._query_seconds += perf_counter() - started
        self._query_count += 1
        return hits

    def _continuity_channel(
        self,
        direct_ranking: Sequence[int],
        direct_scores: np.ndarray,
    ) -> tuple[list[int], dict[int, float], dict[int, int]]:
        proposals: dict[int, float] = {}
        anchors: dict[int, int] = {}
        for anchor in direct_ranking[: self.config.continuity_anchor_k]:
            anchor_episode = self.episodes[self._episode_order[anchor]]
            for distance in range(1, self.config.continuity_radius + 1):
                for index in (anchor - distance, anchor + distance):
                    if not 0 <= index < len(self._episode_order):
                        continue
                    episode = self.episodes[self._episode_order[index]]
                    if episode.session_id != anchor_episode.session_id:
                        continue
                    score = float(
                        0.78 * direct_scores[anchor]
                        + 0.22 * direct_scores[index]
                        - 0.04 * (distance - 1)
                    )
                    if score > proposals.get(index, -1.0):
                        proposals[index] = score
                        anchors[index] = anchor
        ranking = sorted(proposals, key=lambda index: (-proposals[index], index))
        return ranking[: self.config.retrieval_candidate_k], proposals, anchors

    def _eligible_schemas(self, route: str) -> list[EnergySchema]:
        schemas = list(self.schemas.values())
        if route == "current":
            return [schema for schema in schemas if schema.status == "active"]
        if route == "historical":
            historical = [schema for schema in schemas if schema.status == "historical"]
            return historical or schemas
        return schemas

    def _schema_memberships(self) -> dict[str, set[str]]:
        memberships: dict[str, set[str]] = {}
        for schema in self.schemas.values():
            for identifier in schema.all_provenance():
                memberships.setdefault(identifier, set()).add(schema.status)
        return memberships

    def _state_bias(self, route: str, memberships: Mapping[str, set[str]]) -> np.ndarray:
        result = np.zeros(len(self._episode_order), dtype=np.float32)
        if route not in {"current", "historical"}:
            return result
        wanted = "active" if route == "current" else "historical"
        unwanted = "historical" if route == "current" else "active"
        for index, identifier in enumerate(self._episode_order):
            states = memberships.get(identifier, set())
            result[index] += 0.10 * float(wanted in states)
            result[index] -= 0.05 * float(unwanted in states and wanted not in states)
        return result

    def snapshot(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": self.method_name,
            "method_version": self.method_version,
            "retrieval_version": self.retrieval_version,
            "config": asdict(self.config),
            "llm_calls_during_ingestion": 0,
            "episodes": [
                {
                    "episode_id": episode.episode_id,
                    "conversation_id": episode.conversation_id,
                    "session_id": episode.session_id,
                    "timestamp": episode.timestamp,
                    "speaker": episode.speaker,
                    "features": {
                        "entities": episode.features.entities,
                        "roles": episode.features.roles,
                        "contexts": episode.features.contexts,
                        "attributes": episode.features.attributes,
                        "explicit_change": episode.features.explicit_change,
                    },
                }
                for episode in (self.episodes[item] for item in self._episode_order)
            ],
            "schemas": [
                schema.snapshot() for schema in sorted(self.schemas.values(), key=lambda item: item.schema_id)
            ],
            "transitions": [
                {
                    "episode_id": record.episode_id,
                    "operation": record.operation,
                    "target_schema_id": record.target_schema_id,
                    "resulting_schema_ids": record.resulting_schema_ids,
                    "candidate_energies": record.candidate_energies,
                    "selected_energy_terms": record.energy_terms.get(record.operation, {}),
                    "signals": record.signals,
                    "hysteresis_applied": record.hysteresis_applied,
                    "stored_as_exception": record.stored_as_exception,
                }
                for record in self.operation_log.all()
            ],
            "telemetry": self.telemetry(),
        }
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output

    def telemetry(self) -> dict[str, Any]:
        operations = Counter(record.operation for record in self.operation_log.all())
        active = [schema for schema in self.schemas.values() if schema.status == "active"]
        historical = [schema for schema in self.schemas.values() if schema.status == "historical"]
        term_totals: Counter[str] = Counter()
        term_count = 0
        for record in self.operation_log.all():
            if record.operation in record.energy_terms:
                for name, value in record.energy_terms[record.operation].items():
                    term_totals[name] += value
                term_count += 1
        return {
            "method_version": self.method_version,
            "retrieval_version": self.retrieval_version,
            "ablation": self.config.ablation,
            "episode_count": len(self.episodes),
            "schema_count": len(self.schemas),
            "active_schema_count": len(active),
            "historical_schema_count": len(historical),
            "exception_count": sum(len(schema.exception_episode_ids) for schema in self.schemas.values()),
            "operation_counts": dict(sorted(operations.items())),
            "mean_selected_energy_terms": {
                name: value / max(1, term_count) for name, value in sorted(term_totals.items())
            },
            "ingest_seconds": self._ingest_seconds,
            "mean_query_seconds": self._query_seconds / max(1, self._query_count),
            "llm_ingestion_calls": 0,
            "llm_retrieval_calls": 0,
        }

    def validate_invariants(self) -> dict[str, int | bool]:
        episode_ids = set(self.episodes)
        errors: list[str] = []
        provenance_counts: Counter[str] = Counter()
        for schema in self.schemas.values():
            provenance = schema.all_provenance()
            missing = set(provenance) - episode_ids
            if missing:
                errors.append(f"{schema.schema_id} has missing provenance: {sorted(missing)}")
            if len(provenance) != len(set(provenance)):
                errors.append(f"{schema.schema_id} contains duplicate provenance")
            if schema.medoid_episode_id not in schema.support_episode_ids:
                errors.append(f"{schema.schema_id} medoid is not supporting evidence")
            if schema.status == "historical" and (
                schema.valid_until is None or not schema.successor_schema_ids
            ):
                errors.append(f"historical schema {schema.schema_id} lacks a successor interval")
            if schema.predecessor_schema_id:
                predecessor = self.schemas.get(schema.predecessor_schema_id)
                if predecessor is None or schema.schema_id not in predecessor.successor_schema_ids:
                    errors.append(f"broken predecessor edge for {schema.schema_id}")
            provenance_counts.update(provenance)
        transitioned = Counter(record.episode_id for record in self.operation_log.all())
        if set(transitioned) != episode_ids or any(value != 1 for value in transitioned.values()):
            errors.append("every episode must induce exactly one transition")
        if errors:
            raise ValueError("Memory invariant violation: " + "; ".join(errors))
        return {
            "valid": True,
            "episode_count": len(episode_ids),
            "schema_count": len(self.schemas),
            "transition_count": len(self.operation_log.all()),
            "grounded_episode_count": len(provenance_counts),
        }


def _render_event(event: ConversationEvent) -> str:
    return f"[{event.timestamp}] {event.speaker}: {event.text}"


def _render_episode(episode: EnergyEpisode) -> str:
    return f"[{episode.timestamp}] {episode.speaker}: {episode.text}"


def _render_episode_compact(episode: EnergyEpisode) -> str:
    return f"[{episode.timestamp}] | Speaker: {episode.speaker}\n{episode.text}"


def _query_route(query: str) -> str:
    lowered = query.casefold()
    if re.search(r"\b(before|previously|formerly|used to|past|earlier|originally)\b", lowered):
        return "historical"
    if re.search(r"\b(now|currently|current|latest|today|these days|at present)\b", lowered):
        return "current"
    if re.search(r"\b(at work|at home|weekends?|weekdays?|context|depending)\b", lowered):
        return "contextual"
    return "generic"


QUERY_EXPANSIONS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("book", "novel", "read"), ("book", "novel", "read", "author", "title")),
    (("travel", "visit", "country"), ("travel", "visited", "trip", "country", "flight")),
    (("job", "work", "employer"), ("job", "work", "worked", "employer", "office")),
    (("live", "location", "where"), ("live", "lived", "moved", "location", "home")),
    (("prefer", "favorite", "like"), ("prefer", "favorite", "like", "love", "choice")),
)


def _expand_query(query: str) -> str:
    tokens = set(tokenize(query))
    additions: list[str] = []
    for triggers, expansion in QUERY_EXPANSIONS:
        if tokens.intersection(triggers):
            additions.extend(expansion)
    return query + (" " + " ".join(dict.fromkeys(additions)) if additions else "")


def _entity_query_score(
    query_tokens: set[str],
    entities: Counter[str],
    roles: Counter[str],
) -> float:
    values = set(entities) | set(roles)
    return len(query_tokens & values) / max(1, len(query_tokens & values) + 1)


def _episode_entity_score(query_tokens: set[str], episode: EnergyEpisode) -> float:
    values = set(episode.features.entities) | set(episode.features.roles)
    return float(bool(query_tokens & values))


def _episode_entity_overlap(
    query_tokens: set[str], episode: EnergyEpisode
) -> float:
    values: set[str] = set()
    for value in (*episode.features.entities, *episode.features.roles):
        values.update(tokenize(value))
    if not values:
        return 0.0
    return len(query_tokens & values) / len(values | query_tokens)


def _coverage_entities(episode: EnergyEpisode) -> set[str]:
    values = set(episode.features.entities[1:]) | set(episode.features.roles)
    return {value for value in values if value}


def _episode_query_temporal_score(
    query: str,
    episode: EnergyEpisode,
    episode_count: int,
) -> float:
    query_years = [int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", query)]
    episode_years = [
        int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", episode.timestamp)
    ]
    if query_years and episode_years:
        distance = min(abs(left - right) for left in query_years for right in episode_years)
        return 1.0 / (1.0 + distance)
    return _episode_temporal_score(episode, _query_route(query), episode_count)


def _episode_validity_score(route: str, statuses: Sequence[str]) -> float:
    values = set(statuses)
    if route == "current":
        return 1.0 if "active" in values else 0.0
    if route == "historical":
        return 1.0 if "historical" in values else 0.25
    if "active" in values:
        return 1.0
    if "historical" in values:
        return 0.60
    return 0.50


def _schema_temporal_score(schema: EnergySchema, route: str) -> float:
    if route == "current":
        return float(schema.status == "active")
    if route == "historical":
        return float(schema.status == "historical")
    return 0.5


def _episode_temporal_score(
    episode: EnergyEpisode,
    route: str,
    episode_count: int,
) -> float:
    relative = episode.ordinal / max(1, episode_count - 1)
    if route == "current":
        return relative
    if route == "historical":
        return 1.0 - relative
    return 0.5
