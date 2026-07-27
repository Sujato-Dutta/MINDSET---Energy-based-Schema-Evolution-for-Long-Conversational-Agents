from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np

from mindset.memory.energy import unit_vector


RetrievalMode = Literal[
    "cross_encoder_only",
    "hybrid",
    "coverage",
    "query_adaptive",
]

COMPONENT_NAMES = (
    "cross_encoder",
    "dense",
    "bm25",
    "schema",
    "entity",
    "temporal",
    "validity",
)


@dataclass(frozen=True)
class RetrievalWeights:
    cross_encoder: float = 0.50
    dense: float = 0.15
    bm25: float = 0.10
    schema: float = 0.10
    entity: float = 0.05
    temporal: float = 0.05
    validity: float = 0.05

    def __post_init__(self) -> None:
        values = self.as_dict()
        if any(value < 0 for value in values.values()):
            raise ValueError("retrieval weights must be non-negative")
        if abs(sum(values.values()) - 1.0) > 1e-6:
            raise ValueError("retrieval weights must sum to one")

    def as_dict(self) -> dict[str, float]:
        return {
            "cross_encoder": self.cross_encoder,
            "dense": self.dense,
            "bm25": self.bm25,
            "schema": self.schema,
            "entity": self.entity,
            "temporal": self.temporal,
            "validity": self.validity,
        }


@dataclass(frozen=True)
class LocalCandidate:
    episode_index: int
    episode_id: str
    text: str
    vector: np.ndarray
    timestamp_value: float
    ordinal: int
    entities: frozenset[str]
    time_bucket: str
    statuses: tuple[str, ...]
    components: Mapping[str, float]
    schema_ids: tuple[str, ...] = ()


def utility(candidate: LocalCandidate, weights: RetrievalWeights) -> float:
    return sum(
        weight * float(candidate.components.get(name, 0.0))
        for name, weight in weights.as_dict().items()
    )


def rank_candidates(
    candidates: Sequence[LocalCandidate],
    *,
    mode: RetrievalMode,
    weights: RetrievalWeights,
) -> list[LocalCandidate]:
    if mode == "cross_encoder_only":
        return sorted(
            candidates,
            key=lambda item: (
                -float(item.components["cross_encoder"]),
                item.episode_id,
            ),
        )
    return sorted(
        candidates,
        key=lambda item: (-utility(item, weights), item.episode_id),
    )


def select_candidates(
    query: str,
    candidates: Sequence[LocalCandidate],
    *,
    mode: RetrievalMode,
    weights: RetrievalWeights,
    top_k: int,
    redundancy_penalty: float = 0.15,
    new_entity_bonus: float = 0.08,
    new_time_bonus: float = 0.08,
) -> list[LocalCandidate]:
    if top_k <= 0:
        return []
    ranked = rank_candidates(candidates, mode=mode, weights=weights)
    if mode in {"cross_encoder_only", "hybrid"}:
        return ranked[:top_k]

    remaining = list(ranked)
    selected: list[LocalCandidate] = []
    seen_entities: set[str] = set()
    seen_times: set[str] = set()
    while remaining and len(selected) < top_k:
        best = max(
            remaining,
            key=lambda item: (
                utility(item, weights)
                - redundancy_penalty
                * max(
                    (
                        float(
                            np.dot(
                                unit_vector(item.vector),
                                unit_vector(chosen.vector),
                            )
                        )
                        for chosen in selected
                    ),
                    default=0.0,
                )
                + new_entity_bonus * float(bool(item.entities - seen_entities))
                + new_time_bonus * float(item.time_bucket not in seen_times),
                utility(item, weights),
                item.episode_id,
            ),
        )
        selected.append(best)
        remaining.remove(best)
        seen_entities.update(best.entities)
        seen_times.add(best.time_bucket)

    if mode == "query_adaptive":
        return adaptive_order(query, selected, weights)
    return selected


def adaptive_order(
    query: str,
    selected: Sequence[LocalCandidate],
    weights: RetrievalWeights,
) -> list[LocalCandidate]:
    kind = query_kind(query)
    if kind == "temporal":
        return sorted(
            selected,
            key=lambda item: (item.timestamp_value, item.ordinal, item.episode_id),
        )
    if kind == "current":
        return sorted(
            selected,
            key=lambda item: (
                -float("active" in item.statuses),
                -float(item.components["cross_encoder"]),
                item.episode_id,
            ),
        )
    if kind == "historical":
        return sorted(
            selected,
            key=lambda item: (
                -float(item.components["temporal"]),
                -float("historical" in item.statuses),
                -float(item.components["cross_encoder"]),
                item.episode_id,
            ),
        )
    if kind == "multi_hop":
        return chain_order(selected, weights)
    return sorted(
        selected,
        key=lambda item: (
            -float(item.components["cross_encoder"]),
            item.episode_id,
        ),
    )


def chain_order(
    selected: Sequence[LocalCandidate],
    weights: RetrievalWeights,
) -> list[LocalCandidate]:
    remaining = list(selected)
    if not remaining:
        return []
    first = max(
        remaining,
        key=lambda item: (utility(item, weights), item.episode_id),
    )
    ordered = [first]
    remaining.remove(first)
    chain_entities = set(first.entities)
    while remaining:
        next_item = max(
            remaining,
            key=lambda item: (
                len(item.entities & chain_entities),
                len(item.entities - chain_entities),
                utility(item, weights),
                item.episode_id,
            ),
        )
        ordered.append(next_item)
        remaining.remove(next_item)
        chain_entities.update(next_item.entities)
    return ordered


def query_kind(query: str) -> str:
    lowered = query.casefold()
    if re.search(r"\b(now|currently|current|latest|today|these days|at present)\b", lowered):
        return "current"
    if re.search(r"\b(previously|formerly|used to|past|earlier|originally)\b", lowered):
        return "historical"
    if re.search(
        r"\b(when|before|after|during|then|date|year|month|day|first|last|recent)\b",
        lowered,
    ):
        return "temporal"
    if re.search(
        r"\b(and|both|together|relationship|related|sequence|between)\b",
        lowered,
    ):
        return "multi_hop"
    return "factual"
