from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Mapping, Sequence

import numpy as np
from dateutil import parser as date_parser


ActionName = Literal["reinforce", "supersede", "split", "new"]


@dataclass(frozen=True)
class EpisodeFeatures:
    entities: tuple[str, ...]
    roles: tuple[str, ...]
    contexts: tuple[str, ...]
    attributes: tuple[tuple[str, str], ...]
    tokens: tuple[str, ...]
    explicit_change: bool = False

    @property
    def primary_slot(self) -> str:
        return self.attributes[0][0] if self.attributes else "discourse"

    def attribute_map(self) -> dict[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = {}
        for slot, value in self.attributes:
            grouped.setdefault(slot, []).append(value)
        return {slot: tuple(values) for slot, values in grouped.items()}


@dataclass(frozen=True)
class EnergyEpisode:
    episode_id: str
    conversation_id: str
    session_id: str
    timestamp: str
    speaker: str
    text: str
    vector: np.ndarray
    features: EpisodeFeatures
    ordinal: int
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass
class EnergySchema:
    schema_id: str
    topic: str
    centroid: np.ndarray
    medoid_episode_id: str
    support_episode_ids: list[str]
    exception_episode_ids: list[str]
    valid_from: str
    valid_until: str | None = None
    status: Literal["active", "historical"] = "active"
    predecessor_schema_id: str | None = None
    successor_schema_ids: list[str] = field(default_factory=list)
    entity_counts: Counter[str] = field(default_factory=Counter)
    role_counts: Counter[str] = field(default_factory=Counter)
    context_counts: Counter[str] = field(default_factory=Counter)
    attribute_counts: dict[str, Counter[str]] = field(default_factory=dict)
    mean_distortion: float = 0.0

    def all_provenance(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.support_episode_ids, *self.exception_episode_ids)))

    def add_support(
        self,
        episode: EnergyEpisode,
        episodes: Mapping[str, EnergyEpisode],
    ) -> None:
        if episode.episode_id in self.all_provenance():
            return
        count = len(self.support_episode_ids)
        old_centroid = self.centroid
        new_centroid = unit_vector(old_centroid * count + episode.vector)
        drift = 1.0 - cosine(old_centroid, new_centroid)
        new_distance = 1.0 - cosine(episode.vector, new_centroid)
        self.mean_distortion = (
            count * (self.mean_distortion + drift) + new_distance
        ) / (count + 1)
        self.centroid = new_centroid
        self.support_episode_ids.append(episode.episode_id)
        self._add_features(episode.features)
        current_distance = 1.0 - cosine(
            episodes[self.medoid_episode_id].vector, self.centroid
        )
        if (new_distance, episode.episode_id) < (current_distance, self.medoid_episode_id):
            self.medoid_episode_id = episode.episode_id

    def add_exception(self, episode: EnergyEpisode) -> None:
        if episode.episode_id not in self.all_provenance():
            self.exception_episode_ids.append(episode.episode_id)

    def _add_features(self, features: EpisodeFeatures) -> None:
        self.entity_counts.update(features.entities)
        self.role_counts.update(features.roles)
        self.context_counts.update(features.contexts)
        for slot, value in features.attributes:
            self.attribute_counts.setdefault(slot, Counter()).update((value,))

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "topic": self.topic,
            "medoid_episode_id": self.medoid_episode_id,
            "support_episode_ids": self.support_episode_ids,
            "exception_episode_ids": self.exception_episode_ids,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "status": self.status,
            "predecessor_schema_id": self.predecessor_schema_id,
            "successor_schema_ids": self.successor_schema_ids,
            "entity_distribution": normalized_counts(self.entity_counts),
            "role_distribution": normalized_counts(self.role_counts),
            "context_distribution": normalized_counts(self.context_counts),
            "attribute_distributions": {
                slot: normalized_counts(counts)
                for slot, counts in sorted(self.attribute_counts.items())
            },
            "centroid_dimension": int(self.centroid.size),
            "mean_distortion": self.mean_distortion,
        }


@dataclass(frozen=True)
class TransitionSignals:
    dense_similarity: float
    entity_overlap: float
    role_overlap: float
    context_match: float
    topic_match: float
    compatibility: float
    contradiction_probability: float
    temporal_potential: float
    contextual_potential: float
    recurrence: float
    explicit_change: float


@dataclass(frozen=True)
class EnergyTerms:
    distortion: float
    contradiction: float
    historical_damage: float
    fragmentation: float
    inconsistency: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return {
            "distortion": self.distortion,
            "contradiction": self.contradiction,
            "historical_damage": self.historical_damage,
            "fragmentation": self.fragmentation,
            "inconsistency": self.inconsistency,
            "total": self.total,
        }


@dataclass(frozen=True)
class TransitionRecord:
    episode_id: str
    operation: ActionName
    target_schema_id: str | None
    resulting_schema_ids: tuple[str, ...]
    candidate_energies: dict[ActionName, float]
    energy_terms: dict[ActionName, dict[str, float]]
    signals: dict[str, float]
    hysteresis_applied: bool
    stored_as_exception: bool = False


class TransitionLog:
    def __init__(self) -> None:
        self._records: list[TransitionRecord] = []

    def append(self, record: TransitionRecord) -> None:
        self._records.append(record)

    def all(self) -> tuple[TransitionRecord, ...]:
        return tuple(self._records)


def normalized_counts(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values())
    if not total:
        return {}
    return {key: value / total for key, value in sorted(counts.items())}


def unit_vector(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    return value / norm if norm else value


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    if not left.size or not right.size:
        return 0.0
    return float(np.clip(np.dot(unit_vector(left), unit_vector(right)), -1.0, 1.0))


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "do", "does", "for", "from", "had", "has", "have", "he", "her",
    "hers", "him", "his", "i", "in", "is", "it", "its", "me", "my",
    "of", "on", "or", "our", "she", "so", "that", "the", "their",
    "them", "they", "this", "to", "was", "we", "were", "with", "you",
    "your",
}
MULTI_VALUED_SLOTS = {
    "ability", "dislike", "experience", "plan", "possession", "preference"
}
CHANGE_CUES = re.compile(
    r"(?i)\b(moved|changed|switched|stopped|started|now|currently|"
    r"no longer|used to|formerly|previously|became)\b"
)
CONTEXT_PATTERNS = (
    r"(?i)\b(at work|at home|at school|at university|in the office)\b",
    r"(?i)\b(on weekends?|on weekdays?|during [a-z][a-z -]{1,30})\b",
    r"(?i)\b(when (?:i|we|he|she|they) [^,.!?;]{2,45})",
)
ATTRIBUTE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("location", r"(?i)\bI\s+(?:currently\s+)?(?:live|am living|moved|move)\s+(?:in|to)\s+([^,.!?;]+)"),
    ("employer", r"(?i)\bI\s+(?:currently\s+)?(?:work|am working|worked)\s+(?:at|for)\s+([^,.!?;]+)"),
    ("preference", r"(?i)\bI\s+(?:really\s+)?(?:prefer|like|love)\s+([^,.!?;]+)"),
    ("dislike", r"(?i)\bI\s+(?:really\s+)?(?:dislike|hate)\s+([^,.!?;]+)"),
    ("plan", r"(?i)\bI\s+(?:plan|am planning)\s+(?:to|on)?\s*([^,.!?;]+)"),
    ("possession", r"(?i)\bI\s+(?:have|own|bought|got|adopted)\s+([^,.!?;]+)"),
    ("experience", r"(?i)\bI\s+(?:went|visited|attended|joined|completed|won|created|made|tried)\s+([^,.!?;]+)"),
    ("ability", r"(?i)\bI\s+(?:can|know how to|am good at)\s+([^,.!?;]+)"),
)


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(token for token in TOKEN_RE.findall(text.casefold()) if token not in STOPWORDS)


def extract_episode_features(speaker: str, text: str) -> EpisodeFeatures:
    normalized = " ".join(text.split())
    entities: list[str] = [speaker.casefold()]
    entities.extend(
        match.casefold()
        for match in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", normalized)
        if match.casefold() not in {"i", speaker.casefold()}
    )
    roles = [
        "_".join(TOKEN_RE.findall(value.casefold()))
        for value in re.findall(
            r"(?i)\bmy\s+([a-z][a-z -]{0,30}?)\s+(?:is|are|was|were|has|have)\b",
            normalized,
        )
    ]
    contexts: list[str] = []
    for pattern in CONTEXT_PATTERNS:
        contexts.extend(match.casefold() for match in re.findall(pattern, normalized))

    attributes: list[tuple[str, str]] = []
    favorite = re.search(r"(?i)\bmy\s+favorite\s+([a-z -]+?)\s+is\s+([^,.!?;]+)", normalized)
    if favorite:
        slot = "favorite_" + "_".join(TOKEN_RE.findall(favorite.group(1).casefold()))
        attributes.append((slot, clean_value(favorite.group(2), contexts)))
    relation = re.search(
        r"(?i)\bmy\s+([a-z][a-z -]{0,30}?)\s+(?:is named|is called|is|are|was|were)\s+([^,.!?;]+)",
        normalized,
    )
    if relation:
        slot = "relationship_" + "_".join(TOKEN_RE.findall(relation.group(1).casefold()))
        attributes.append((slot, clean_value(relation.group(2), contexts)))
    for slot, pattern in ATTRIBUTE_PATTERNS:
        for match in re.finditer(pattern, normalized):
            value = clean_value(match.group(1), contexts)
            if value:
                attributes.append((slot, value))
    if not attributes:
        state = re.search(r"(?i)^(?:I\s+am|I'm|I\s+was)\s+([^,.!?;]+)", normalized)
        if state:
            attributes.append(("attribute", clean_value(state.group(1), contexts)))

    return EpisodeFeatures(
        entities=tuple(dict.fromkeys(entities)),
        roles=tuple(dict.fromkeys(item for item in roles if item)),
        contexts=tuple(dict.fromkeys(contexts)),
        attributes=tuple(dict.fromkeys(item for item in attributes if item[1])),
        tokens=tokenize(normalized),
        explicit_change=bool(CHANGE_CUES.search(normalized)),
    )


def clean_value(value: str, contexts: Sequence[str]) -> str:
    cleaned = value.strip().casefold()
    for context in contexts:
        if cleaned.endswith(context):
            cleaned = cleaned[: -len(context)].strip()
    return " ".join(TOKEN_RE.findall(cleaned))


def timestamp_value(value: str) -> float:
    try:
        return date_parser.parse(value, fuzzy=True).timestamp()
    except (OSError, OverflowError, TypeError, ValueError):
        return 0.0


def distribution_overlap(left: Counter[str], right: Sequence[str]) -> float:
    if not left and not right:
        return 0.5
    if not left or not right:
        return 0.5
    return sum(left.get(item, 0) for item in set(right)) / max(1, sum(left.values()))


def value_similarity(left: str, right: str) -> float:
    a, b = set(tokenize(left)), set(tokenize(right))
    if not a or not b:
        return float(left == right)
    return len(a & b) / len(a | b)


def transition_signals(
    schema: EnergySchema,
    episode: EnergyEpisode,
    episodes: Mapping[str, EnergyEpisode],
) -> TransitionSignals:
    dense = max(0.0, cosine(schema.centroid, episode.vector))
    entity = distribution_overlap(schema.entity_counts, episode.features.entities)
    role = distribution_overlap(schema.role_counts, episode.features.roles)
    context = distribution_overlap(schema.context_counts, episode.features.contexts)
    topic_match = float(
        episode.features.primary_slot != "discourse"
        and schema.topic.endswith("::" + episode.features.primary_slot)
    )
    contradiction = contradiction_probability(schema, episode.features)
    recurrence = exception_recurrence(schema, episode.features, episodes)
    latest = max(
        (timestamp_value(episodes[item].timestamp) for item in schema.support_episode_ids),
        default=0.0,
    )
    later = float(timestamp_value(episode.timestamp) >= latest)
    temporal = contradiction * min(
        1.0,
        0.35 * context
        + 0.20 * later
        + 0.40 * float(episode.features.explicit_change)
        + 0.25 * recurrence,
    )
    contextual_strength = max(contradiction, 0.65 * topic_match)
    contextual = contextual_strength * min(
        1.0,
        0.45 * (1.0 - context)
        + 0.20 * (1.0 - role)
        + 0.35 * (1.0 - dense),
    )
    compatibility = float(
        np.clip(
            0.52 * dense
            + 0.18 * entity
            + 0.12 * role
            + 0.10 * context
            + 0.08 * topic_match,
            0.0,
            1.0,
        )
    )
    return TransitionSignals(
        dense_similarity=dense,
        entity_overlap=entity,
        role_overlap=role,
        context_match=context,
        topic_match=topic_match,
        compatibility=compatibility,
        contradiction_probability=contradiction,
        temporal_potential=temporal,
        contextual_potential=contextual,
        recurrence=recurrence,
        explicit_change=float(episode.features.explicit_change),
    )


def contradiction_probability(schema: EnergySchema, features: EpisodeFeatures) -> float:
    incoming = features.attribute_map()
    contradictions: list[float] = []
    for slot, values in incoming.items():
        if slot in MULTI_VALUED_SLOTS or slot not in schema.attribute_counts:
            continue
        existing = tuple(schema.attribute_counts[slot])
        best = max(
            (value_similarity(old, new) for old in existing for new in values),
            default=0.0,
        )
        contradictions.append(1.0 - best)
    return max(contradictions, default=0.0)


def exception_recurrence(
    schema: EnergySchema,
    features: EpisodeFeatures,
    episodes: Mapping[str, EnergyEpisode],
) -> float:
    incoming = features.attribute_map()
    if not incoming:
        return 0.0
    matches = 0
    for identifier in schema.exception_episode_ids:
        previous = episodes[identifier].features.attribute_map()
        if any(
            slot in previous
            and max(
                value_similarity(left, right)
                for left in values
                for right in previous[slot]
            ) >= 0.8
            for slot, values in incoming.items()
        ):
            matches += 1
    return min(1.0, matches / 2.0)


def compute_action_energies(
    schema: EnergySchema,
    episode: EnergyEpisode,
    episodes: Mapping[str, EnergyEpisode],
    signals: TransitionSignals,
    *,
    active_schema_count: int,
    episode_count: int,
    tiny_schema_count: int,
    weights: Mapping[str, float],
    fragmentation_eta: float,
    minimum_schema_size: int,
    use_temporal: bool = True,
    use_contextual: bool = True,
) -> dict[ActionName, EnergyTerms]:
    temporal = signals.temporal_potential if use_temporal else 0.0
    contextual = signals.contextual_potential if use_contextual else 0.0
    projected_distortion = distortion_with_episode(schema, episode, episodes)
    drift = 1.0 - cosine(schema.centroid, unit_vector(
        schema.centroid * len(schema.support_episode_ids) + episode.vector
    ))
    projected_inconsistency = schema_inconsistency(schema) + 0.5 * max(temporal, contextual)
    normalizer = max(1.0, math.sqrt(episode_count + 1.0))

    def fragmentation(action: ActionName) -> float:
        creates_active = int(action in {"split", "new"})
        projected_active = active_schema_count + creates_active
        projected_tiny = tiny_schema_count + creates_active
        if action == "reinforce" and len(schema.support_episode_ids) + 1 >= minimum_schema_size:
            projected_tiny = max(0, projected_tiny - int(len(schema.support_episode_ids) < minimum_schema_size))
        return projected_active / normalizer + fragmentation_eta * projected_tiny / normalizer

    raw: dict[ActionName, tuple[float, float, float, float, float]] = {
        "reinforce": (
            projected_distortion,
            signals.contradiction_probability,
            drift + 0.35 * signals.contradiction_probability,
            fragmentation("reinforce"),
            projected_inconsistency,
        ),
        "supersede": (
            0.0,
            0.0,
            0.0,
            fragmentation("supersede"),
            max(0.0, contextual - temporal),
        ),
        "split": (
            0.0,
            0.0,
            0.0,
            fragmentation("split"),
            max(0.0, temporal - contextual),
        ),
        "new": (
            0.0,
            0.0,
            0.0,
            fragmentation("new"),
            signals.compatibility,
        ),
    }
    results: dict[ActionName, EnergyTerms] = {}
    for action, values in raw.items():
        distortion, contradiction, history, fragment, inconsistency = values
        total = (
            weights["distortion"] * distortion
            + weights["contradiction"] * contradiction
            + weights["historical_damage"] * history
            + weights["fragmentation"] * fragment
            + weights["inconsistency"] * inconsistency
        )
        results[action] = EnergyTerms(
            distortion=distortion,
            contradiction=contradiction,
            historical_damage=history,
            fragmentation=fragment,
            inconsistency=inconsistency,
            total=float(total),
        )
    return results


def distortion_with_episode(
    schema: EnergySchema,
    episode: EnergyEpisode,
    episodes: Mapping[str, EnergyEpisode],
) -> float:
    count = len(schema.support_episode_ids)
    centroid = unit_vector(schema.centroid * count + episode.vector)
    drift = 1.0 - cosine(schema.centroid, centroid)
    new_distance = 1.0 - cosine(episode.vector, centroid)
    return float((count * (schema.mean_distortion + drift) + new_distance) / (count + 1))


def schema_inconsistency(schema: EnergySchema) -> float:
    components: list[float] = []
    for counts in schema.attribute_counts.values():
        total = sum(counts.values())
        if total:
            components.append(1.0 - max(counts.values()) / total)
    for counts in (schema.role_counts, schema.context_counts):
        total = sum(counts.values())
        if total:
            components.append(1.0 - max(counts.values()) / total)
    return float(np.mean(components)) if components else 0.0
