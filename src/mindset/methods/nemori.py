from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from mindset.methods.base import ConversationEvent, MemoryHit
from mindset.methods.cache import write_json_once
from mindset.utils.hashing import stable_hash


TRANSITION_CUES = (
    "anyway",
    "by the way",
    "on another note",
    "moving on",
    "speaking of",
    "later that day",
    "the next day",
)


@dataclass(frozen=True)
class NemoriEpisode:
    episode_id: str
    narrative: str
    source_event_ids: tuple[str, ...]
    start_time: str
    end_time: str


@dataclass(frozen=True)
class NemoriSemanticMemory:
    memory_id: str
    text: str
    source_event_ids: tuple[str, ...]
    prediction_gap: float


class NemoriFaithfulReproduction:
    """CPU-local reproduction of Nemori's three alignment mechanisms.

    Boundary alignment groups adjacent events using semantic continuity and
    explicit discourse cues. Representation alignment preserves an episode
    narrative with temporal and speaker anchors. Predict-calibrate compares a
    new episode with retrieved prior knowledge and distils extractive semantic
    facts only when the prediction gap exceeds the frozen threshold.
    """

    method_name = "nemori"
    reproduction_label = "faithful_reproduction"
    upstream_commit = "d2a6dff6e5481214a0be6a2d10147feccfc16244"

    def __init__(
        self,
        encoder: object,
        *,
        boundary_threshold: float = 0.32,
        prediction_gap_threshold: float = 0.48,
        keyword_weight: float = 0.25,
    ) -> None:
        self.encoder = encoder
        self.boundary_threshold = boundary_threshold
        self.prediction_gap_threshold = prediction_gap_threshold
        self.keyword_weight = keyword_weight
        self.episodes: list[NemoriEpisode] = []
        self.semantic_memories: list[NemoriSemanticMemory] = []
        self._vectors = np.empty((0, 0), dtype=np.float32)
        self._semantic_vectors = np.empty((0, 0), dtype=np.float32)
        self._ingest_seconds = 0.0

    def ingest(self, events: Sequence[ConversationEvent]) -> None:
        if not events:
            return
        started = perf_counter()
        semantic_start = len(self.semantic_memories)
        event_vectors = self._encode([event.text for event in events])
        groups: list[list[int]] = [[0]]
        for index in range(1, len(events)):
            similarity = float(event_vectors[index - 1] @ event_vectors[index])
            cue = events[index].text.lower().lstrip()
            explicit_boundary = any(cue.startswith(item) for item in TRANSITION_CUES)
            session_boundary = events[index].session_id != events[index - 1].session_id
            if explicit_boundary or session_boundary or similarity < self.boundary_threshold:
                groups.append([index])
            else:
                groups[-1].append(index)

        new_episodes: list[NemoriEpisode] = []
        for group in groups:
            selected = [events[index] for index in group]
            narrative = "\n".join(
                f"[{event.timestamp}] {event.speaker}: {event.text}" for event in selected
            )
            identifiers = tuple(event.event_id for event in selected)
            new_episodes.append(NemoriEpisode(
                episode_id="nemori-episode-" + stable_hash(identifiers)[:16],
                narrative=narrative,
                source_event_ids=identifiers,
                start_time=selected[0].timestamp,
                end_time=selected[-1].timestamp,
            ))
        episode_vectors = self._encode([episode.narrative for episode in new_episodes])
        prior_vectors = self._vectors
        intra_similarity = episode_vectors @ episode_vectors.T
        for index, episode in enumerate(new_episodes):
            similarities: list[float] = []
            if prior_vectors.size:
                similarities.append(float(np.max(prior_vectors @ episode_vectors[index])))
            if index:
                similarities.append(float(np.max(intra_similarity[index, :index])))
            self._calibrate(episode, max(similarities) if similarities else None)
        self.episodes.extend(new_episodes)
        self._vectors = (
            episode_vectors if prior_vectors.size == 0 else np.vstack((prior_vectors, episode_vectors))
        )
        new_semantic = self.semantic_memories[semantic_start:]
        if new_semantic:
            vectors = self._encode([memory.text for memory in new_semantic])
            self._semantic_vectors = (
                vectors if self._semantic_vectors.size == 0 else np.vstack((self._semantic_vectors, vectors))
            )
        self._ingest_seconds += perf_counter() - started

    def _calibrate(self, episode: NemoriEpisode, best_similarity: float | None) -> None:
        if best_similarity is None:
            return
        gap = max(0.0, 1.0 - best_similarity)
        if gap < self.prediction_gap_threshold:
            return
        facts = _extractive_facts(episode.narrative)
        if not facts:
            return
        text = " ".join(facts)
        memory = NemoriSemanticMemory(
                memory_id="nemori-semantic-" + stable_hash((episode.episode_id, text))[:16],
                text=text,
                source_event_ids=episode.source_event_ids,
                prediction_gap=gap,
        )
        self.semantic_memories.append(memory)

    def retrieve(self, query: str, top_k: int = 10) -> list[MemoryHit]:
        if top_k <= 0:
            return []
        query_vector = self._encode([query])[0]
        query_tokens = _tokens(query)
        candidates: list[MemoryHit] = []
        for memory_type, memories, vectors in (
            ("episodic", self.episodes, self._vectors),
            ("semantic", self.semantic_memories, self._semantic_vectors),
        ):
            for index, memory in enumerate(memories):
                text = memory.narrative if isinstance(memory, NemoriEpisode) else memory.text
                semantic = float(vectors[index] @ query_vector)
                overlap = len(query_tokens & _tokens(text)) / max(1, len(query_tokens))
                score = (1.0 - self.keyword_weight) * semantic + self.keyword_weight * overlap
                candidates.append(
                    MemoryHit(
                        memory_id=memory.episode_id if isinstance(memory, NemoriEpisode) else memory.memory_id,
                        text=text,
                        score=score,
                        source_event_ids=memory.source_event_ids,
                        memory_type=memory_type,
                        metadata={
                            "reproduction": self.reproduction_label,
                            "upstream_commit": self.upstream_commit,
                        },
                    )
                )
        return sorted(candidates, key=lambda item: (-item.score, item.memory_id))[:top_k]

    def snapshot(self, path: str | Path) -> Path:
        return write_json_once(
            path,
            {
                "method": self.method_name,
                "reproduction": self.reproduction_label,
                "upstream_commit": self.upstream_commit,
                "deviations": [
                    "local NumPy vectors replace Postgres and Qdrant storage",
                    "extractive deterministic semantic distillation replaces the upstream LLM distiller",
                ],
                "config": {
                    "boundary_threshold": self.boundary_threshold,
                    "prediction_gap_threshold": self.prediction_gap_threshold,
                    "keyword_weight": self.keyword_weight,
                },
                "episodes": [asdict(item) for item in self.episodes],
                "semantic_memories": [asdict(item) for item in self.semantic_memories],
            },
        )

    def telemetry(self) -> dict[str, Any]:
        return {
            "episode_count": len(self.episodes),
            "semantic_memory_count": len(self.semantic_memories),
            "ingest_seconds": self._ingest_seconds,
            "reproduction": self.reproduction_label,
        }

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            dimension = self._vectors.shape[1] if self._vectors.ndim == 2 else 0
            return np.empty((0, dimension), dtype=np.float32)
        values = self.encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(values, dtype=np.float32)


def _extractive_facts(narrative: str) -> list[str]:
    facts: list[str] = []
    for line in narrative.splitlines():
        content = line.split(": ", 1)[-1].strip()
        if len(content.split()) >= 3 and re.search(
            r"\b(?:is|are|was|were|has|have|likes?|loves?|prefers?|moved?|works?|plans?)\b",
            content,
            re.IGNORECASE,
        ):
            facts.append(content)
    return facts[:5]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _append_vector(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    row = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    if matrix.size == 0:
        return row
    if matrix.shape[1] != row.shape[1]:
        raise ValueError("Embedding dimension changed during ingestion")
    return np.vstack((matrix, row))
