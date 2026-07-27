from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mindset.baselines.event_rag import render_event
from mindset.methods.base import ConversationEvent, MemoryHit
from mindset.retrieval.hybrid import ranked_indices
from mindset.utils.hashing import stable_hash


@dataclass(frozen=True)
class LightMemMTMItem:
    memory_id: str
    summary: str
    source_event_id: str
    timestamp: str
    user_id: str


@dataclass(frozen=True)
class LightMemLTMNode:
    node_id: str
    medoid: str
    source_event_ids: tuple[str, ...]
    neighbors: tuple[str, ...] = ()


class LightMemACL2026Reproduction:
    """Paper-guided local reproduction of LightMem (ACL 2026).

    The paper's three online SLM roles are preserved as separate operations:
    extractive HQ planning, bounded MTM writing, and two-stage semantic
    selection. Its unpublished LoRA selector and offline consolidation model are
    replaced by a pinned local cross-encoder and deterministic medoid graph.
    """

    method_name = "lightmem"
    reproduction_label = "paper_guided_cpu_local_reproduction"
    paper_url = "https://aclanthology.org/2026.acl-long.588/"

    def __init__(
        self,
        encoder: Any,
        semantic_selector: Any,
        *,
        selector_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        mtm_capacity: int = 10_000,
        ltm_similarity_threshold: float = 0.72,
        max_hypothesis_queries: int = 2,
    ) -> None:
        if mtm_capacity <= 0:
            raise ValueError("mtm_capacity must be positive")
        if max_hypothesis_queries <= 0:
            raise ValueError("max_hypothesis_queries must be positive")
        self.encoder = encoder
        self.semantic_selector = semantic_selector
        self.selector_model = selector_model
        self.mtm_capacity = mtm_capacity
        self.ltm_similarity_threshold = ltm_similarity_threshold
        self.max_hypothesis_queries = max_hypothesis_queries
        self.events: list[ConversationEvent] = []
        self.mtm: list[LightMemMTMItem] = []
        self.ltm: list[LightMemLTMNode] = []
        self._event_index_by_id: dict[str, int] = {}
        self._mtm_vectors = np.empty((0, 0), dtype=np.float32)
        self._ltm_vectors = np.empty((0, 0), dtype=np.float32)
        self._retrieval_calls = 0

    def ingest(self, events: Sequence[ConversationEvent]) -> None:
        self.events = list(events)[-self.mtm_capacity :]
        self._event_index_by_id = {
            event.event_id: index for index, event in enumerate(self.events)
        }
        self.mtm = [
            LightMemMTMItem(
                memory_id="lightmem-mtm-" + stable_hash(event.event_id)[:16],
                summary=render_event(event),
                source_event_id=event.event_id,
                timestamp=event.timestamp,
                user_id=event.conversation_id,
            )
            for event in self.events
        ]
        self._mtm_vectors = self._encode([item.summary for item in self.mtm])
        self._consolidate_ltm()

    def retrieve(self, query: str, top_k: int = 10) -> list[MemoryHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self.mtm:
            return []
        self._retrieval_calls += 1
        hypothesis_queries = self._plan_hypothesis_queries(query)
        coarse_budget = min(len(self.mtm), 2 * top_k)
        per_query = max(1, int(np.ceil(coarse_budget / len(hypothesis_queries))))
        candidate_scores: dict[int, float] = {}
        for hypothesis in hypothesis_queries:
            query_vector = self._encode([hypothesis])[0]
            mtm_scores = self._mtm_vectors @ query_vector
            for rank, index in enumerate(ranked_indices(mtm_scores, per_query), start=1):
                candidate_scores[index] = max(
                    candidate_scores.get(index, float("-inf")),
                    float(mtm_scores[index]) + 1.0 / (60 + rank),
                )
            if self.ltm:
                ltm_scores = self._ltm_vectors @ query_vector
                for node_index in ranked_indices(
                    ltm_scores, min(per_query, len(self.ltm))
                ):
                    node = self.ltm[node_index]
                    for source_id in node.source_event_ids:
                        event_index = self._event_index_by_id.get(source_id)
                        if event_index is not None:
                            candidate_scores[event_index] = max(
                                candidate_scores.get(event_index, float("-inf")),
                                float(ltm_scores[node_index]),
                            )
        coarse = sorted(
            candidate_scores,
            key=lambda index: (-candidate_scores[index], self.events[index].event_id),
        )[:coarse_budget]
        selector_scores = self.semantic_selector.predict(
            [(query, self.mtm[index].summary) for index in coarse],
            batch_size=32,
            show_progress_bar=False,
        )
        selected = sorted(
            zip(coarse, selector_scores, strict=True),
            key=lambda item: (-float(item[1]), self.events[item[0]].event_id),
        )[:top_k]
        return [
            MemoryHit(
                memory_id=self.mtm[index].memory_id,
                text=self.mtm[index].summary,
                score=float(score),
                source_event_ids=(self.mtm[index].source_event_id,),
                memory_type="lightmem_mtm",
                metadata={
                    **self.events[index].metadata,
                    "selector_model": self.selector_model,
                    "coarse_budget": coarse_budget,
                    "hypothesis_query_count": len(hypothesis_queries),
                    "reproduction_label": self.reproduction_label,
                },
            )
            for index, score in selected
        ]

    def snapshot(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": self.method_name,
            "reproduction_label": self.reproduction_label,
            "paper_url": self.paper_url,
            "mtm": [asdict(item) for item in self.mtm],
            "ltm": [asdict(node) for node in self.ltm],
        }
        target.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

    def telemetry(self) -> dict[str, Any]:
        return {
            "reproduction_label": self.reproduction_label,
            "paper_url": self.paper_url,
            "mtm_items": len(self.mtm),
            "ltm_nodes": len(self.ltm),
            "mtm_capacity": self.mtm_capacity,
            "selector_model": self.selector_model,
            "selector_training": (
                "unreleased_paper_lora_replaced_by_pinned_pretrained_cross_encoder"
            ),
            "offline_consolidation": "deterministic_medoid_graph",
            "retrieval_calls": self._retrieval_calls,
            "write_api_calls": 0,
            "retrieval_api_calls": 0,
        }

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
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

    def _plan_hypothesis_queries(self, query: str) -> list[str]:
        queries = [query.strip()]
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", query)
        stopwords = {
            "a", "an", "and", "are", "did", "do", "does", "for", "from", "how",
            "in", "is", "it", "of", "on", "the", "to", "was", "were", "what",
            "when", "where", "which", "who", "why", "with",
        }
        hint = " ".join(token for token in tokens if token.casefold() not in stopwords)
        if hint and hint.casefold() != query.strip().casefold():
            queries.append(hint)
        return queries[: self.max_hypothesis_queries]

    def _consolidate_ltm(self) -> None:
        if not self.mtm:
            self.ltm = []
            self._ltm_vectors = np.empty((0, 0), dtype=np.float32)
            return
        clusters: list[list[int]] = []
        centroids: list[np.ndarray] = []
        for index, vector in enumerate(self._mtm_vectors):
            if not centroids:
                clusters.append([index])
                centroids.append(vector.copy())
                continue
            similarities = np.asarray(centroids) @ vector
            best = int(np.argmax(similarities))
            if float(similarities[best]) >= self.ltm_similarity_threshold:
                clusters[best].append(index)
                centroid = np.mean(self._mtm_vectors[clusters[best]], axis=0)
                norm = float(np.linalg.norm(centroid))
                centroids[best] = centroid / norm if norm else centroid
            else:
                clusters.append([index])
                centroids.append(vector.copy())
        preliminary: list[LightMemLTMNode] = []
        for cluster in clusters:
            vectors = self._mtm_vectors[cluster]
            centroid = np.mean(vectors, axis=0)
            medoid_local = int(np.argmax(vectors @ centroid))
            medoid_index = cluster[medoid_local]
            source_ids = tuple(self.mtm[index].source_event_id for index in cluster)
            preliminary.append(
                LightMemLTMNode(
                    node_id="lightmem-ltm-" + stable_hash(source_ids)[:16],
                    medoid=self.mtm[medoid_index].summary,
                    source_event_ids=source_ids,
                )
            )
        ltm_vectors = self._encode([node.medoid for node in preliminary])
        linked: list[LightMemLTMNode] = []
        for index, node in enumerate(preliminary):
            scores = ltm_vectors @ ltm_vectors[index]
            neighbors = [
                preliminary[item].node_id
                for item in ranked_indices(scores, min(4, len(preliminary)))
                if item != index
            ][:3]
            linked.append(
                LightMemLTMNode(
                    node.node_id, node.medoid, node.source_event_ids, tuple(neighbors)
                )
            )
        self.ltm = linked
        self._ltm_vectors = ltm_vectors
