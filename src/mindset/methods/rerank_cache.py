from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mindset.utils.hashing import stable_hash


class CachedReranker:
    """Persistent exact cache for deterministic local cross-encoder pair scores."""

    def __init__(self, reranker: Any, model_name: str, path: str | Path) -> None:
        self.reranker = reranker
        self.model_name = model_name
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rerank_scores (
                    cache_key TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    query TEXT NOT NULL,
                    document TEXT NOT NULL,
                    score REAL NOT NULL
                )
                """
            )

    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        keys = [
            stable_hash({"model": self.model_name, "query": query, "document": document})
            for query, document in pairs
        ]
        cached: dict[str, float] = {}
        if keys:
            with self._connect() as connection:
                for start in range(0, len(keys), 500):
                    selected = keys[start : start + 500]
                    placeholders = ",".join("?" for _ in selected)
                    rows = connection.execute(
                        f"SELECT cache_key, score FROM rerank_scores WHERE cache_key IN ({placeholders})",
                        selected,
                    ).fetchall()
                    cached.update({str(row[0]): float(row[1]) for row in rows})
        missing: dict[str, tuple[str, str]] = {}
        for key, pair in zip(keys, pairs, strict=True):
            if key not in cached:
                missing.setdefault(key, pair)
        if missing:
            missing_keys = list(missing)
            missing_pairs = [missing[key] for key in missing_keys]
            scores = self.reranker.predict(
                missing_pairs,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                **kwargs,
            )
            with self._connect() as connection:
                connection.executemany(
                    """
                    INSERT OR IGNORE INTO rerank_scores
                    (cache_key, model_name, query, document, score)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (key, self.model_name, pair[0], pair[1], float(score))
                        for key, pair, score in zip(
                            missing_keys, missing_pairs, scores, strict=True
                        )
                    ],
                )
            cached.update(
                {key: float(score) for key, score in zip(missing_keys, scores, strict=True)}
            )
        return np.asarray([cached[key] for key in keys], dtype=np.float32)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection
