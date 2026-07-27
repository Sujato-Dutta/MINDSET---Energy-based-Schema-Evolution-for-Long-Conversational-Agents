from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mindset.api.models import CompletionResult


@dataclass(frozen=True)
class CacheClaim:
    state: Literal["hit", "claimed", "busy"]
    owner: str
    result: CompletionResult | None = None


class RequestCache:
    """SQLite response cache with leases to deduplicate concurrent dispatches."""

    def __init__(self, path: str | Path, lease_seconds: float = 300.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lease_seconds = lease_seconds
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS request_cache (
                    cache_key TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK(status IN ('pending', 'complete')),
                    owner TEXT NOT NULL,
                    lease_until REAL,
                    response_json TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def claim(self, cache_key: str, owner: str | None = None) -> CacheClaim:
        owner = owner or uuid.uuid4().hex
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM request_cache WHERE cache_key = ?", (cache_key,)
            ).fetchone()
            if row is None:
                connection.execute(
                    """INSERT INTO request_cache
                       (cache_key, status, owner, lease_until, created_at, updated_at)
                       VALUES (?, 'pending', ?, ?, ?, ?)""",
                    (cache_key, owner, now + self.lease_seconds, now, now),
                )
                connection.commit()
                return CacheClaim("claimed", owner)
            if row["status"] == "complete":
                result = CompletionResult.model_validate(json.loads(row["response_json"]))
                connection.commit()
                return CacheClaim("hit", owner, result.model_copy(update={"cache_hit": True}))
            if float(row["lease_until"] or 0) <= now:
                connection.execute(
                    """UPDATE request_cache SET owner = ?, lease_until = ?, updated_at = ?
                       WHERE cache_key = ?""",
                    (owner, now + self.lease_seconds, now, cache_key),
                )
                connection.commit()
                return CacheClaim("claimed", owner)
            connection.commit()
            return CacheClaim("busy", owner)

    def wait_for_result(
        self, cache_key: str, timeout_seconds: float = 300.0, poll_seconds: float = 0.05
    ) -> CompletionResult | None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            claim = self.claim(cache_key)
            if claim.state == "hit":
                return claim.result
            if claim.state == "claimed":
                # The prior lease expired. Caller should dispatch under this new claim,
                # but this helper cannot return its owner, so release it immediately.
                self.release(cache_key, claim.owner)
                return None
            time.sleep(poll_seconds)
        return None

    def store(self, cache_key: str, owner: str, result: CompletionResult) -> None:
        now = time.time()
        payload = result.model_copy(update={"cache_hit": False}).model_dump_json()
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE request_cache
                   SET status = 'complete', response_json = ?, lease_until = NULL, updated_at = ?
                   WHERE cache_key = ? AND owner = ? AND status = 'pending'""",
                (payload, now, cache_key, owner),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Cache lease was lost before result storage")

    def release(self, cache_key: str, owner: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM request_cache WHERE cache_key = ? AND owner = ? AND status = 'pending'",
                (cache_key, owner),
            )

    def get(self, cache_key: str) -> CompletionResult | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT response_json FROM request_cache WHERE cache_key = ? AND status = 'complete'",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return CompletionResult.model_validate(json.loads(row["response_json"])).model_copy(
            update={"cache_hit": True}
        )
