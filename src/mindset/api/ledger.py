from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from mindset.api.models import TokenUsage


class TokenBudgetExceeded(RuntimeError):
    pass


class TokenLedger:
    """Atomic reservations prevent concurrent workers from overspending a model."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS model_usage (
                    model_name TEXT PRIMARY KEY,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    reserved_tokens INTEGER NOT NULL DEFAULT 0,
                    hard_budget INTEGER NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reservations (
                    reservation_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    reserved_tokens INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS key_usage (
                    provider TEXT NOT NULL,
                    key_alias TEXT NOT NULL,
                    requests INTEGER NOT NULL DEFAULT 0,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    failures INTEGER NOT NULL DEFAULT 0,
                    retries INTEGER NOT NULL DEFAULT 0,
                    cooldown_until REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY(provider, key_alias)
                );
                CREATE TABLE IF NOT EXISTS request_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    cache_key TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    key_alias TEXT NOT NULL,
                    request_id TEXT,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    latency_seconds REAL NOT NULL,
                    retry_count INTEGER NOT NULL,
                    cache_hit INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS daily_key_usage (
                    model_name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    key_alias TEXT NOT NULL,
                    period TEXT NOT NULL,
                    requests INTEGER NOT NULL DEFAULT 0,
                    tokens INTEGER NOT NULL DEFAULT 0,
                    reserved_requests INTEGER NOT NULL DEFAULT 0,
                    reserved_tokens INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(model_name, provider, key_alias, period)
                );
                CREATE TABLE IF NOT EXISTS daily_key_reservations (
                    reservation_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    key_alias TEXT NOT NULL,
                    period TEXT NOT NULL,
                    reserved_tokens INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );
                INSERT OR IGNORE INTO daily_key_usage
                    (model_name, provider, key_alias, period, requests, tokens)
                SELECT model_name, provider, key_alias, date(timestamp, 'unixepoch'),
                       COUNT(*), SUM(input_tokens + output_tokens)
                FROM request_log
                WHERE key_alias != 'cache'
                GROUP BY model_name, provider, key_alias, date(timestamp, 'unixepoch');
                """
            )

    def ensure_model(self, model_name: str, hard_budget: int) -> None:
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO model_usage(model_name, hard_budget, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(model_name) DO UPDATE SET
                     hard_budget = excluded.hard_budget,
                     updated_at = excluded.updated_at""",
                (model_name, hard_budget, now),
            )

    def reserve(self, model_name: str, estimated_tokens: int, hard_budget: int) -> str:
        if estimated_tokens <= 0:
            raise ValueError("estimated_tokens must be positive")
        self.ensure_model(model_name, hard_budget)
        reservation_id = uuid.uuid4().hex
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM model_usage WHERE model_name = ?", (model_name,)
            ).fetchone()
            committed = int(row["input_tokens"]) + int(row["output_tokens"])
            projected = committed + int(row["reserved_tokens"]) + estimated_tokens
            budget = int(row["hard_budget"])
            if projected > budget:
                connection.rollback()
                raise TokenBudgetExceeded(
                    f"{model_name} budget stop: projected {projected:,} > hard limit {budget:,}"
                )
            connection.execute(
                "UPDATE model_usage SET reserved_tokens = reserved_tokens + ?, updated_at = ? "
                "WHERE model_name = ?",
                (estimated_tokens, now, model_name),
            )
            connection.execute(
                "INSERT INTO reservations VALUES (?, ?, ?, ?)",
                (reservation_id, model_name, estimated_tokens, now),
            )
            connection.commit()
        return reservation_id

    def reserve_daily_key(
        self,
        *,
        model_name: str,
        provider: str,
        key_alias: str,
        period: str,
        estimated_tokens: int,
        token_budget: int | None,
        request_budget: int | None,
    ) -> str | None:
        """Atomically reserve one provider request within a per-key daily window."""
        if token_budget is None and request_budget is None:
            return None
        reservation_id = uuid.uuid4().hex
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT OR IGNORE INTO daily_key_usage
                   (model_name, provider, key_alias, period) VALUES (?, ?, ?, ?)""",
                (model_name, provider, key_alias, period),
            )
            row = connection.execute(
                """SELECT * FROM daily_key_usage WHERE model_name = ? AND provider = ?
                   AND key_alias = ? AND period = ?""",
                (model_name, provider, key_alias, period),
            ).fetchone()
            token_projection = int(row["tokens"]) + int(row["reserved_tokens"]) + estimated_tokens
            request_projection = int(row["requests"]) + int(row["reserved_requests"]) + 1
            if (token_budget is not None and token_projection > token_budget) or (
                request_budget is not None and request_projection > request_budget
            ):
                connection.rollback()
                return ""
            connection.execute(
                """UPDATE daily_key_usage SET reserved_requests = reserved_requests + 1,
                   reserved_tokens = reserved_tokens + ? WHERE model_name = ? AND provider = ?
                   AND key_alias = ? AND period = ?""",
                (estimated_tokens, model_name, provider, key_alias, period),
            )
            connection.execute(
                "INSERT INTO daily_key_reservations VALUES (?, ?, ?, ?, ?, ?, ?)",
                (reservation_id, model_name, provider, key_alias, period, estimated_tokens, now),
            )
            connection.commit()
        return reservation_id

    def settle_daily_key(self, reservation_id: str | None, usage: TokenUsage) -> None:
        if reservation_id is None:
            return
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM daily_key_reservations WHERE reservation_id = ?", (reservation_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(f"Unknown daily key reservation: {reservation_id}")
            connection.execute(
                """UPDATE daily_key_usage SET requests = requests + 1, tokens = tokens + ?,
                   reserved_requests = reserved_requests - 1,
                   reserved_tokens = reserved_tokens - ? WHERE model_name = ? AND provider = ?
                   AND key_alias = ? AND period = ?""",
                (
                    usage.total_tokens,
                    row["reserved_tokens"],
                    row["model_name"],
                    row["provider"],
                    row["key_alias"],
                    row["period"],
                ),
            )
            connection.execute(
                "DELETE FROM daily_key_reservations WHERE reservation_id = ?", (reservation_id,)
            )
            connection.commit()

    def settle(self, reservation_id: str, usage: TokenUsage) -> None:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?", (reservation_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(f"Unknown reservation: {reservation_id}")
            connection.execute(
                """UPDATE model_usage SET
                     input_tokens = input_tokens + ?,
                     output_tokens = output_tokens + ?,
                     reserved_tokens = reserved_tokens - ?,
                     updated_at = ?
                   WHERE model_name = ?""",
                (
                    usage.input_tokens,
                    usage.output_tokens,
                    row["reserved_tokens"],
                    now,
                    row["model_name"],
                ),
            )
            connection.execute(
                "DELETE FROM reservations WHERE reservation_id = ?", (reservation_id,)
            )
            connection.commit()

    def release(self, reservation_id: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?", (reservation_id,)
            ).fetchone()
            if row is not None:
                connection.execute(
                    "UPDATE model_usage SET reserved_tokens = reserved_tokens - ? "
                    "WHERE model_name = ?",
                    (row["reserved_tokens"], row["model_name"]),
                )
                connection.execute(
                    "DELETE FROM reservations WHERE reservation_id = ?", (reservation_id,)
                )
            connection.commit()

    def record_key_result(
        self,
        provider: str,
        key_alias: str,
        usage: TokenUsage | None = None,
        *,
        failure: bool = False,
        retry: bool = False,
        cooldown_until: float = 0.0,
    ) -> None:
        usage = usage or TokenUsage()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO key_usage
                   (provider, key_alias, requests, input_tokens, output_tokens,
                    failures, retries, cooldown_until)
                   VALUES (?, ?, 1, ?, ?, ?, ?, ?)
                   ON CONFLICT(provider, key_alias) DO UPDATE SET
                     requests = key_usage.requests + 1,
                     input_tokens = key_usage.input_tokens + excluded.input_tokens,
                     output_tokens = key_usage.output_tokens + excluded.output_tokens,
                     failures = key_usage.failures + excluded.failures,
                     retries = key_usage.retries + excluded.retries,
                     cooldown_until = MAX(key_usage.cooldown_until, excluded.cooldown_until)""",
                (
                    provider,
                    key_alias,
                    usage.input_tokens,
                    usage.output_tokens,
                    int(failure),
                    int(retry),
                    cooldown_until,
                ),
            )

    def key_cooldown(self, provider: str, key_alias: str) -> float:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT cooldown_until FROM key_usage WHERE provider = ? AND key_alias = ?",
                (provider, key_alias),
            ).fetchone()
        return float(row[0]) if row else 0.0

    def log_request(self, **fields: Any) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO request_log
                   (timestamp, cache_key, provider, model_name, model_id, key_alias,
                    request_id, input_tokens, output_tokens, latency_seconds,
                    retry_count, cache_hit, status, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.time(),
                    fields["cache_key"],
                    fields["provider"],
                    fields["model_name"],
                    fields["model_id"],
                    fields.get("key_alias", "cache"),
                    fields.get("request_id"),
                    fields.get("input_tokens", 0),
                    fields.get("output_tokens", 0),
                    fields.get("latency_seconds", 0.0),
                    fields.get("retry_count", 0),
                    int(fields.get("cache_hit", False)),
                    fields.get("status", "complete"),
                    json.dumps(fields.get("metadata", {}), sort_keys=True),
                ),
            )

    def report(self) -> dict[str, Any]:
        with self._connect() as connection:
            models = [dict(row) for row in connection.execute("SELECT * FROM model_usage")]
            keys = [dict(row) for row in connection.execute("SELECT * FROM key_usage")]
            daily = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM daily_key_usage ORDER BY period, model_name, key_alias"
                )
            ]
        return {"models": models, "keys": keys, "daily_keys": daily}
