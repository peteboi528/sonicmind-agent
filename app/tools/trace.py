from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

_SENSITIVE_KEYS = {"api_key", "authorization", "cookie", "lyrics", "token", "password", "secret"}


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[redacted]" if str(key).lower() in _SENSITIVE_KEYS else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value[:50]]
    if isinstance(value, str) and len(value) > 500:
        return value[:500] + "…"
    return value


class LocalTraceStore:
    def __init__(self, path: str | Path, *, retention_days: int = 30, enabled: bool = True) -> None:
        self.path = Path(path)
        self.retention_days = retention_days
        self.enabled = enabled
        self._lock = threading.RLock()
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, user_hash TEXT NOT NULL,
                    query_excerpt TEXT NOT NULL, status TEXT NOT NULL, started_at TEXT NOT NULL,
                    finished_at TEXT, duration_ms REAL, attrs_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS spans (
                    span_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, parent_span_id TEXT,
                    name TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL,
                    started_at TEXT NOT NULL, duration_ms REAL, retries INTEGER NOT NULL DEFAULT 0,
                    attrs_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_spans_run ON spans(run_id);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, span_id TEXT,
                    name TEXT NOT NULL, created_at TEXT NOT NULL, attrs_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
                """
            )

    def start_run(self, run_id: str, thread_id: str, user_id: str, query: str, **attrs: Any) -> None:
        if not self.enabled:
            return
        user_hash = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO runs(run_id,thread_id,user_hash,query_excerpt,status,started_at,attrs_json) VALUES(?,?,?,?,?,?,?)",
                (
                    run_id,
                    thread_id,
                    user_hash,
                    query[:200],
                    "running",
                    datetime.now(UTC).isoformat(),
                    json.dumps(_redact(attrs), ensure_ascii=False),
                ),
            )

    def finish_run(self, run_id: str, status: str, duration_ms: float, **attrs: Any) -> None:
        if not self.enabled:
            return
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE runs SET status=?,finished_at=?,duration_ms=?,attrs_json=? WHERE run_id=?",
                (
                    status,
                    datetime.now(UTC).isoformat(),
                    duration_ms,
                    json.dumps(_redact(attrs), ensure_ascii=False),
                    run_id,
                ),
            )

    def span(
        self,
        *,
        span_id: str,
        run_id: str,
        name: str,
        kind: str,
        status: str,
        started_at: str,
        duration_ms: float,
        retries: int = 0,
        parent_span_id: str | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO spans(span_id,run_id,parent_span_id,name,kind,status,started_at,duration_ms,retries,attrs_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    span_id,
                    run_id,
                    parent_span_id,
                    name,
                    kind,
                    status,
                    started_at,
                    duration_ms,
                    retries,
                    json.dumps(_redact(attrs or {}), ensure_ascii=False),
                ),
            )

    def event(self, run_id: str, name: str, *, span_id: str | None = None, **attrs: Any) -> None:
        if not self.enabled:
            return
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO events(run_id,span_id,name,created_at,attrs_json) VALUES(?,?,?,?,?)",
                (run_id, span_id, name, datetime.now(UTC).isoformat(), json.dumps(_redact(attrs), ensure_ascii=False)),
            )

    def cleanup(self) -> None:
        if not self.enabled:
            return
        cutoff = (datetime.now(UTC) - timedelta(days=self.retention_days)).isoformat()
        with self._lock, self._connect() as connection:
            old_runs = [row[0] for row in connection.execute("SELECT run_id FROM runs WHERE started_at < ?", (cutoff,))]
            if old_runs:
                marks = ",".join("?" for _ in old_runs)
                connection.execute(f"DELETE FROM spans WHERE run_id IN ({marks})", old_runs)
                connection.execute(f"DELETE FROM events WHERE run_id IN ({marks})", old_runs)
                connection.execute(f"DELETE FROM runs WHERE run_id IN ({marks})", old_runs)
