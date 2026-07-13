from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


class ActionCheckpointStore:
    """Small durable confirmation ledger; LangGraph state uses the same thread/action ids."""

    def __init__(self, path: str | Path, retention_days: int = 30) -> None:
        self.path = Path(path)
        self.retention_days = retention_days
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS pending_actions(
                action_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, user_hash TEXT NOT NULL,
                tool TEXT NOT NULL, arguments_json TEXT NOT NULL, query_excerpt TEXT NOT NULL,
                status TEXT NOT NULL, created_at TEXT NOT NULL, resolved_at TEXT)"""
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS thread_activity(thread_id TEXT PRIMARY KEY, updated_at TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def put(
        self, action_id: str, thread_id: str, user_hash: str, tool: str, arguments: dict[str, Any], query: str
    ) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO pending_actions VALUES(?,?,?,?,?,?,?, ?,NULL)",
                (
                    action_id,
                    thread_id,
                    user_hash,
                    tool,
                    json.dumps(arguments, ensure_ascii=False),
                    query[:200],
                    "pending",
                    datetime.now(UTC).isoformat(),
                ),
            )
            connection.execute(
                "INSERT OR REPLACE INTO thread_activity(thread_id,updated_at) VALUES(?,?)",
                (thread_id, datetime.now(UTC).isoformat()),
            )
            return cursor.rowcount > 0

    def touch_thread(self, thread_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO thread_activity(thread_id,updated_at) VALUES(?,?)",
                (thread_id, datetime.now(UTC).isoformat()),
            )

    def resolve(self, action_id: str, thread_id: str, user_hash: str, approved: bool) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT tool,arguments_json,status FROM pending_actions WHERE action_id=? AND thread_id=? AND user_hash=?",
                (action_id, thread_id, user_hash),
            ).fetchone()
            if row is None or row[2] != "pending":
                return None
            status = "approved" if approved else "rejected"
            connection.execute(
                "UPDATE pending_actions SET status=?,resolved_at=? WHERE action_id=? AND status='pending'",
                (status, datetime.now(UTC).isoformat(), action_id),
            )
            return {"tool": row[0], "arguments": json.loads(row[1]), "approved": approved}

    def cleanup(self) -> None:
        cutoff = (datetime.now(UTC) - timedelta(days=self.retention_days)).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM pending_actions WHERE created_at < ?", (cutoff,))
            stale = [
                row[0]
                for row in connection.execute("SELECT thread_id FROM thread_activity WHERE updated_at < ?", (cutoff,))
            ]
            for thread_id in stale:
                # These tables are created by langgraph-checkpoint-sqlite after first use.
                for table in ("writes", "checkpoints"):
                    exists = connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
                    ).fetchone()
                    if exists:
                        connection.execute(f"DELETE FROM {table} WHERE thread_id=?", (thread_id,))
                connection.execute("DELETE FROM thread_activity WHERE thread_id=?", (thread_id,))
