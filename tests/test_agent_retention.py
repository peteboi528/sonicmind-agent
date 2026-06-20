from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.tools.checkpoints import ActionCheckpointStore
from app.tools.trace import LocalTraceStore


def test_trace_cleanup_removes_expired_run_spans_and_events(tmp_path):
    store = LocalTraceStore(tmp_path / "trace.sqlite", retention_days=30)
    store.start_run("old", "thread-old", "user", "old query")
    store.span(
        span_id="span-old", run_id="old", name="tool", kind="tool", status="ok",
        started_at=datetime.now(UTC).isoformat(), duration_ms=1,
    )
    store.event("old", "fallback")
    store.start_run("fresh", "thread-fresh", "user", "fresh query")
    old_date = (datetime.now(UTC) - timedelta(days=31)).isoformat()
    with store._connect() as connection:
        connection.execute("UPDATE runs SET started_at=? WHERE run_id='old'", (old_date,))

    store.cleanup()

    with store._connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM runs WHERE run_id='old'").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM spans WHERE run_id='old'").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM events WHERE run_id='old'").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM runs WHERE run_id='fresh'").fetchone()[0] == 1


def test_checkpoint_cleanup_removes_expired_actions_and_thread_state(tmp_path):
    store = ActionCheckpointStore(tmp_path / "checkpoint.sqlite", retention_days=30)
    store.put("old-action", "old-thread", "user", "favorite_track", {"track_id": "1"}, "收藏")
    store.put("fresh-action", "fresh-thread", "user", "favorite_track", {"track_id": "2"}, "收藏")
    old_date = (datetime.now(UTC) - timedelta(days=31)).isoformat()
    with store._connect() as connection:
        connection.execute("UPDATE pending_actions SET created_at=? WHERE action_id='old-action'", (old_date,))
        connection.execute("UPDATE thread_activity SET updated_at=? WHERE thread_id='old-thread'", (old_date,))
        connection.execute("CREATE TABLE checkpoints(thread_id TEXT, payload TEXT)")
        connection.execute("CREATE TABLE writes(thread_id TEXT, payload TEXT)")
        connection.execute("INSERT INTO checkpoints VALUES('old-thread','x'),('fresh-thread','y')")
        connection.execute("INSERT INTO writes VALUES('old-thread','x'),('fresh-thread','y')")

    store.cleanup()

    with store._connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM pending_actions WHERE action_id='old-action'").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM pending_actions WHERE action_id='fresh-action'").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id='old-thread'").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM writes WHERE thread_id='old-thread'").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id='fresh-thread'").fetchone()[0] == 1
