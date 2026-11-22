from __future__ import annotations

import asyncio
import time

from app.tools.checkpoints import ActionCheckpointStore
from app.tools.contracts import ToolCall, ToolContext, ToolResult, ToolStatus
from app.tools.registry import TOOL_REGISTRY
from app.tools.runtime import ToolRuntime


def _context() -> ToolContext:
    return ToolContext(thread_id="thread-1", user_id="user-1", query="test")


def test_runtime_validates_arguments_before_handler():
    spec = TOOL_REGISTRY["recommend"]
    original = spec.handler
    called = False

    def handler(_arguments, _context):
        nonlocal called
        called = True
        return ToolResult(tool="recommend", status=ToolStatus.OK)

    spec.handler = handler
    try:
        result = asyncio.run(ToolRuntime().execute(ToolCall(name="recommend", arguments={}), _context()))
    finally:
        spec.handler = original
    assert result.status == ToolStatus.ERROR
    assert result.error and result.error.kind == "validation_error"
    assert called is False


def test_external_write_requires_confirmation_and_executes_once_confirmed():
    spec = TOOL_REGISTRY["favorite_track"]
    original = spec.handler
    calls = 0

    def handler(arguments, _context):
        nonlocal calls
        calls += 1
        return ToolResult(tool="favorite_track", status=ToolStatus.OK, data=arguments)

    spec.handler = handler
    call = ToolCall(call_id="action-1", name="favorite_track", arguments={"track_id": "42"})
    try:
        pending = asyncio.run(ToolRuntime().execute(call, _context()))
        confirmed_context = _context().model_copy(update={"confirmation": {"action_id": "action-1", "approved": True}})
        completed = asyncio.run(ToolRuntime().execute(call, confirmed_context))
    finally:
        spec.handler = original
    assert pending.status == ToolStatus.CONFIRMATION_REQUIRED
    assert calls == 1
    assert completed.status == ToolStatus.OK


def test_checkpoint_resolution_is_thread_scoped_and_idempotent(tmp_path):
    store = ActionCheckpointStore(tmp_path / "checkpoints.sqlite")
    store.put("action-1", "thread-1", "user-1", "favorite_track", {"track_id": "42"}, "收藏")
    assert store.resolve("action-1", "wrong-thread", "user-1", True) is None
    resolved = store.resolve("action-1", "thread-1", "user-1", True)
    assert resolved and resolved["approved"] is True
    assert store.resolve("action-1", "thread-1", "user-1", True) is None


def test_agent_plan_materializes_dependency_stages():
    from app.graph.nodes import _materialize_tool_stages, build_agent_plan

    plan = _materialize_tool_stages(build_agent_plan("推荐几首适合专注的歌"), "推荐几首适合专注的歌", 5)
    names = [[call.name for call in stage.calls] for stage in plan.stages]
    assert names == [["web_music_search"], ["recommend"]]
    assert plan.stages[1].parallel is False


def test_async_runtime_times_out_sync_handler_without_retry():
    spec = TOOL_REGISTRY["recommend"]
    original_handler = spec.handler
    original_timeout = spec.timeout_seconds
    original_retries = spec.max_retries
    calls = 0

    def slow_handler(_arguments, _context):
        nonlocal calls
        calls += 1
        time.sleep(0.15)
        return ToolResult(tool="recommend", status=ToolStatus.OK)

    spec.handler = slow_handler
    spec.timeout_seconds = 0.03
    spec.max_retries = 2

    async def execute():
        started = time.monotonic()
        result = await ToolRuntime().execute(
            ToolCall(name="recommend", arguments={"query": "test", "top_k": 5}),
            _context(),
        )
        return result, time.monotonic() - started

    try:
        result, elapsed = asyncio.run(execute())
    finally:
        spec.handler = original_handler
        spec.timeout_seconds = original_timeout
        spec.max_retries = original_retries
    assert elapsed < 0.1
    assert calls == 1
    assert result.status == ToolStatus.ERROR
    assert result.error and result.error.kind == "timeout"
    assert "timed out" in result.error.message


def test_runtime_deadline_caps_tool_timeout():
    """请求级 deadline 比 tool spec 更短时，必须优先按剩余时间中止。"""
    spec = TOOL_REGISTRY["recommend"]
    original_handler = spec.handler
    original_timeout = spec.timeout_seconds

    def slow_handler(_arguments, _context):
        time.sleep(0.2)
        return ToolResult(tool="recommend", status=ToolStatus.OK)

    spec.handler = slow_handler
    spec.timeout_seconds = 1.0

    async def execute():
        started = time.monotonic()
        result = await ToolRuntime().execute(
            ToolCall(name="recommend", arguments={"query": "test", "top_k": 5}),
            _context().model_copy(update={"deadline_at": started + 0.03}),
        )
        return result, time.monotonic() - started

    try:
        result, elapsed = asyncio.run(execute())
    finally:
        spec.handler = original_handler
        spec.timeout_seconds = original_timeout
    assert elapsed < 0.12
    assert result.status == ToolStatus.ERROR
    assert result.error and result.error.kind == "timeout"


def test_timed_out_sync_worker_holds_capacity_until_it_exits(monkeypatch):
    from threading import Event

    spec = TOOL_REGISTRY["recommend"]
    started = Event()
    finished = Event()
    calls = 0

    def handler(_arguments, _context):
        nonlocal calls
        calls += 1
        started.set()
        finished.wait(1.0)
        return ToolResult(tool="recommend", status=ToolStatus.OK)

    monkeypatch.setattr(spec, "async_handler", None)
    monkeypatch.setattr(spec, "handler", handler)
    monkeypatch.setattr(spec, "max_concurrency", 1)
    monkeypatch.setattr(spec, "timeout_seconds", 0.08)
    call = ToolCall(name="recommend", arguments={"query": "test", "top_k": 5})

    async def execute():
        runtime = ToolRuntime()
        try:
            first = await runtime.execute(call, _context())
            assert started.is_set()
            assert first.metrics["timeout_phase"] == "execution"
            assert first.metrics["worker_still_running"]
            queued = await runtime.execute(call, _context())
            assert queued.metrics["timeout_phase"] == "queue"
            assert not queued.metrics["worker_still_running"]
            assert calls == 1
        finally:
            finished.set()
        recovered = await runtime.execute(call, _context())
        assert recovered.status == ToolStatus.OK
        assert calls == 2

    asyncio.run(execute())


def test_queue_and_retries_share_the_same_deadline(monkeypatch):
    spec = TOOL_REGISTRY["recommend"]
    deadlines = []

    async def handler(_arguments, context):
        deadlines.append(context.deadline_at)
        if len(deadlines) == 1:
            raise ConnectionResetError("first attempt")
        await asyncio.sleep(1.0)

    monkeypatch.setattr(spec, "async_handler", handler)
    monkeypatch.setattr(spec, "timeout_seconds", 0.28)
    monkeypatch.setattr(spec, "max_retries", 1)
    call = ToolCall(name="recommend", arguments={"query": "test", "top_k": 5})
    result = asyncio.run(ToolRuntime().execute(call, _context()))
    assert len(deadlines) == 2
    assert deadlines[0] == deadlines[1]
    assert result.metrics["timeout_phase"] == "execution"
    assert result.metrics["duration_ms"] < 400
