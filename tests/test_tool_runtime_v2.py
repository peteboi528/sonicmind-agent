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
