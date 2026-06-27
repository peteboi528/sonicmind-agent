from __future__ import annotations

import asyncio

from app.agent import AudioVisualAgent
from app.graph.nodes import _run_tool_async
from app.models import AgentPlan, StreamEvent
from app.storage import JsonStore
from app.tools.contracts import ToolCall, ToolResult, ToolStatus
from app.services.tools import checkpoint_store, tool_runtime


def test_tool_context_receives_graph_thread_and_run_ids(tmp_path, monkeypatch):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    captured = {}

    async def execute(call, context):
        captured.update({"call": call, "thread_id": context.thread_id, "run_id": context.run_id})
        return ToolResult(tool="recommend", status=ToolStatus.EMPTY, summary="empty")

    monkeypatch.setattr(tool_runtime, "execute", execute)
    asyncio.run(_run_tool_async(
        agent,
        ToolCall(name="recommend", arguments={"query": "深夜", "top_k": 5}),
        AgentPlan(intent="recommend", tools_needed=["recommend"]),
        "深夜",
        "user-1",
        5,
        [], [], [], [],
        "thread-custom", "run-custom", False,
    ))

    assert captured["thread_id"] == "thread-custom"
    assert captured["run_id"] == "run-custom"


def test_confirmation_checkpoint_uses_active_thread(tmp_path, monkeypatch):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    captured = {}

    async def execute(call, _context):
        return ToolResult(
            tool="favorite_track",
            status=ToolStatus.CONFIRMATION_REQUIRED,
            summary="confirm",
            data={"action_id": call.call_id},
        )

    def put(action_id, thread_id, user_id, tool, arguments, query):
        captured.update({
            "action_id": action_id, "thread_id": thread_id, "user_id": user_id,
            "tool": tool, "arguments": arguments, "query": query,
        })

    monkeypatch.setattr(tool_runtime, "execute", execute)
    monkeypatch.setattr(checkpoint_store, "put", put)
    call = ToolCall(call_id="action-custom", name="favorite_track", arguments={"track_id": "42"})
    asyncio.run(_run_tool_async(
        agent, call, AgentPlan(intent="feedback", tools_needed=["favorite_track"]),
        "收藏这首歌", "user-1", 5, [], [], [], [],
        "thread-custom", "run-custom", False,
    ))

    assert captured["action_id"] == "action-custom"
    assert captured["thread_id"] == "thread-custom"


def test_async_chat_forwards_trace_context_to_graph(tmp_path):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    captured = {}

    class Graph:
        async def astream(self, **kwargs):
            captured.update(kwargs)
            yield StreamEvent(type="final", content="ok")

    agent.graph = Graph()

    async def collect():
        return [event async for event in agent.stream_chat_async(
            "user-1", "你好", thread_id="thread-custom", run_id="run-custom",
        )]

    events = asyncio.run(collect())
    assert events[-1].type == "final"
    assert captured["thread_id"] == "thread-custom"
    assert captured["run_id"] == "run-custom"
