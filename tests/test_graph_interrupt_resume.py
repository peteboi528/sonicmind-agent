from __future__ import annotations

import asyncio

import aiosqlite
import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph

from app.agent import AudioVisualAgent
from app.graph.builder import AgentGraphRunner
from app.graph.nodes import execute_tools_async
from app.graph.state import AgentState
from app.models import AgentPlan, StreamEvent, ToolStage
from app.storage import JsonStore
from app.tools.checkpoints import ActionCheckpointStore
from app.tools.contracts import ToolCall, ToolResult, ToolStatus
from app.tools.registry import TOOL_REGISTRY
from app.services.tools import checkpoint_store


@pytest.mark.parametrize("approved,expected_status,expected_calls", [(True, "ok", 1), (False, "cancelled", 0)])
def test_checkpointed_graph_interrupts_and_resumes_external_write(
    tmp_path, monkeypatch, approved, expected_status, expected_calls,
):
    calls = {"count": 0, "arguments": {}}
    spec = TOOL_REGISTRY["favorite_track"]
    original = spec.handler

    def handler(arguments, _context):
        calls["count"] += 1
        calls["arguments"] = arguments
        return ToolResult(tool="favorite_track", status=ToolStatus.OK, summary="done")

    spec.handler = handler
    ledger = ActionCheckpointStore(tmp_path / "actions.sqlite")
    monkeypatch.setattr(checkpoint_store, "put", ledger.put)
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    action_id = f"action-{approved}"
    thread_id = f"thread-{approved}"
    plan = AgentPlan(
        intent="feedback",
        tools_needed=["favorite_track"],
        stages=[ToolStage(calls=[ToolCall(
            call_id=action_id,
            name="favorite_track",
            arguments={"track_id": "42"},
        )], parallel=False)],
    )

    async def run():
        connection = await aiosqlite.connect(tmp_path / f"graph-{approved}.sqlite")

        async def emitting_execute(state):
            before = len(state.get("events", []))
            output = await execute_tools_async(agent, state)
            writer = get_stream_writer()
            for event in output.get("events", [])[before:]:
                writer(event.model_dump(mode="json"))
            return output

        graph = StateGraph(AgentState)
        graph.add_node("execute_tools", emitting_execute)
        graph.set_entry_point("execute_tools")
        graph.add_edge("execute_tools", END)
        compiled = graph.compile(checkpointer=AsyncSqliteSaver(connection))
        runner = AgentGraphRunner(agent)
        runner.compiled_stream = compiled
        runner._checkpoint_connection = connection
        config = {"configurable": {"thread_id": thread_id}}
        state = {
            "user_id": "user-1",
            "run_id": "run-1",
            "thread_id": thread_id,
            "query": "收藏这首歌",
            "top_k": 5,
            "plan": plan,
            "results": [],
            "trace": [],
            "events": [],
            "tool_outcomes": [],
            "_refine_count": 0,
            "_interrupt_enabled": True,
        }
        initial = [item async for item in compiled.astream(state, config=config, stream_mode="custom")]
        resolved = ledger.resolve(action_id, thread_id, "user-1", approved)
        resumed = [item async for item in runner.resume(
            thread_id=thread_id, action_id=action_id, approved=approved,
        )]
        runner._checkpoint_connection = None
        await connection.close()
        return initial, resumed, resolved

    try:
        initial, resumed, resolved = asyncio.run(run())
    finally:
        spec.handler = original

    assert resolved is not None
    assert any(item["type"] == "confirmation_required" for item in initial)
    result_event = next(event for event in resumed if isinstance(event, StreamEvent) and event.type == "tool_result")
    assert result_event.payload["status"] == expected_status
    assert calls["count"] == expected_calls
    if approved:
        assert calls["arguments"]["confirm"] is True
