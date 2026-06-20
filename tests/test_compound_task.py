from __future__ import annotations

import asyncio

import pytest

from app.agent import AudioVisualAgent
from app.compound import is_compound_task
from app.graph.builder import _hydrate_subtask_query
from app.graph.decompose import SubTask, decompose_compound_async
from app.models import AgentAnswer, StreamEvent
from app.storage import JsonStore


@pytest.mark.parametrize("query", [
    "帮我导入网易云歌单，然后挑几首适合跑步的",
    "搜一下周杰伦，再推荐他类似风格的",
    "find my playlist then recommend similar songs",
])
def test_compound_detected(query):
    assert is_compound_task(query)


@pytest.mark.parametrize("query", ["推荐几首歌", "介绍周杰伦", "你好", "做一个跑步歌单"])
def test_simple_not_compound(query):
    assert not is_compound_task(query)


def test_decompose_compound_is_async(tmp_path):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    tasks, _, _ = asyncio.run(decompose_compound_async(
        agent, "导入网易云歌单，然后推荐适合跑步的，再做个歌单",
    ))
    assert len(tasks) >= 2
    assert any(task.depends_on_prev for task in tasks[1:])


def test_hydrate_subtask_query_uses_scratchpad():
    task = SubTask(intent="playlist", query="基于上一步结果做个夜跑歌单", depends_on_prev=True)
    hydrated = _hydrate_subtask_query(task, {
        "last_query": "推荐跑步歌", "last_summary": "已经找到真实候选。",
    })
    assert "上一步任务" in hydrated and "上一步摘要" in hydrated


def test_compound_uses_same_async_langgraph_subgraph(tmp_path, monkeypatch):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    calls: list[str] = []

    async def decompose(*_args, **_kwargs):
        return ([
            SubTask(intent="recommend", query="推荐跑步歌"),
            SubTask(intent="playlist", query="基于上一步做歌单", depends_on_prev=True),
        ], {"decompose": "v-test"}, {})

    async def single(_user, _asset, query, *_args, **_kwargs):
        calls.append(query)
        answer = AgentAnswer(answer=f"完成：{query}", evidences=[])
        yield StreamEvent(type="final", content=answer.answer, payload=answer.model_dump(mode="json"))

    monkeypatch.setattr("app.graph.builder.decompose_compound_async", decompose)
    monkeypatch.setattr(agent.graph, "_astream_single", single)

    async def collect():
        return [event async for event in agent.graph.astream(
            "u", None, "推荐跑步歌然后做歌单", thread_id="compound-thread",
        )]

    events = asyncio.run(collect())
    assert len(calls) == 2
    assert "上一步任务" in calls[1]
    assert events[-1].type == "final"
    assert any("[compound_plan]" in line for line in events[-1].payload["agent_trace"])
