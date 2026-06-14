"""复合任务检测 + Deep 模式调度测试。

is_compound_task 是纯函数（确定性，进 CI）；调度测试用 monkeypatch 强制非 mock 模式，
验证复合查询走 react、简单查询走图、enable_deep_mode=False 时复合也走图。
"""
from __future__ import annotations

import pytest

from app.agent import AudioVisualAgent
from app.compound import is_compound_task
from app.config import settings
from app.models import AgentAnswer
from app.storage import JsonStore

COMPOUND_QUERIES = [
    "帮我导入网易云歌单，然后挑几首适合跑步的",
    "搜一下周杰伦，再推荐他类似风格的",
    "导入我的歌单，然后分析曲风分布",
    "find my playlist then recommend similar songs",
    "先推荐几首，之后做个歌单",
]

SIMPLE_QUERIES = [
    "推荐几首歌",
    "找一首歌",
    "介绍周杰伦",
    "你好",
    "搜一下Beyond的摇滚",
    "做一个跑步歌单",
    "分析我的品味",
]


def _ans(answer: str, trace: list[str] | None = None) -> AgentAnswer:
    """构造测试用 AgentAnswer（evidences 必填，统一给空）。"""
    return AgentAnswer(answer=answer, evidences=[], agent_trace=trace or [])


@pytest.mark.parametrize("q", COMPOUND_QUERIES)
def test_compound_detected(q):
    assert is_compound_task(q), f"应判为复合: {q}"


@pytest.mark.parametrize("q", SIMPLE_QUERIES)
def test_simple_not_compound(q):
    assert not is_compound_task(q), f"不应判为复合: {q}"


def test_empty_query_not_compound():
    assert not is_compound_task("")
    assert not is_compound_task("   ")


@pytest.fixture
def agent(tmp_path):
    return AudioVisualAgent(JsonStore(tmp_path / "store"))


def _force_real_llm(monkeypatch):
    """让 settings.mock_mode 返回 False（Deep 模式分支才生效）。"""
    monkeypatch.setattr(settings, "enable_deep_mode", True)
    monkeypatch.setattr(settings, "llm_api_key", "fake-key")  # mock_mode = not llm_api_key → False


def test_chat_routes_compound_to_react(agent, monkeypatch):
    _force_real_llm(monkeypatch)
    calls = {"react": 0}

    def fake_react(**kw):
        calls["react"] += 1
        return _ans("deep mode answer", ["import_netease_playlist", "recommend"])

    monkeypatch.setattr(agent.react, "run", fake_react)

    def fail_invoke(**kw):
        raise AssertionError("复合查询不应走 graph.invoke")
    if agent.graph is not None:
        monkeypatch.setattr(agent.graph, "invoke", fail_invoke)

    answer = agent.chat("u", "导入网易云歌单，然后推荐适合跑步的")
    assert calls["react"] == 1
    assert "deep mode" in answer.answer


def test_chat_routes_simple_to_graph(agent, monkeypatch):
    _force_real_llm(monkeypatch)
    if agent.graph is None:
        pytest.skip("graph 不可用")
    called = {"graph": False}

    def fake_graph_invoke(**kw):
        called["graph"] = True
        return _ans("graph answer", ["recommend"])

    monkeypatch.setattr(agent.graph, "invoke", fake_graph_invoke)
    agent.chat("u", "推荐几首歌")
    assert called["graph"], "简单查询应走 graph.invoke"


def test_deep_mode_disabled_routes_compound_to_graph(agent, monkeypatch):
    """enable_deep_mode=False 时，复合查询也走图（Deep 模式可关）。"""
    monkeypatch.setattr(settings, "enable_deep_mode", False)
    monkeypatch.setattr(settings, "llm_api_key", "fake-key")
    if agent.graph is None:
        pytest.skip("graph 不可用")
    called = {"graph": False, "react": 0}

    def fake_graph_invoke(**kw):
        called["graph"] = True
        return _ans("graph", ["import"])

    def fake_react(**kw):
        called["react"] += 1
        return _ans("react")

    monkeypatch.setattr(agent.graph, "invoke", fake_graph_invoke)
    monkeypatch.setattr(agent.react, "run", fake_react)
    agent.chat("u", "导入歌单然后推荐跑步的")
    assert called["graph"]
    assert called["react"] == 0
