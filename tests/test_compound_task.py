"""复合任务检测 + compound graph 调度测试。"""
from __future__ import annotations

import time

import pytest

from app.agent import AudioVisualAgent
from app.compound import is_compound_task
from app.config import settings
from app.graph.builder import _compose_compound_answer, _hydrate_subtask_query, _run_compound_subtasks
from app.graph.decompose import SubTask, decompose_compound
from app.models import AgentAnswer, StreamEvent
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


def test_chat_routes_compound_to_graph_compound(agent, monkeypatch):
    _force_real_llm(monkeypatch)
    if agent.graph is None:
        pytest.skip("graph 不可用")
    calls = {"compound": 0, "react": 0}

    def fake_compound(**kw):
        calls["compound"] += 1
        return _ans("compound graph answer", ["[compound_plan]", "import_netease_playlist", "recommend"])

    monkeypatch.setattr(agent.graph, "invoke_compound", fake_compound)

    def fail_invoke(**kw):
        raise AssertionError("复合查询不应走 graph.invoke")
    monkeypatch.setattr(agent.graph, "invoke", fail_invoke)
    monkeypatch.setattr(agent.react, "run", lambda **kw: calls.__setitem__("react", calls["react"] + 1) or _ans("react"))

    answer = agent.chat("u", "导入网易云歌单，然后推荐适合跑步的")
    assert calls["compound"] == 1
    assert calls["react"] == 0
    assert "compound graph" in answer.answer


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
    called = {"graph": False, "compound": 0, "react": 0}

    def fake_graph_invoke(**kw):
        called["graph"] = True
        return _ans("graph", ["import"])

    def fake_compound(**kw):
        called["compound"] += 1
        return _ans("compound")

    def fake_react(**kw):
        called["react"] += 1
        return _ans("react")

    monkeypatch.setattr(agent.graph, "invoke", fake_graph_invoke)
    monkeypatch.setattr(agent.graph, "invoke_compound", fake_compound)
    monkeypatch.setattr(agent.react, "run", fake_react)
    agent.chat("u", "导入歌单然后推荐跑步的")
    assert called["graph"]
    assert called["compound"] == 0
    assert called["react"] == 0


def test_stream_chat_routes_compound_to_graph_stream(agent, monkeypatch):
    _force_real_llm(monkeypatch)
    if agent.graph is None:
        pytest.skip("graph 不可用")
    called = {"compound_stream": 0}

    def fake_stream_compound(**kw):
        from app.models import StreamEvent

        called["compound_stream"] += 1
        yield StreamEvent(type="plan", content="复合任务拆解为：")
        yield StreamEvent(type="final", content="compound stream answer", payload={"answer": "compound stream answer"})

    monkeypatch.setattr(agent.graph, "stream_compound", fake_stream_compound)
    events = list(agent.stream_chat("u", "导入网易云歌单，然后推荐适合跑步的"))

    assert called["compound_stream"] == 1
    assert events[0].type == "plan"
    assert events[-1].type == "final"


def test_decompose_compound_prefers_structured_llm(agent):
    tasks = decompose_compound(
        agent,
        "导入网易云歌单，然后推荐适合跑步的，再做个歌单",
        history=None,
    )

    assert len(tasks) >= 2
    assert tasks[0].intent in {"import", "recommend"}
    assert any(task.depends_on_prev for task in tasks[1:])


def test_hydrate_subtask_query_uses_explicit_scratchpad_summary():
    task = SubTask(intent="playlist", query="基于上一步结果做个夜跑歌单", depends_on_prev=True)
    scratchpad = {
        "last_query": "推荐几首适合跑步的歌",
        "last_summary": "已经找到 5 首适合跑步的真实候选。",
        "last_answer": "长答案正文",
    }

    hydrated = _hydrate_subtask_query(task, scratchpad)

    assert "上一步任务" in hydrated
    assert "上一步摘要" in hydrated
    assert "适合跑步的歌" in hydrated


def test_compound_independent_subtasks_run_parallel(agent, monkeypatch):
    if agent.graph is None:
        pytest.skip("graph 不可用")
    monkeypatch.setattr(settings, "enable_parallel_tools", True, raising=False)

    def fake_invoke(_user_id, _asset_id, query, history=None, top_k=5):
        time.sleep(0.1)
        return {"answer": _ans(query, [query]), "events": []}

    monkeypatch.setattr(agent.graph, "_invoke_state", fake_invoke)
    subtasks = [
        SubTask(intent="recommend", query="推荐跑步歌"),
        SubTask(intent="taste", query="分析我的品味"),
    ]

    start = time.monotonic()
    out = _run_compound_subtasks(agent.graph, subtasks, {}, "u", None, None, 5)
    elapsed = time.monotonic() - start

    assert [item[2] for item in out] == ["推荐跑步歌", "分析我的品味"]
    assert elapsed < 0.18


def test_compound_dependent_subtask_waits_for_scratchpad(agent, monkeypatch):
    if agent.graph is None:
        pytest.skip("graph 不可用")
    monkeypatch.setattr(settings, "enable_parallel_tools", True, raising=False)
    seen_queries: list[str] = []

    def fake_invoke(_user_id, _asset_id, query, history=None, top_k=5):
        seen_queries.append(query)
        return {"answer": _ans("已经找到 5 首适合跑步的真实候选。", [query]), "events": []}

    monkeypatch.setattr(agent.graph, "_invoke_state", fake_invoke)
    subtasks = [
        SubTask(intent="recommend", query="推荐跑步歌"),
        SubTask(intent="playlist", query="基于上一步做歌单", depends_on_prev=True),
    ]

    _run_compound_subtasks(agent.graph, subtasks, {}, "u", None, None, 5)

    assert len(seen_queries) == 2
    assert "上一步任务" in seen_queries[1]
    assert "推荐跑步歌" in seen_queries[1]


def test_compound_answer_prefers_llm_synthesis(agent):
    subtasks = [
        SubTask(intent="import", query="导入网易云歌单"),
        SubTask(intent="recommend", query="推荐适合跑步的歌", depends_on_prev=True),
    ]
    answers = [
        _ans("已导入 20 首歌。"),
        _ans("我挑出了 5 首适合跑步的歌。"),
    ]

    text, prompt_versions, runtime_metrics = _compose_compound_answer(agent, "导入歌单然后推荐跑步歌", subtasks, answers)

    assert "分步处理好了" in text
    assert "我把" not in text
    assert prompt_versions["compound_synth"].startswith("v")
    assert "llm_calls" in runtime_metrics


def test_stream_compound_final_payload_merges_cards(agent, monkeypatch):
    if agent.graph is None:
        pytest.skip("graph 不可用")

    subtask_states = [
        {
            "answer": _ans("先给你两首适合跑步的歌"),
            "events": [
                StreamEvent(
                    type="final",
                    content="先给你两首适合跑步的歌",
                    payload={
                        "answer": "先给你两首适合跑步的歌",
                        "cards": [
                            {"title": "Song A", "artist": "Artist A", "source": "netease", "source_id": "1"},
                            {"title": "Song B", "artist": "Artist B", "source": "netease", "source_id": "2"},
                        ],
                    },
                )
            ],
        },
        {
            "answer": _ans("再补一首类似风格的歌"),
            "events": [
                StreamEvent(
                    type="final",
                    content="再补一首类似风格的歌",
                    payload={
                        "answer": "再补一首类似风格的歌",
                        "cards": [
                            {"title": "Song B", "artist": "Artist B", "source": "netease", "source_id": "2"},
                            {"title": "Song C", "artist": "Artist C", "source": "netease", "source_id": "3"},
                        ],
                    },
                )
            ],
        },
    ]

    monkeypatch.setattr(
        "app.graph.builder.decompose_compound",
        lambda *_args, **_kwargs: [
            SubTask(intent="recommend", query="推荐跑步歌"),
            SubTask(intent="recommend", query="再来一首类似的", depends_on_prev=True),
        ],
    )

    calls = iter(subtask_states)
    monkeypatch.setattr(agent.graph, "_invoke_state", lambda *_args, **_kw: next(calls))

    events = list(agent.graph.stream_compound("u", None, "推荐跑步歌然后再来一首类似的"))

    final = events[-1]
    assert final.type == "final"
    assert "cards" in final.payload
    assert [card["title"] for card in final.payload["cards"]] == ["Song A", "Song B", "Song C"]


def test_compound_answer_carries_runtime_metrics(agent, monkeypatch):
    if agent.graph is None:
        pytest.skip("graph 不可用")

    monkeypatch.setattr(
        "app.graph.builder.decompose_compound_with_meta",
        lambda *_args, **_kwargs: (
            [SubTask(intent="recommend", query="推荐跑步歌")],
            {"decompose": "v-test"},
            {"llm_calls": 1, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "latency_ms": 0.0, "estimated_cost_usd": 0.0},
        ),
    )
    monkeypatch.setattr(
        agent.graph,
        "_invoke_state",
        lambda *_args, **_kwargs: {
            "answer": AgentAnswer(
                answer="给你一首歌",
                evidences=[],
                recommended_tracks=[],
                runtime_metrics={"llm_calls": 2, "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "latency_ms": 12.3, "estimated_cost_usd": 0.0},
            )
        },
    )

    answer = agent.graph.invoke_compound("u", None, "推荐跑步歌")

    assert answer.runtime_metrics["llm_calls"] >= 2
    assert any("[meta]" in line for line in answer.agent_trace)
