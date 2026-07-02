"""端到端验证：多意图并行走完整生产 graph（flag on），产两段 + 双 payload。

这是对手动 web 测试的离线镜像——不打真网络，用脚本化 LLM 让 planner 吐出 secondary，
再跑真实 build_stream_graph，断言 final 事件里 cards 与 dossier 同时非空。
放 tests/ 下随套件跑（默认 flag off 时本文件用 monkeypatch 显式开）。
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.config import settings
from app.graph import nodes
from tests.offline_fakes import start_offline_patches


@pytest.fixture
def offline():
    stack = start_offline_patches()
    yield
    stack.close()


class _ScriptedLLM:
    """planner 阶段吐一个带 secondary 的 query_plan JSON；其余（引言/合成）走普通文本。"""

    def __init__(self):
        self.calls = 0

    async def agenerate(self, prompt: str, system: str = "", **kwargs) -> str:
        # query_plan 调用：system 里含"意图规划器"。返回双意图 payload。
        if "意图规划器" in (system or "") or "intent" in (system or ""):
            return json.dumps({
                "intent": "recommend",
                "entities": ["The Weeknd"],
                "use_local": True,
                "use_vector": False,
                "use_web": True,
                "search_query": "The Weeknd",
                "search_variants": [],
                "language": "",
                "target_count": None,
                "reasoning": "推荐+专辑解读双意图",
                "secondary": {
                    "intent": "album_deep_dive",
                    "entities": ["Starboy"],
                    "search_query": "The Weeknd Starboy",
                },
            }, ensure_ascii=False)
        return "为你整理了 The Weeknd 的候选。"

    async def agenerate_stream(self, prompt: str, **kwargs):
        yield "为你推荐了 The Weeknd 的歌，"
        yield "下面讲讲 Starboy 这张专辑。"


def test_full_graph_multi_intent_yields_both_segments(offline, monkeypatch):
    monkeypatch.setattr(settings, "enable_multi_intent", True)

    scripted = _ScriptedLLM()
    # planner 用 fast tier，合成用 default——统一注入脚本化 LLM。
    monkeypatch.setattr(nodes, "select_llm", lambda *_a, **_k: scripted)

    from app.agent import AudioVisualAgent
    from app.storage import JsonStore

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        agent = AudioVisualAgent(JsonStore(td + "/store"))
        # 脚本化 planner 的实体解析走 fast LLM；把 agent 内的也换掉，保证 query_plan 走脚本。
        agent.llm = scripted
        agent.llm_fast = scripted

        async def _drive():
            final_payload = None
            texts = []
            async for event in agent.stream_chat_async("u-e2e", "推几首 The Weeknd，顺便讲讲他的 Starboy 专辑"):
                etype = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)
                payload = event.get("payload") if isinstance(event, dict) else getattr(event, "payload", None)
                content = event.get("content") if isinstance(event, dict) else getattr(event, "content", None)
                if etype == "token" and content:
                    texts.append(content)
                if etype == "final":
                    final_payload = payload
            return final_payload, "".join(texts)

        final_payload, _text = asyncio.run(_drive())

    assert final_payload is not None, "没有收到 final 事件"
    # 核心断言：一条 message 同时带 track cards 与知识 dossier
    assert final_payload.get("cards"), f"缺少 track cards；payload keys={list(final_payload.keys())}"
    assert final_payload.get("dossier"), f"缺少 dossier；payload keys={list(final_payload.keys())}"
