"""V2 工具路径多轮增强测试。

锁定延续去重 / 翻页取新歌 / search_variants 多路召回 / 语言加权过滤在 ToolRuntime
handler 链路中真正生效（经 ctx.plan 注入），以及 plan=None 的直接 Runtime 调用
时安全跳过、维持兜底行为。

这些逻辑原本只在旧同步 Graph 分支里实现，
而 V2 handler 不读 ctx.plan，导致默认 V2=True 路径下悄悄失效。移植到 handler 后由
``ctx.plan`` 防御性读取恢复——本文件锁住这条契约。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.models import ExternalTrack
from app.services.tools import tool_runtime
from app.tools.contracts import ToolCall, ToolContext


class _Library:
    def upsert_external(self, track) -> None:
        return None


class _FakeAgent:
    """记录各方法调用 kwargs + 返回可配置结果，让 handler 走真实 ToolRuntime 链路。"""

    library = _Library()

    def __init__(self) -> None:
        self.web_results: list[ExternalTrack] = []
        self.last_web: dict = {}
        self.last_rec: dict = {}
        self.search_response = SimpleNamespace(external=[], local=[])
        self.last_search: dict = {}
        self.playlist = SimpleNamespace(tracks=[])
        self.last_playlist: dict = {}

    def search_web_music(
        self, query, top_k=5, relevance_query="", include_video_sources=False, offset=0, variants=None
    ):
        self.last_web = dict(
            query=query, top_k=top_k, relevance_query=relevance_query, offset=offset, variants=variants
        )
        return list(self.web_results)

    async def search_web_music_async(
        self, query, top_k=5, relevance_query="", include_video_sources=False, offset=0, variants=None
    ):
        return self.search_web_music(query, top_k, relevance_query, include_video_sources, offset, variants)

    def recommend_for_query(
        self,
        user_id,
        goal,
        top_k=5,
        *,
        excluded_tracks=None,
        search_variants=None,
        seed_tracks=None,
        search_query_override=None,
        budget_degrade_level=None,
        entities=None,
    ):
        self.last_rec = dict(
            goal=goal,
            excluded_tracks=excluded_tracks,
            search_variants=search_variants,
            search_query_override=search_query_override,
            budget_degrade_level=budget_degrade_level,
            entities=entities,
        )
        return SimpleNamespace(tracks=[])

    def search(self, user_id, query, include_external=True, top_k=12, offset=0):
        self.last_search = dict(offset=offset)
        return self.search_response

    def generate_playlist(self, user_id, instruction, seed_tracks=None, target_count=None):
        self.last_playlist = dict(instruction=instruction)
        return self.playlist


def _track(title: str, source_id: str = "1", source: str = "netease", artist: str = "Artist") -> ExternalTrack:
    return ExternalTrack(external_id=source_id, title=title, artist=artist, source=source)


def _plan(*, excluded=None, variants=None, search_query="", language="", target=None, entities=None) -> dict:
    """模拟 graph V2 分支传入的 plan_payload（plan.model_dump 后的 dict）。"""
    return {
        "_excluded_tracks": excluded or [],
        "target_count": target,
        "retrieval_plan": {
            "search_query": search_query,
            "search_variants": variants,
            "language_filter": language,
            "entities": entities or [],
        },
    }


def _ctx(agent: _FakeAgent, plan: dict | None, query: str = "relaxing music") -> ToolContext:
    return ToolContext(thread_id="t1", user_id="u1", query=query, plan=plan, agent=agent)


def _exec(name: str, arguments: dict, ctx: ToolContext):
    return asyncio.run(tool_runtime.execute(ToolCall(name=name, arguments=arguments), ctx))


def test_web_search_excludes_shown_tracks():
    agent = _FakeAgent()
    agent.web_results = [_track("Duplicate", "1"), _track("Fresh", "2")]
    ctx = _ctx(agent, _plan(excluded=[{"title": "Duplicate", "source_id": "1"}]))
    result = _exec("web_music_search", {"query": "relaxing music", "top_k": 5}, ctx)
    titles = [t.title for t in result.data["tracks"]]
    assert "Duplicate" not in titles
    assert "Fresh" in titles


def test_web_search_pages_offset_for_excluded():
    agent = _FakeAgent()
    agent.web_results = []
    excluded = [{"title": f"T{i}", "source_id": str(i)} for i in range(3)]
    ctx = _ctx(agent, _plan(excluded=excluded))
    _exec("web_music_search", {"query": "music", "top_k": 5}, ctx)
    assert agent.last_web["offset"] == 3


def test_web_search_passes_variants():
    agent = _FakeAgent()
    agent.web_results = []
    ctx = _ctx(agent, _plan(variants=["chill beats", "lofi"]))
    _exec("web_music_search", {"query": "music", "top_k": 5}, ctx)
    assert agent.last_web["variants"] == ["chill beats", "lofi"]


def test_web_search_applies_language_filter():
    agent = _FakeAgent()
    # 一首中文（晴天，CJK 主导→zh）一首英文（Hello，latin 主导→en）；language=en、target=2
    # 时英文恰好达到 target 一半（1/1），过滤生效，只剩英文。
    agent.web_results = [_track("晴天", "1", artist="周杰伦"), _track("Hello", "2", artist="Adele")]
    ctx = _ctx(agent, _plan(language="en", target=2))
    result = _exec("web_music_search", {"query": "music", "top_k": 2}, ctx)
    assert [t.title for t in result.data["tracks"]] == ["Hello"]


def test_recommend_passes_excluded_and_variants():
    agent = _FakeAgent()
    ctx = _ctx(
        agent, _plan(excluded=[{"title": "X", "source_id": "9"}], variants=["synonym"], search_query="chill r&b")
    )
    result = _exec("recommend", {"query": "music", "top_k": 5}, ctx)
    assert agent.last_rec["excluded_tracks"] == [{"title": "X", "source_id": "9"}]
    assert agent.last_rec["search_variants"] == ["synonym"]
    assert agent.last_rec["search_query_override"] == "chill r&b"
    assert result.status.value == "empty"


def test_recommend_forwards_llm_entities_as_anchors():
    """回归：plan.retrieval_plan.entities 必须传给 recommend_for_query，
    才能让长尾歌手（不在硬编码名单里的 The Weeknd）触发艺人锚点过滤。"""
    agent = _FakeAgent()
    ctx = _ctx(agent, _plan(search_query="The Weeknd", entities=["The Weeknd"]))
    _exec("recommend", {"query": "music", "top_k": 5}, ctx)
    assert agent.last_rec["entities"] == ["The Weeknd"]


def test_recommend_passes_budget_degrade_level():
    agent = _FakeAgent()
    ctx = ToolContext(
        thread_id="t1",
        user_id="u1",
        query="music",
        plan=_plan(search_query="chill r&b"),
        agent=agent,
        latency_budget={"budget_degrade_level": "soft"},
    )
    _exec("recommend", {"query": "music", "top_k": 5}, ctx)
    assert agent.last_rec["budget_degrade_level"] == "soft"


def test_search_filters_external_by_excluded():
    agent = _FakeAgent()
    agent.search_response = SimpleNamespace(
        external=[_track("Dup", "1"), _track("New", "2")],
        local=[_track("LocalKeep", "3")],
    )
    ctx = _ctx(agent, _plan(excluded=[{"title": "Dup", "source_id": "1"}]))
    _exec("search", {"query": "music"}, ctx)
    assert [t.title for t in agent.search_response.external] == ["New"]
    # local 不受去重影响
    assert [t.title for t in agent.search_response.local] == ["LocalKeep"]
    assert agent.last_search["offset"] == 1


def test_playlist_filters_tracks_by_excluded():
    agent = _FakeAgent()
    agent.playlist = SimpleNamespace(tracks=[_track("Dup", "1"), _track("Keep", "2")])
    ctx = _ctx(agent, _plan(excluded=[{"title": "Dup", "source_id": "1"}]))
    result = _exec("playlist", {"instruction": "late night"}, ctx)
    assert [t.title for t in result.data["playlist"].tracks] == ["Keep"]


def test_no_plan_skips_enhancement():
    """plan=None（直接 Runtime 调用）时跳过所有增强，维持兜底行为。"""
    agent = _FakeAgent()
    agent.web_results = [_track("A", "1")]
    ctx = _ctx(agent, plan=None)
    _exec("web_music_search", {"query": "music", "top_k": 5}, ctx)
    assert agent.last_web["offset"] == 0
    assert agent.last_web["variants"] is None
