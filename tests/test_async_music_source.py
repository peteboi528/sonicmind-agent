from __future__ import annotations

import asyncio

import pytest

from app.agent import AudioVisualAgent
from app.graph.builder import AgentGraphRunner
from app.models import ExternalTrack
from app.sources import bilibili as bilibili_source
from app.sources import web_search as web_search_source
from app.sources import youtube as youtube_source
from app.sources.http_transport import source_transport
from app.sources.netease import asearch_netease_many
from app.storage import JsonStore
from app.tools.contracts import ToolCall, ToolContext, ToolResult, ToolStatus
from app.tools.handlers import _web_music_search, _web_music_search_async
from app.tools.registry import TOOL_REGISTRY
from app.tools.runtime import ToolRuntime

_REAL_ASYNC_VIDEO_SEARCH = AudioVisualAgent.search_videos_async
_REAL_ASYNC_WEB_SEARCH = AudioVisualAgent.search_web_music_async


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_async_netease_queries_endpoints_concurrently(monkeypatch):
    active = 0
    max_active = 0

    async def request(_source, _method, url, **_kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        songs = [{
            "id": 1, "name": "Night Song", "artists": [{"name": "Artist"}],
            "album": {"name": "Album", "picUrl": "cover"},
        }] if "cloudsearch" in url else []
        return _Response({"result": {"songs": songs}})

    monkeypatch.setattr(source_transport, "request", request)
    result = asyncio.run(asearch_netease_many("night", limit=5))

    assert max_active >= 2
    assert result == [{
        "song_id": "1", "title": "Night Song", "artist": "Artist",
        "album": "Album", "cover": "cover",
    }]


def test_async_netease_cancellation_reaches_all_requests(monkeypatch):
    cancelled = 0
    started = 0
    all_started = asyncio.Event()

    async def request(*_args, **_kwargs):
        nonlocal cancelled, started
        started += 1
        if started == 3:
            all_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled += 1

    monkeypatch.setattr(source_transport, "request", request)

    async def run():
        task = asyncio.create_task(asearch_netease_many("night"))
        await all_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert cancelled == 3


def test_async_runtime_prefers_async_handler():
    spec = TOOL_REGISTRY["web_music_search"]
    original_sync = spec.handler
    original_async = spec.async_handler
    calls = {"sync": 0, "async": 0}

    def sync_handler(_args, _context):
        calls["sync"] += 1
        return ToolResult(tool=spec.name, status=ToolStatus.OK)

    async def async_handler(_args, _context):
        calls["async"] += 1
        return ToolResult(tool=spec.name, status=ToolStatus.OK)

    spec.handler = sync_handler
    spec.async_handler = async_handler
    try:
        result = asyncio.run(ToolRuntime().execute(
            ToolCall(name=spec.name, arguments={"query": "night", "top_k": 5}),
            ToolContext(thread_id="t", user_id="u", query="night"),
        ))
    finally:
        spec.handler = original_sync
        spec.async_handler = original_async

    assert result.status == ToolStatus.OK
    assert calls == {"sync": 0, "async": 1}


def test_async_web_music_fallback_uses_keyword_arguments(tmp_path, monkeypatch):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))

    async def boom(*_args, **_kwargs):
        raise RuntimeError("offline")

    called = {}

    def fake_sync(*, query, top_k=5, relevance_query="", include_video_sources=False, offset=0, variants=None):
        called.update({
            "query": query,
            "top_k": top_k,
            "relevance_query": relevance_query,
            "include_video_sources": include_video_sources,
            "offset": offset,
            "variants": variants,
        })
        return [_track for _track in [
            ExternalTrack(external_id="1", title="Night Song", artist="Artist", source="netease")
        ]]

    monkeypatch.setattr("app.sources.netease.asearch_netease_many", boom)
    monkeypatch.setattr(agent, "search_web_music", fake_sync)

    result = asyncio.run(_REAL_ASYNC_WEB_SEARCH(
        agent,
        "night run",
        top_k=4,
        relevance_query="night",
        include_video_sources=True,
        offset=2,
        variants=None,
    ))

    assert called == {
        "query": "night run",
        "top_k": 4,
        "relevance_query": "night",
        "include_video_sources": True,
        "offset": 2,
        "variants": None,
    }
    assert result[0].title == "Night Song"


def test_streaming_graph_uses_async_music_handler(tmp_path, monkeypatch):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    tracks = [
        ExternalTrack(
            external_id=str(index), title=f"Night {index}", artist="Artist",
            source="netease", playback_url=f"https://music.163.com/song?id={index}",
        )
        for index in range(1, 6)
    ]
    calls = {"async": 0}

    async def async_search(*_args, **_kwargs):
        calls["async"] += 1
        return tracks

    def forbidden_sync(*_args, **_kwargs):
        raise AssertionError("streaming graph used sync web search")

    monkeypatch.setattr(agent, "search_web_music_async", async_search)
    monkeypatch.setattr(agent, "search_web_music", forbidden_sync)

    async def collect():
        return [event async for event in AgentGraphRunner(agent).astream(
            "u", None, "推荐几首适合深夜的歌", thread_id="thread-async",
        )]

    events = asyncio.run(collect())
    assert calls["async"] >= 1
    assert any(event.type == "candidates" for event in events)
    assert events[-1].type == "final"


def test_streaming_graph_cancellation_reaches_async_handler(tmp_path, monkeypatch):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    started = asyncio.Event()
    cancelled = {"value": False}

    async def waiting_search(*_args, **_kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled["value"] = True

    monkeypatch.setattr(agent, "search_web_music_async", waiting_search)

    async def run():
        async def consume():
            async for _event in AgentGraphRunner(agent).astream(
                "u", None, "推荐几首适合深夜的歌", thread_id="thread-cancel",
            ):
                pass

        task = asyncio.create_task(consume())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert cancelled["value"] is True


def test_sync_and_async_web_handlers_share_result_contract(tmp_path, monkeypatch):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    track = ExternalTrack(
        external_id="1", title="Night", artist="Artist", source="netease",
        playback_url="https://music.163.com/song?id=1",
    )
    monkeypatch.setattr(agent, "search_web_music", lambda *_args, **_kwargs: [track])

    async def async_search(*_args, **_kwargs):
        return [track]

    monkeypatch.setattr(agent, "search_web_music_async", async_search)
    context = ToolContext(
        thread_id="t", user_id="u", query="night", agent=agent,
        plan={"target_count": 1, "retrieval_plan": {"search_query": "night"}},
    )
    arguments = {"query": "night", "top_k": 1}

    sync_result = _web_music_search(arguments, context)
    async_result = asyncio.run(_web_music_search_async(arguments, context))

    assert async_result.status == sync_result.status == ToolStatus.OK
    assert async_result.data == sync_result.data
    assert async_result.cards == sync_result.cards
    assert async_result.provenance == sync_result.provenance


def test_async_video_sources_run_concurrently_and_keep_source_order(tmp_path, monkeypatch):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    active = 0
    max_active = 0

    async def bili(*_args, **_kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return [{"bvid": "BV1", "title": "Bili Live", "author": "Singer"}]

    async def youtube(*_args, **_kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return [{"video_id": "abcdefghijk", "title": "YouTube Live"}]

    monkeypatch.setattr(bilibili_source, "asearch_bilibili_many", bili)
    monkeypatch.setattr(youtube_source, "asearch_youtube_many", youtube)
    result = asyncio.run(_REAL_ASYNC_VIDEO_SEARCH(agent, "live", top_k=2))

    assert max_active == 2
    assert [track.source for track in result] == ["bilibili", "youtube"]


def test_async_web_info_uses_tavily_then_returns_structured_results(monkeypatch):
    calls = []

    async def request(source, method, url, **kwargs):
        calls.append((source, method, url, kwargs))
        return _Response({"results": [{"title": "Artist", "content": "Bio", "url": "https://example.test"}]})

    monkeypatch.setattr(source_transport, "request", request)
    result = asyncio.run(web_search_source.asearch_web_info("Artist", api_key="key"))

    assert result == [{"title": "Artist", "content": "Bio", "url": "https://example.test"}]
    assert calls[0][0:2] == ("tavily", "POST")
    assert calls[0][3]["retries"] == 0


def test_streaming_video_intent_never_calls_sync_sources(tmp_path, monkeypatch):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))

    async def async_videos(*_args, **_kwargs):
        return [ExternalTrack(
            external_id="BV1", title="Live", artist="Singer", source="bilibili",
            playback_url="https://player.bilibili.com/player.html?bvid=BV1",
        )]

    monkeypatch.setattr(agent, "search_videos_async", async_videos)
    monkeypatch.setattr(agent, "search_videos", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("sync video search used")
    ))

    async def collect():
        return [event async for event in AgentGraphRunner(agent).astream(
            "u", None, "找一个现场 Live 视频", thread_id="thread-video",
        )]

    events = asyncio.run(collect())
    assert any(event.type == "candidates" for event in events)
    assert events[-1].type == "final"


def test_streaming_graph_never_calls_sync_llm_methods(tmp_path, monkeypatch):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))

    class AsyncOnlyLLM:
        last_stats = {}

        def generate(self, *_args, **_kwargs):
            raise AssertionError("streaming graph used sync LLM generate")

        def generate_stream(self, *_args, **_kwargs):
            raise AssertionError("streaming graph used sync LLM stream")

        async def agenerate(self, *_args, **_kwargs):
            return (
                '{"intent":"recommend","entities":[],"use_local":true,'
                '"use_vector":true,"use_web":true,"target_count":3,'
                '"reasoning":"深夜放松推荐","search_query":"late night chill",'
                '"search_variants":["night rnb"],"language":""}'
            )

        async def agenerate_stream(self, *_args, **_kwargs):
            yield "给你一组适合深夜放松的真实候选："

    llm = AsyncOnlyLLM()
    agent.llm = agent.llm_fast = agent.llm_strong = llm
    agent._llm_default_ref = llm

    async def collect():
        return [event async for event in AgentGraphRunner(agent).astream(
            "u", None, "来点深夜放松的歌曲", thread_id="thread-async-llm",
        )]

    events = asyncio.run(collect())
    assert any(event.type == "token" for event in events)
    assert events[-1].type == "final"


def test_missing_langgraph_has_no_secondary_fallback(tmp_path):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    runner = AgentGraphRunner(agent)
    runner.compiled_stream = None

    async def collect():
        with pytest.raises(RuntimeError, match="no secondary orchestrator"):
            return [event async for event in runner.astream(
                "u", None, "推荐三首深夜歌曲", thread_id="thread-async-fallback",
            )]

    asyncio.run(collect())
