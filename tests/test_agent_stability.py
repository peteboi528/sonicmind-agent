"""Agent 稳定性修复：图异常 final 契约与候选池并发写。"""
from __future__ import annotations

import asyncio

from app.agent import AudioVisualAgent
from app.graph.builder import AgentGraphRunner
from app.library import ResourceLibrary
from app.models import ExternalTrack
from app.storage import JsonStore


class _BrokenGraph:
    async def astream(self, *_args, **_kwargs):
        raise RuntimeError("internal provider detail")
        yield  # pragma: no cover -- keeps this an async generator


def test_graph_failure_still_emits_safe_final(tmp_path):
    runner = AgentGraphRunner(AudioVisualAgent(JsonStore(tmp_path / "store")))
    runner.compiled_stream = _BrokenGraph()

    async def collect():
        return [event async for event in runner.astream("u", None, "推荐音乐")]

    events = asyncio.run(collect())
    assert [event.type for event in events] == ["error", "final"]
    assert "internal provider detail" not in events[0].content
    assert events[-1].payload["fallback_reason"] == "graph_execution_failed"

    answer = asyncio.run(runner.ainvoke("u", None, "推荐音乐"))
    assert answer.fallback_reason == "graph_execution_failed"


def test_resource_library_serializes_concurrent_batch_writes(tmp_path):
    library = ResourceLibrary(tmp_path / "resource.sqlite")
    batches = [
        [ExternalTrack(external_id=f"{batch}-{i}", title=f"Song {batch}-{i}", artist="Artist", source="netease")
         for i in range(10)]
        for batch in range(8)
    ]

    async def write_all():
        await asyncio.gather(*(asyncio.to_thread(library.upsert_externals, batch) for batch in batches))

    asyncio.run(write_all())
    assert len(library.list_tracks(100)) == 80
    with library._connect() as conn:  # noqa: SLF001 - verify the connection mode promised by this regression.
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_cache_write_failure_is_non_fatal(tmp_path, monkeypatch):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    monkeypatch.setattr(agent.library, "upsert_externals", lambda _tracks: (_ for _ in ()).throw(OSError("locked")))

    # 缓存故障被吞掉；调用方继续保有原先已验证的曲目列表。
    tracks = [ExternalTrack(external_id="1", title="Hello", artist="Adele", source="netease")]
    assert agent._search_service()._cache_external_tracks(tracks) is None
