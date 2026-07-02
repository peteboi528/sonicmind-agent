"""听歌记录读取端点 GET /history/listening 与 record_listen 展示元数据内联的测试。

覆盖四条路径：
- 新格式事件（写入时带 title/source/source_id）→ 端点直接回显，available=True；
- 旧格式事件（只有 asset_id）→ 端点用曲库回查补 title/artist/cover；
- 在线曲 source_id 不在库里、又没带 title（旧在线事件）→ available=False；
- 倒序：最近一次播放排最前。
另直接断言 record_listen 把元数据落进 ListeningEvent。
"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.main import agent, app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_listening_history_new_metadata_online_track(client):
    """新格式：在线曲写入时带 title/source/source_id → 端点直接回显，available=True。"""
    uid = "lstn-new"
    agent.record_listen(
        uid, "netease-123", duration=210, completed=True,
        title="夜曲", artist="周杰伦", cover_url="https://img/1.jpg",
        source="netease", source_id="123",
    )
    resp = await client.get(f"/history/listening/{uid}")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    it = items[0]
    assert it["title"] == "夜曲"
    assert it["artist"] == "周杰伦"
    assert it["cover_url"] == "https://img/1.jpg"
    assert it["source"] == "netease"
    assert it["source_id"] == "123"
    assert it["available"] is True
    assert it["completed"] is True
    assert it["duration_listened"] == 210


@pytest.mark.anyio
async def test_listening_history_old_event_resolved_via_library(client):
    """旧格式：只传 asset_id（无 title），端点用曲库回查补 title/artist/cover。"""
    uid = "lstn-old"
    asset = agent.ingest_video("https://example.com/lstn-old-track")
    asset.title = "图书馆里这首歌"
    asset.artist = "某歌手"
    asset.cover_url = "https://img/cover.jpg"
    agent.store.write_model("assets", asset.asset_id, asset)
    agent._invalidate_assets_cache()  # list_assets 有缓存，写回后须失效

    # 旧客户端只传 asset_id，不带展示元数据
    agent.record_listen(uid, asset.asset_id, duration=60, completed=False)

    resp = await client.get(f"/history/listening/{uid}")
    items = resp.json()["items"]
    assert len(items) == 1
    it = items[0]
    assert it["title"] == "图书馆里这首歌"
    assert it["artist"] == "某歌手"
    assert it["cover_url"] == "https://img/cover.jpg"
    assert it["available"] is True


@pytest.mark.anyio
async def test_listening_history_unresolvable_online_available_false(client):
    """在线曲 source_id 不在库里、又没带 title（旧在线事件）→ available=False、title 空。"""
    uid = "lstn-unknown"
    agent.record_listen(uid, "netease-999", duration=5, completed=False)
    resp = await client.get(f"/history/listening/{uid}")
    items = resp.json()["items"]
    assert len(items) == 1
    assert items[0]["available"] is False
    assert items[0]["title"] == ""


@pytest.mark.anyio
async def test_listening_history_descending_order(client):
    """倒序：最近一次播放排最前。"""
    uid = "lstn-order"
    agent.record_listen(uid, "a-1", duration=10, completed=True, title="第一首")
    agent.record_listen(uid, "a-2", duration=20, completed=True, title="第二首")
    agent.record_listen(uid, "a-3", duration=30, completed=True, title="第三首")
    resp = await client.get(f"/history/listening/{uid}")
    items = resp.json()["items"]
    assert [it["title"] for it in items] == ["第三首", "第二首", "第一首"]


@pytest.mark.anyio
async def test_record_listen_persists_display_metadata():
    """record_listen 把展示元数据落进 ListeningEvent（不止 asset_id）。"""
    uid = "lstn-persist"
    memory = agent.record_listen(
        uid, "netease-555", duration=180, completed=True,
        title="晴天", artist="周杰伦", cover_url="https://img/q.jpg",
        source="netease", source_id="555",
    )
    ev = memory.listening_history[-1]
    assert ev.title == "晴天"
    assert ev.artist == "周杰伦"
    assert ev.cover_url == "https://img/q.jpg"
    assert ev.source == "netease"
    assert ev.source_id == "555"
