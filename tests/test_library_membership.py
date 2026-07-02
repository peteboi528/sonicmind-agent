"""库成员卫生：play≠入库、显式入库即分类、× 移除库内条目、import 跳过 disliked、污染清理。

离线隔离：AudioVisualAgent(JsonStore(tmp_path/"store"))，绝不触碰真实 data/store。
"""

from app.agent import AudioVisualAgent
from app.models import Asset, AssetStatus, DislikeRequest, ExternalTrack
from app.storage import JsonStore


def _agent(tmp_path):
    return AudioVisualAgent(JsonStore(tmp_path / "store"))


# ── A: 播放不入库（asset_id_for_track 是纯函数，不落盘）──────────────────────
def test_asset_id_for_track_does_not_persist(tmp_path):
    agent = _agent(tmp_path)
    track = ExternalTrack(
        external_id="111", title="t", artist="a", source="netease",
        playback_url="https://music.163.com/song?id=111",
    )
    assert agent.list_assets() == []
    logical_id = agent.library_svc.asset_id_for_track(track)
    assert logical_id  # 有稳定 id
    assert agent.list_assets() == []  # 但库里依旧空——播放没入库


def test_asset_id_for_track_matches_persisted_id(tmp_path):
    # 同一首歌：逻辑 id == 真入库后的 asset_id（将来 listen 才能对上）
    agent = _agent(tmp_path)
    track = ExternalTrack(external_id="111", title="t", artist="a", source="netease")
    logical = agent.library_svc.asset_id_for_track(track)
    asset = agent.library_svc.ensure_asset_from_track(track)
    assert asset is not None and asset.asset_id == logical


# ── B: 显式入库即分类 ────────────────────────────────────────────────────────
def test_classify_asset_tags_unknown_track(tmp_path):
    agent = _agent(tmp_path)
    asset = Asset(
        asset_id="x1", source_url="https://music.163.com/song?id=1",
        title="God's Plan", artist="Drake", duration_seconds=200,
        status=AssetStatus.ANALYZED,
    )
    agent.store.write_model("assets", "x1", asset)
    out = agent.classify_asset("x1")
    assert out is not None
    assert "欧美说唱" in out.genre  # Drake 命中艺人映射表
    assert out.features_source == "estimated"
    assert out.tempo_bpm is not None and out.energy_level is not None


# ── C1: × 移除库内条目 ───────────────────────────────────────────────────────
def test_record_dislike_removes_matching_library_asset(tmp_path):
    agent = _agent(tmp_path)
    asset = Asset(
        asset_id="d1", source_url="https://music.163.com/song?id=777",
        title="Disliked", artist="X", duration_seconds=200,
        status=AssetStatus.ANALYZED, source="local",
    )
    agent.store.write_model("assets", "d1", asset)
    assert any(a.asset_id == "d1" for a in agent.list_assets())

    agent.record_dislike(DislikeRequest(
        user_id="test-dislike-rm", title="Disliked", artist="X",
        source="netease", source_id="777",
    ))
    assert not any(a.asset_id == "d1" for a in agent.list_assets())  # 已移除


def test_record_dislike_noop_when_not_in_library(tmp_path):
    agent = _agent(tmp_path)
    agent.record_dislike(DislikeRequest(
        user_id="test-dislike-noop", title="Never Imported", artist="Y",
        source="netease", source_id="999",
    ))
    assert agent.list_assets() == []  # 没入库的歌，× 不产生库副作用


# ── C2: import 跳过 disliked ─────────────────────────────────────────────────
def test_import_skips_disliked_tracks(tmp_path, monkeypatch):
    agent = _agent(tmp_path)
    # 先把 A 标为不喜欢
    agent.record_dislike(DislikeRequest(
        user_id="test-import-skip", title="Track A", artist="AA",
        source="netease", source_id="100",
    ))
    # 打桩歌单抓取：A(被嫌弃) + B(正常)
    def fake_fetch(pid, cookie="", limit=200):
        return {"name": "pl", "total": 2, "tags": [],
                "tracks": [{"song_id": "100", "title": "Track A", "artist": "AA"},
                           {"song_id": "101", "title": "Track B", "artist": "Drake"}]}
    monkeypatch.setattr("app.netease_auth.fetch_playlist_tracks", fake_fetch)
    # LLM 分类打桩为空（走关键词/艺人映射兜底，离线确定）
    monkeypatch.setattr(agent.library_svc, "_batch_classify_tracks", lambda pairs: [{} for _ in pairs])

    result = agent.import_netease_playlist("123", user_id="test-import-skip")
    assert result["disliked_skipped"] == 1
    assert result["imported"] == 1
    assert any(a.title == "Track B" for a in agent.list_assets())
    assert not any(a.title == "Track A" for a in agent.list_assets())  # 被嫌弃的没进库


def test_import_rejects_junk_tracks(tmp_path, monkeypatch):
    """Phase 1.1：导入质量闸门——教程/合集/DJ串烧等非歌曲实体不进库（也不创建占位 asset），
    计入 result['rejected']，避免污染 compute_taste_profile 的长期画像。"""
    agent = _agent(tmp_path)

    def fake_fetch(pid, cookie="", limit=200):
        return {"name": "pl", "total": 2, "tags": [],
                "tracks": [{"song_id": "1", "title": "真歌", "artist": "Drake"},
                           {"song_id": "2", "title": "钢琴轻音乐教程合集", "artist": "教程君"}]}

    monkeypatch.setattr("app.netease_auth.fetch_playlist_tracks", fake_fetch)
    monkeypatch.setattr(agent.library_svc, "_batch_classify_tracks", lambda pairs: [{} for _ in pairs])

    result = agent.import_netease_playlist("123", user_id="test-import-gate")
    assert result["imported"] == 1
    assert result["rejected"] == 1
    titles = [a.title for a in agent.list_assets()]
    assert any("真歌" in t for t in titles)
    assert not any("教程" in t or "合集" in t for t in titles)  # 脏数据没进库


# ── D: 污染清理 ──────────────────────────────────────────────────────────────
def test_cleanup_deletes_external_tagless_and_reclassifies_local(tmp_path):
    agent = _agent(tmp_path)
    ext = Asset(asset_id="e1", source_url="https://music.163.com/song?id=2",
                title="Played", artist="Q", duration_seconds=200,
                status=AssetStatus.ANALYZED, source="external")  # 无标签 → 播放垃圾
    untyped_local = Asset(asset_id="u1", source_url="https://music.163.com/song?id=3",
                          title="Sicko Mode", artist="Travis Scott", duration_seconds=200,
                          status=AssetStatus.ANALYZED, source="local", genre=["未分类"])  # 漏分类
    tagged = Asset(asset_id="t1", source_url="https://music.163.com/song?id=4",
                   title="Tagged", artist="Z", duration_seconds=200,
                   status=AssetStatus.ANALYZED, source="local", genre=["流行"], mood=["欢快"])
    for a in (ext, untyped_local, tagged):
        agent.store.write_model("assets", a.asset_id, a)

    result = agent.cleanup_play_pollution()
    assert result == {"deleted": 1, "reclassified": 1}
    ids = {a.asset_id for a in agent.list_assets()}
    assert "e1" not in ids            # 外部无标签 → 删除
    assert "u1" in ids                # 本地未分类 → 保留并重分类
    u1 = agent.store.read_model("assets", "u1", Asset)
    assert u1.genre and u1.genre != ["未分类"]  # Travis Scott → 欧美说唱/Trap
    assert "t1" in ids                # 有真实标签 → 不动
