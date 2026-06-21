"""候选池卫生：拦截假候选入库、上限裁剪、一次性清理污染。"""
from __future__ import annotations

from app.library import ResourceLibrary
from app.models import ExternalTrack


def _ext(title: str, source: str = "netease", sid: str = "") -> ExternalTrack:
    return ExternalTrack(
        external_id=sid or title,
        title=title,
        artist="A",
        source=source,
        playback_url="https://x",
    )


def test_upsert_external_blocks_fallback_and_mock(tmp_path):
    lib = ResourceLibrary(tmp_path / "lib.sqlite")
    lib.upsert_external(_ext("Real", source="netease", sid="1"))
    lib.upsert_external(_ext("Fake", source="netease-fallback", sid="2"))
    lib.upsert_external(_ext("Mockie", source="mock", sid="3"))
    lib.upsert_external(_ext("Llmy", source="llm", sid="4"))

    titles = {t.title for t in lib.list_tracks(100)}
    assert "Real" in titles
    assert "Fake" not in titles  # fallback 被拦
    assert "Mockie" not in titles
    assert "Llmy" not in titles


def test_purge_fallback_sources_removes_legacy_pollution(tmp_path):
    lib = ResourceLibrary(tmp_path / "lib.sqlite")
    # 直接走 upsert_track 模拟历史遗留的脏数据（绕过新拦截）。
    from app.models import ResourceTrack

    lib.upsert_track(ResourceTrack(title="Good", artist="A", source="netease", source_id="1", verified=True))
    lib.upsert_track(ResourceTrack(title="Bad", artist="A", source="netease-fallback", source_id="2"))
    lib.upsert_track(ResourceTrack(title="Mocky", artist="A", source="mock-fallback", source_id="3"))

    removed = lib.purge_fallback_sources()
    assert removed == 2
    assert {t.title for t in lib.list_tracks(100)} == {"Good"}


def test_purge_orphan_local_drops_dangling_rows(tmp_path):
    lib = ResourceLibrary(tmp_path / "lib.sqlite")
    from app.models import ResourceTrack

    lib.upsert_track(ResourceTrack(title="Live", artist="A", source="local", source_id="live-1", verified=True))
    lib.upsert_track(ResourceTrack(title="Dead", artist="A", source="local", source_id="gone-9", verified=True))

    removed = lib.purge_orphan_local(live_asset_ids={"live-1"})
    assert removed == 1
    assert {t.title for t in lib.list_tracks(100)} == {"Live"}


def test_prune_caps_pool_and_protects_local_and_exposed(tmp_path):
    lib = ResourceLibrary(tmp_path / "lib.sqlite", max_tracks=10)
    from app.models import ResourceTrack

    # 1 个 local（受保护）+ 1 个被曝光过的外部（受保护）+ 20 个普通外部候选。
    lib.upsert_track(ResourceTrack(title="LocalKeep", artist="A", source="local", source_id="L1", verified=True))
    lib.upsert_track(ResourceTrack(title="Exposed", artist="A", source="netease", source_id="E1", verified=True, exposure_count=3))
    for i in range(20):
        lib.upsert_track(ResourceTrack(title=f"Cand{i}", artist="A", source="netease", source_id=f"c{i}", verified=True))

    lib.prune()
    titles = {t.title for t in lib.list_tracks(1000)}
    assert "LocalKeep" in titles      # local 永不淘汰
    assert "Exposed" in titles        # 曝光过的不淘汰
    assert len(titles) <= 12          # 受保护 2 个 + 上限附近


def test_verified_only_filter_in_sql(tmp_path):
    lib = ResourceLibrary(tmp_path / "lib.sqlite")
    from app.models import ResourceTrack

    lib.upsert_track(ResourceTrack(title="V", artist="A", source="netease", source_id="1", verified=True))
    lib.upsert_track(ResourceTrack(title="U", artist="A", source="netease", source_id="2", verified=False))

    assert {t.title for t in lib.list_tracks(100, verified_only=True)} == {"V"}
    assert {t.title for t in lib.list_tracks(100)} == {"V", "U"}


def test_embedding_persisted_and_dirty_incremental(tmp_path, monkeypatch):
    """新写入行标 dirty；warm_embeddings 算好并落库后变 clean；改 genre 重新 dirty。"""
    lib = ResourceLibrary(tmp_path / "lib.sqlite")
    import app.retrieval.embeddings as emb
    from app.models import ResourceTrack

    lib.upsert_track(ResourceTrack(title="Rainy", artist="A", source="netease", source_id="1", verified=True))
    # fake encode: 用文本长度造确定性向量,避免依赖真模型
    monkeypatch.setattr(emb, "embeddings_available", lambda: True)
    monkeypatch.setattr(emb, "encode", lambda texts: [[float(len(t) % 7) / 7.0] * 4 for t in texts] or None)

    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "lib.sqlite"))
    row = conn.execute("SELECT embedding, embed_dirty FROM tracks WHERE title='Rainy'").fetchone()
    assert row[0] == "" and row[1] == 1  # 新行 dirty、无向量

    warmed = lib.warm_embeddings()
    assert warmed >= 1
    row = conn.execute("SELECT embedding, embed_dirty FROM tracks WHERE title='Rainy'").fetchone()
    assert row[0] != "" and row[1] == 0  # 已算并落库、变 clean

    # 改 genre → 应重新标 dirty（embedding 文本变了）。
    lib.upsert_track(ResourceTrack(title="Rainy", artist="A", source="netease", source_id="1", genre=["R&B"], verified=True))
    row = conn.execute("SELECT embed_dirty FROM tracks WHERE title='Rainy'").fetchone()
    assert row[0] == 1
    conn.close()


def test_semantic_search_bails_when_too_many_dirty(tmp_path, monkeypatch):
    """冷启动保护：dirty 行 >32 时不同步算，让位词法（返回空）。"""
    lib = ResourceLibrary(tmp_path / "lib.sqlite")
    import app.retrieval.embeddings as emb
    from app.models import ResourceTrack

    for i in range(40):
        lib.upsert_track(ResourceTrack(title=f"T{i}", artist="A", source="netease", source_id=str(i), verified=True))
    monkeypatch.setattr(emb, "embeddings_available", lambda: True)
    monkeypatch.setattr(emb, "encode", lambda texts: None)  # 不该被调用
    assert lib.semantic_search("query", limit=5) == []  # 40 dirty > 32 → bail
