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
