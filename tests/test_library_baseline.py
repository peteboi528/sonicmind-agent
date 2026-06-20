from __future__ import annotations

import sqlite3

from app.library import ResourceLibrary
from app.library_baseline import build_library_baseline, compare_library_baselines
from app.models import ResourceTrack


def test_library_baseline_is_read_only_and_ignores_ranking_counters(tmp_path):
    store = tmp_path / "store"
    assets = store / "assets"
    assets.mkdir(parents=True)
    (assets / "a1.json").write_text('{"asset_id":"a1","title":"One"}', encoding="utf-8")
    (assets / "a2.json").write_text('{"asset_id":"a2","title":"Two"}', encoding="utf-8")
    media = tmp_path / "media"
    media.mkdir()
    (media / "cover.jpg").write_bytes(b"cover-bytes")
    resource_path = tmp_path / "resource.sqlite"
    library = ResourceLibrary(resource_path)
    library.upsert_track(ResourceTrack(
        title="One", artist="Artist", source="local", source_id="a1", verified=True,
    ))

    before = build_library_baseline(
        store_root=store, resource_library=resource_path, media_root=media,
    )
    before_mtime = resource_path.stat().st_mtime_ns
    with sqlite3.connect(resource_path) as connection:
        connection.execute("UPDATE tracks SET exposure_count=99,ts_alpha=8,ts_beta=3")
    after_counters = build_library_baseline(
        store_root=store, resource_library=resource_path, media_root=media,
    )

    assert before.asset_count == 2
    assert before.resource_track_count == 1
    assert before.local_track_count == 1
    assert before.resource_source_counts == {"local": 1}
    assert before.media_file_count == 1
    assert compare_library_baselines(before, after_counters)["unchanged"] is True
    assert resource_path.stat().st_mtime_ns >= before_mtime


def test_library_baseline_detects_asset_or_media_changes(tmp_path):
    store = tmp_path / "store"
    assets = store / "assets"
    assets.mkdir(parents=True)
    asset = assets / "a1.json"
    asset.write_text('{"asset_id":"a1","title":"One"}', encoding="utf-8")
    media = tmp_path / "media"
    media.mkdir()
    resource = tmp_path / "missing.sqlite"
    before = build_library_baseline(store_root=store, resource_library=resource, media_root=media)

    asset.write_text('{"asset_id":"a1","title":"Changed"}', encoding="utf-8")
    (media / "audio.bin").write_bytes(b"audio")
    after = build_library_baseline(store_root=store, resource_library=resource, media_root=media)
    comparison = compare_library_baselines(before, after)

    assert comparison["unchanged"] is False
    assert "asset_digest" in comparison["changed"]
    assert "media_file_count" in comparison["changed"]
