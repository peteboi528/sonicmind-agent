from __future__ import annotations

import pytest

from app.models import Asset
from app.services.playlist import PlaylistService
from app.storage import JsonStore


def _asset(asset_id: str, title: str) -> Asset:
    return Asset(asset_id=asset_id, title=title, artist="A", source_url=f"http://x/{asset_id}", duration_seconds=180)


@pytest.fixture
def service(tmp_path):
    library = [_asset("a1", "曲一"), _asset("a2", "曲二"), _asset("a3", "曲三")]
    return PlaylistService(
        store=JsonStore(tmp_path / "store"),
        memory=None,
        llm=None,
        list_assets=lambda: library,
        search_web_music=lambda *a, **k: [],
        source=None,
        summarize_taste=lambda *a, **k: "",
        query_has_entity=lambda q: False,
    )


def test_create_from_assets_keeps_selected_order(service):
    pl = service.create_playlist_from_assets("u", "我的精选", ["a3", "a1"])
    assert pl.name == "我的精选"
    assert pl.generated_by == "manual"
    assert [t.asset_id for t in pl.tracks] == ["a3", "a1"]  # 保留传入顺序
    # 已落库，可读回
    assert service.list_playlists("u")[0].playlist_id == pl.playlist_id


def test_create_from_assets_dedupes_and_drops_unknown(service):
    pl = service.create_playlist_from_assets("u", "去重", ["a1", "a1", "nope", "a2"])
    assert [t.asset_id for t in pl.tracks] == ["a1", "a2"]


def test_create_from_assets_blank_name_falls_back(service):
    pl = service.create_playlist_from_assets("u", "   ", ["a1"])
    assert pl.name == "我的歌单"


def test_create_from_assets_rejects_empty_selection(service):
    with pytest.raises(ValueError):
        service.create_playlist_from_assets("u", "空", ["nope1", "nope2"])
