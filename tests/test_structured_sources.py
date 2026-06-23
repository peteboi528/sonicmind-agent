"""结构化知识源合流测试：_apply_structured_sources + 无凭证降级。"""
from __future__ import annotations

from app.knowledge import _apply_structured_sources, _discogs_metadata, _spotify_metadata
from app.models import MusicEntity


def test_apply_merges_three_sources_into_entity_and_citations():
    entity = MusicEntity(type="artist", name="frank ocean")
    metadata: list = []
    citations: list = []
    mb = {"canonical_name": "Frank Ocean", "mbid": "mb1", "tags": ["r&b"], "summary": "Person，来自 US"}
    sp = {"external_id": "sp1", "genres": ["neo soul"], "summary": "标签：neo soul", "image": "img"}
    dc = {"external_id": "dc1", "styles": ["Alternative R&B"], "summary": "细类：Alternative R&B", "type": "master"}

    _apply_structured_sources(entity, [mb, sp, dc], metadata, citations)

    assert entity.name == "Frank Ocean"  # MB 权威名纠正
    assert entity.external_ids["musicbrainz"] == "mb1"
    assert entity.external_ids["spotify"] == "sp1"
    assert entity.external_ids["discogs"] == "dc1"
    assert entity.image == "img"  # Spotify 封面
    assert {c.source for c in citations} == {"musicbrainz", "spotify", "discogs"}


def test_apply_skips_none_sources_gracefully():
    """任一外部源失败（None）不影响其余，全失败时 entity 不变。"""
    entity = MusicEntity(type="album", name="Blonde")
    metadata: list = []
    citations: list = []
    _apply_structured_sources(entity, [None, None, None], metadata, citations)
    assert metadata == []
    assert citations == []
    assert entity.name == "Blonde"
    assert entity.external_ids == {}


def test_apply_partial_failure_keeps_survivors():
    entity = MusicEntity(type="artist", name="sza")
    metadata: list = []
    citations: list = []
    mb = {"canonical_name": "SZA", "mbid": "mb2", "tags": ["pop"], "summary": "Person"}
    # Spotify/Discogs 挂掉
    _apply_structured_sources(entity, [mb, None, None], metadata, citations)
    assert entity.name == "SZA"
    assert len(citations) == 1
    assert citations[0].source == "musicbrainz"


def test_spotify_metadata_none_without_credentials(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "spotify_client_id", "", raising=False)
    monkeypatch.setattr(settings, "spotify_client_secret", "", raising=False)
    assert _spotify_metadata(MusicEntity(type="artist", name="x")) is None


def test_discogs_metadata_none_without_token(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "discogs_token", "", raising=False)
    assert _discogs_metadata(MusicEntity(type="album", name="x")) is None
