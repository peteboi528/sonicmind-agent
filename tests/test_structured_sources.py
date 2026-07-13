"""结构化知识源合流测试：_apply_structured_sources + 无凭证降级。"""

from __future__ import annotations

from app.knowledge import _apply_structured_sources, _discogs_metadata, _source_from_url, _spotify_metadata
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


def test_apply_musicbrainz_relations_as_citations():
    entity = MusicEntity(type="album", name="OK Computer", artist="Radiohead")
    metadata: list = []
    citations: list = []
    mb = {
        "canonical_name": "OK Computer",
        "mbid": "rg1",
        "tags": ["alternative rock"],
        "summary": "艺人 Radiohead，发行 1997-06-16，Album",
        "relations": [
            {"type": "review", "url": "https://www.bbc.co.uk/music/reviews/wcp2"},
            {"type": "allmusic", "url": "https://www.allmusic.com/album/mw0000024289"},
            {"type": "youtube", "url": "https://www.youtube.com/watch?v=x"},
        ],
    }

    _apply_structured_sources(entity, [mb, None, None], metadata, citations)

    urls = {c.url for c in citations}
    assert "https://www.bbc.co.uk/music/reviews/wcp2" in urls
    assert "https://www.allmusic.com/album/mw0000024289" in urls
    assert "https://www.youtube.com/watch?v=x" not in urls
    bbc = next(c for c in citations if c.url.endswith("/wcp2"))
    assert bbc.source == "bbc"
    assert bbc.kind == "review"
    assert "Radiohead" in bbc.title
    assert any("MusicBrainz 关联链接" in item.get("summary", "") for item in metadata)


def test_source_from_archive_url_uses_original_domain():
    url = (
        "http://web.archive.org/web/20010303103405/www.pitchforkmedia.com/record-reviews/r/radiohead/ok-computer.shtml"
    )
    assert _source_from_url(url) == "pitchforkmedia"


def test_spotify_metadata_none_without_credentials(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "spotify_client_id", "", raising=False)
    monkeypatch.setattr(settings, "spotify_client_secret", "", raising=False)
    assert _spotify_metadata(MusicEntity(type="artist", name="x")) is None


def test_discogs_metadata_none_without_token(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "discogs_token", "", raising=False)
    assert _discogs_metadata(MusicEntity(type="album", name="x")) is None


def test_canonicalize_release_prefers_exact_title_over_higher_scored_fuzzy():
    """裸标题「Blonde」：MB 同时返回高分模糊的「Blonde on Blonde」与精确的 Frank Ocean《Blonde》，
    必须按精确同名选 Frank Ocean，而非被高分模糊带偏（旧 bug 落到 Blonde on Blonde）。"""
    from app.knowledge import _canonicalize_release_entity

    class _FakeMB:
        def search_release_group(self, name, artist="", limit=10):
            return [
                {"title": "Blonde on Blonde", "artist": "Bob Dylan", "score": 95, "mbid": "b1"},
                {"title": "Blonde", "artist": "Frank Ocean", "score": 88, "mbid": "f1"},
            ]

    entity = MusicEntity(type="album", name="Blonde")
    _canonicalize_release_entity(_FakeMB(), entity)
    assert entity.name == "Blonde"
    assert entity.artist == "Frank Ocean"
    assert entity.external_ids.get("musicbrainz") == "f1"
    assert entity.ambiguity == "resolved"


def test_enrich_review_content_constructs_lastfm_and_fills(monkeypatch):
    """MB 死时：_enrich_review_content 用实体 (artist,name) 构造 last.fm URL 兜底，填回真实正文。"""
    import time

    from app import knowledge
    from app.config import settings
    from app.models import MusicCitation

    monkeypatch.setattr(settings, "tavily_api_key", "k", raising=False)
    monkeypatch.setattr(settings, "discogs_token", "", raising=False)
    monkeypatch.setattr(settings, "knowledge_review_extract_max_sources", 4, raising=False)

    def fake_fetch(url, api_key="", timeout=None, max_chars=2000):
        return "Blonde Metascore 87，广受好评的 alternative R&B 专辑。" if "last.fm" in url else ""

    monkeypatch.setattr(knowledge.web_search_source, "fetch_url_content", fake_fetch)

    entity = MusicEntity(type="album", name="Blonde", artist="Frank Ocean")
    cites = [
        MusicCitation(
            source="musicbrainz", url="https://musicbrainz.org/release-group/x", kind="encyclopedia", excerpt=""
        )
    ]
    knowledge._enrich_review_content(cites, entity, time.monotonic() + 30)

    lastfm = [c for c in cites if "last.fm" in c.url]
    assert lastfm, "应构造出 last.fm 兜底 citation"
    assert lastfm[0].excerpt.startswith("Blonde Metascore 87")
    assert "Frank+Ocean" in lastfm[0].url  # 用实体拼出的稳定入口


def test_enrich_review_content_skips_when_budget_too_tight():
    """预算不足时整段放弃抓取，原样返回，保住合成预算。"""
    import time

    from app import knowledge
    from app.models import MusicCitation, MusicEntity

    entity = MusicEntity(type="album", name="Blonde", artist="Frank Ocean")
    cites = [
        MusicCitation(
            source="lastfm", url="https://www.last.fm/music/Frank+Ocean/Blonde", kind="encyclopedia", excerpt=""
        )
    ]
    before = len(cites)
    knowledge._enrich_review_content(cites, entity, time.monotonic() + 0.5)  # 剩余不足
    assert len(cites) == before
    assert cites[0].excerpt == ""
