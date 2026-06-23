"""Phase 0 止血：实体消歧 + 证据一致性校验的确定性测试。

覆盖同名专辑/艺人资料混拼的根因——
1) canonicalize_entities 用 MusicBrainz 候选判定 resolved/ambiguous/unresolved；
2) citation_entity_score 按来源类型归属打分，剔除同名异作品乐评；
3) validate_evidence_consistency 剔除别家曲目、报告证据冲突；
4) build_dossier 在歧义/证据冲突时抑制完整合成，返回消歧提示而非硬编。
"""
from __future__ import annotations

import pytest


# ── canonicalize_entities 消歧状态 ────────────────────────────────────────────

def test_canonicalize_flags_same_title_different_artist_as_ambiguous(monkeypatch):
    """裸标题存在多个精确同名、艺人各异 → ambiguous（不再硬编一个完整答案）。"""
    from app.knowledge import canonicalize_entities
    from app.models import MusicEntity
    from app.sources import musicbrainz_client

    class FakeMB:
        def search_release_group(self, title, artist="", limit=3):
            return [
                {"mbid": "m1", "title": "Blonde", "artist": "Frank Ocean", "score": 90, "date": "2016", "type": "Album"},
                {"mbid": "m2", "title": "Blonde", "artist": "West Norwood Cassette Library", "score": 40, "date": "2014", "type": "Album"},
            ]

        def search_artist(self, name, limit=3):
            return []

    monkeypatch.setattr(musicbrainz_client, "MusicBrainzClient", FakeMB)
    monkeypatch.setattr("app.config.settings.enable_musicbrainz", True)

    entities = canonicalize_entities([MusicEntity(type="album", name="Blonde", source="query")])
    assert entities[0].ambiguity == "ambiguous"
    assert len(entities[0].candidates) >= 2
    assert entities[0].confidence <= 0.5


def test_canonicalize_resolved_when_artist_matches_candidate(monkeypatch):
    """用户给了艺人，且候选里存在精确标题+该艺人 → resolved，正确锁定实体。"""
    from app.knowledge import canonicalize_entities
    from app.models import MusicEntity
    from app.sources import musicbrainz_client

    class FakeMB:
        def search_release_group(self, title, artist="", limit=3):
            return [
                {"mbid": "m1", "title": "Blonde", "artist": "Frank Ocean", "score": 95, "date": "2016", "type": "Album"},
                {"mbid": "m2", "title": "Blonde on Blonde", "artist": "Bob Dylan", "score": 50, "date": "1966", "type": "Album"},
            ]

        def search_artist(self, name, limit=3):
            return []

    monkeypatch.setattr(musicbrainz_client, "MusicBrainzClient", FakeMB)
    monkeypatch.setattr("app.config.settings.enable_musicbrainz", True)

    entities = canonicalize_entities([MusicEntity(type="album", name="Blonde", artist="Frank Ocean", source="query")])
    assert entities[0].ambiguity == "resolved"
    assert entities[0].name == "Blonde"
    assert entities[0].artist == "Frank Ocean"


def test_canonicalize_resolved_single_exact_title_backfills_artist(monkeypatch):
    """唯一精确同名作品（无消歧艺人）→ resolved，并回填权威艺人。"""
    from app.knowledge import canonicalize_entities
    from app.models import MusicEntity
    from app.sources import musicbrainz_client

    class FakeMB:
        def search_release_group(self, title, artist="", limit=3):
            return [{"mbid": "k1", "title": "Kid A", "artist": "Radiohead", "score": 95, "date": "2000", "type": "Album"}]

        def search_artist(self, name, limit=3):
            return []

    monkeypatch.setattr(musicbrainz_client, "MusicBrainzClient", FakeMB)
    monkeypatch.setattr("app.config.settings.enable_musicbrainz", True)

    entities = canonicalize_entities([MusicEntity(type="album", name="Kid A", source="query")])
    assert entities[0].ambiguity == "resolved"
    assert entities[0].artist == "Radiohead"


def test_canonicalize_unchanged_when_musicbrainz_disabled(monkeypatch):
    """MB 关闭时原样返回，ambiguity 保持默认 unresolved（维持旧行为，离线契约）。"""
    from app.knowledge import canonicalize_entities
    from app.models import MusicEntity

    monkeypatch.setattr("app.config.settings.enable_musicbrainz", False)
    entity = MusicEntity(type="album", name="Blonde", source="query")
    entities = canonicalize_entities([entity])
    assert entities[0].ambiguity == "unresolved"
    assert entities[0].name == "Blonde"


# ── citation_entity_score 类型感知打分 ────────────────────────────────────────

def test_citation_entity_score_is_kind_aware():
    from app.knowledge import citation_entity_score
    from app.models import MusicEntity, MusicCitation

    entity = MusicEntity(type="album", name="Blonde", artist="Frank Ocean")
    # 结构化平台源按构造归属，默认高分
    assert citation_entity_score(MusicCitation(source="netease", kind="platform"), entity) >= 0.8
    # 散文乐评：艺人 + 标题命中 → 1.0
    assert citation_entity_score(
        MusicCitation(source="pitchfork", kind="review", title="Blonde", excerpt="Frank Ocean returns"), entity
    ) == 1.0
    # 只命中标题不提艺人 → 弱分（同名异作品高风险）
    assert citation_entity_score(
        MusicCitation(source="web", kind="review", title="Blonde", excerpt="by Bob Dylan"), entity
    ) <= 0.25
    # 完全不沾边 → 0
    assert citation_entity_score(
        MusicCitation(source="web", kind="review", title="OK Computer", excerpt="Radiohead"), entity
    ) == 0.0


# ── validate_evidence_consistency ─────────────────────────────────────────────

def test_structured_metadata_citation_kept_without_artist_mention():
    """平台/元数据类来源 excerpt 不重述艺人也保留——它们是按实体检索来的。"""
    from app.knowledge import validate_evidence_consistency
    from app.models import MusicEntity, MusicCitation

    entity = MusicEntity(type="album", name="Blonde", artist="Frank Ocean")
    netease = MusicCitation(source="netease", title="Blonde", kind="platform", excerpt="网易云专辑元数据", confidence=0.85)
    report = validate_evidence_consistency(entity, [], [netease], [])
    assert len(report.kept_citations) == 1
    assert report.ok is True


def test_album_key_tracks_filtered_by_artist():
    """同名专辑混入别家曲目时，已知艺人下剔除不匹配曲目。"""
    from app.knowledge import validate_evidence_consistency
    from app.models import MusicEntity, TrackRef

    entity = MusicEntity(type="album", name="Blonde", artist="Frank Ocean")
    tracks = [
        TrackRef(title="Nikes", artist="Frank Ocean", source="netease"),
        TrackRef(title="Rainy Day Women #12 & 17", artist="Bob Dylan", source="netease"),
    ]
    report = validate_evidence_consistency(entity, [], [], tracks)
    titles = [t.title for t in report.kept_tracks]
    assert "Nikes" in titles
    assert "Rainy Day Women #12 & 17" not in titles
    assert any("曲目" in p for p in report.problems)


# ── build_dossier 集成：歧义/证据冲突抑制完整合成 ────────────────────────────

def test_off_target_review_dropped_for_known_artist():
    """已知艺人时，同名异作品乐评被剔除，不进入最终 dossier 引用。"""
    from app.knowledge import build_dossier
    from app.models import MusicCitation, MusicEntity

    entity = MusicEntity(type="album", name="Blonde", artist="Frank Ocean")
    on_target = MusicCitation(source="pitchfork", title="Blonde - Frank Ocean", url="https://pitchfork.com/a",
                              kind="review", excerpt="Frank Ocean's Blonde is a 2016 album", confidence=0.9)
    off_target = MusicCitation(source="web", title="Blonde on Blonde review", url="https://x.com/b",
                               kind="review", excerpt="Bob Dylan's Blonde on Blonde is a 1966 double album", confidence=0.5)
    dossier = build_dossier(None, "Blonde", "album_deep_dive", [entity], [], [], [on_target, off_target], [], [])
    titles = " ".join(c.title for c in dossier.citations)
    assert "Frank Ocean" in titles
    assert "Bob Dylan" not in titles
    assert "Blonde on Blonde" not in titles


def test_all_off_target_reviews_block_confident_synthesis():
    """所有乐评都偏题时，抑制完整总结，回落安全兜底，不混拼错误实体。"""
    from app.knowledge import build_dossier
    from app.models import MusicCitation, MusicEntity

    entity = MusicEntity(type="album", name="Blonde", artist="Frank Ocean")
    off1 = MusicCitation(source="web", title="Blonde on Blonde", kind="review",
                         excerpt="Bob Dylan's Blonde on Blonde landmark", confidence=0.5)
    off2 = MusicCitation(source="web", title="Another Blonde", kind="review",
                         excerpt="Some other artist's Blonde record", confidence=0.5)
    dossier = build_dossier(None, "Blonde", "album_deep_dive", [entity], [], [], [off1, off2], [], [])
    assert dossier.partial is True
    assert "Bob Dylan" not in dossier.summary
    assert "Some other artist" not in dossier.summary


def test_ambiguous_entity_returns_disambiguation_prompt():
    """歧义实体返回消歧提示，绝不凭猜测合成完整答案。"""
    from app.knowledge import build_dossier, dossier_answer
    from app.models import MusicCitation, MusicEntity

    entity = MusicEntity(
        type="album", name="Blonde", artist="", ambiguity="ambiguous",
        candidates=[{"title": "Blonde", "artist": "Frank Ocean"},
                    {"title": "Blonde", "artist": "West Norwood Cassette Library"}],
    )
    reviews = [MusicCitation(source="web", title="Blonde review", kind="review",
                             excerpt="Another Blonde by someone else entirely", confidence=0.5)]
    dossier = build_dossier(None, "讲讲 Blonde", "album_deep_dive", [entity], [], [], reviews, [], [])
    text = dossier_answer(dossier)
    assert dossier.partial is True
    assert ("同名" in text) or ("歧义" in text) or ("多个" in text)
    assert "Another Blonde by someone else" not in dossier.summary


def test_resolved_entity_still_synthesizes_normally():
    """回归保护：resolved 实体 + 命中艺人乐评 → 正常走合成/兜底，不被误判歧义。"""
    from app.knowledge import build_dossier
    from app.models import MusicCitation, MusicEntity

    entity = MusicEntity(type="album", name="Blonde", artist="Frank Ocean", ambiguity="resolved")
    review = MusicCitation(source="pitchfork", title="Blonde - Frank Ocean", kind="review",
                           excerpt="Frank Ocean's Blonde is acclaimed", confidence=0.9)
    dossier = build_dossier(None, "Blonde", "album_deep_dive", [entity], [], [], [review], [], [])
    # agent=None → 无 LLM → 机械兜底，但仍是「正常」路径（非歧义/非证据冲突），
    # 有命中艺人的乐评 → dossier 不应被判 partial。
    assert dossier.entity.ambiguity == "resolved"
    assert dossier.partial is False
    assert "同名" not in dossier.summary
    assert "归属不一致" not in dossier.summary
    # 命中乐评进入引用列表
    assert any("Frank Ocean" in c.title for c in dossier.citations)


# ── 黄金评测集可加载 ───────────────────────────────────────────────────────────

def test_golden_cases_json_loads():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "evals" / "music_knowledge_cases.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert len(data["cases"]) >= 10
    for case in data["cases"]:
        assert case.get("query")
        assert case.get("intent")
        assert case.get("expected_entity")
