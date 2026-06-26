"""Track hygiene 单测：is_valid_music_track 把教程/合集/歌单/节目挡在结果外。

对应线上脏数据：编曲教程、独立流行真的好难做、弹跳全集 等被当成歌曲塞进歌单。
"""
from __future__ import annotations

from app.models import ExternalTrack
from app.tools.handlers import is_valid_music_track


def _t(title: str, artist: str = "Demo", source: str = "netease", kind: str = "track") -> ExternalTrack:
    return ExternalTrack(
        external_id="x", title=title, artist=artist, source=source, candidate_kind=kind,
    )


def test_dirty_titles_rejected():
    """线上真实脏样本：教程/解说/合集/歌单/混剪 一律拦截。"""
    dirty = [
        "编曲技巧:怎么做一首流行R&B风格的音乐？",
        "独立流行音乐真的好难做",
        "独立流行摇滚弹跳全集",
        "跑步英文流行歌单大合集",
        "流行音乐现场混剪",
        "纯音乐合集 睡眠放松",
        "翻唱合集 2024",
    ]
    for title in dirty:
        assert is_valid_music_track(_t(title)) is False, f"应拦截: {title}"


def test_clean_tracks_accepted():
    """真实歌曲不受影响。"""
    clean = [("Ditto", "NewJeans"), ("ETA", "NewJeans"), ("Firework", "Katy Perry"),
             ("Blinding Lights", "The Weeknd"), ("Nikes", "Frank Ocean")]
    for title, artist in clean:
        assert is_valid_music_track(_t(title, artist)) is True, f"应保留: {title}"


def test_non_track_candidate_kind_rejected():
    """candidate_kind 七分类里非单曲实体直接拦截（搜索阶段已标注）。"""
    for kind in ["playlist", "compilation", "long_mix", "lyrics_video"]:
        assert is_valid_music_track(_t("Some Song", "Artist", kind=kind)) is False


def test_missing_title_or_artist_rejected():
    """必要条件：title 与 artist 都非空。"""
    assert is_valid_music_track(_t("", "Artist")) is False
    assert is_valid_music_track(_t("Song", "")) is False
    assert is_valid_music_track(None) is False


def test_bilibili_tutorial_strictness():
    """bilibili 默认高风险：解说/教学/问答类标题额外拦截，正常歌曲仍保留。"""
    assert is_valid_music_track(_t("怎么做一首流行歌？", "UP主", source="bilibili")) is False
    assert is_valid_music_track(_t("教你编曲", "UP主", source="bilibili")) is False
    assert is_valid_music_track(_t("夜曲", "周杰伦", source="bilibili")) is True
