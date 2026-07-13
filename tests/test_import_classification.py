"""回归测试：导入网易云歌单时 genre/mood 必须永不为空（三层兜底）。

历史 bug：genre/mood 完全依赖单次 LLM JSON 分类，mock/离线模式或 LLM 偶发错位时
导入歌曲的风格分析为空，导致推荐/品味分析过滤掉这些歌。
"""

from __future__ import annotations

import tempfile

import pytest

from app.agent import AudioVisualAgent
from app.storage import JsonStore


@pytest.fixture
def agent():
    return AudioVisualAgent(JsonStore(tempfile.mkdtemp()))


@pytest.fixture
def fake_playlist(monkeypatch):
    def _fetch(pid, cookie="", limit=200):
        return {
            "name": "测试歌单",
            "total": 3,
            "tracks": [
                {"song_id": "1", "title": "热血摇滚", "artist": "Beyond", "duration": 240},
                {"song_id": "2", "title": "晴天", "artist": "周杰伦", "duration": 269},
                {"song_id": "3", "title": "chill放松下午茶", "artist": "Lofi", "duration": 180},
            ],
        }

    monkeypatch.setattr("app.netease_auth.fetch_playlist_tracks", _fetch)


def test_imported_tracks_always_have_genre_and_mood(agent, fake_playlist):
    """mock LLM 下分类 JSON 解析为空，但每首歌仍必须有 genre/mood。"""
    result = agent.import_netease_playlist("playlist?id=999", user_id="u1", limit=10)
    assert result["imported"] == 3
    for asset in result["tracks"]:
        assert asset.genre, f"{asset.title} genre 为空"
        assert asset.mood, f"{asset.title} mood 为空"


def test_imported_tracks_are_analyzed_status(agent, fake_playlist):
    """导入的歌必须标记 analyzed，否则推荐/品味会过滤掉。"""
    result = agent.import_netease_playlist("playlist?id=999", user_id="u1", limit=10)
    for asset in result["tracks"]:
        assert asset.status.value == "analyzed"


def test_keyword_rule_layer_hits(agent):
    """层2：歌名含明确关键词时应命中规则而非随机兜底。"""
    genre, mood = agent._ensure_track_tags("热血摇滚之歌", "Beyond", [], [])
    assert "摇滚" in genre


def test_fallback_is_deterministic(agent):
    """层3：无关键词时确定性兜底，同输入同输出。"""
    a = agent._ensure_track_tags("晴天", "周杰伦", [], [])
    b = agent._ensure_track_tags("晴天", "周杰伦", [], [])
    assert a == b
    assert a[0] and a[1]


def test_llm_result_takes_priority(agent):
    """层1：LLM 给了结果就用 LLM 的。"""
    genre, mood = agent._ensure_track_tags("x", "y", ["电子"], ["激昂"])
    assert genre == ["电子"]
    assert mood == ["激昂"]


# ── 2026-06 修复：歌单 tags 兜底 + 失败标「未分类」不猜 ──


def test_playlist_tags_map_to_genres(agent):
    """网易云歌单 tags 映射成本系统曲风词表。"""
    assert agent._playlist_tags_to_genres(["R&B/Soul", "欧美"]) == ["R&B"]
    assert agent._playlist_tags_to_genres(["摇滚", "Rock"]) == ["摇滚"]
    assert agent._playlist_tags_to_genres(["欧美", "夜晚"]) == []


def test_playlist_genre_fallback_when_llm_empty(agent):
    """LLM 没分出来时，用歌单 tags 映射的曲风兜底，而非瞎猜。"""
    genre, _ = agent._ensure_track_tags("Some English Title", "Unknown", [], [], playlist_genres=["R&B"])
    assert genre == ["R&B"]


def test_unclassified_when_nothing_known(agent):
    """无任何线索 → 如实标「未分类」，绝不用假「流行」或 hash 随机污染品味。"""
    genre, _ = agent._ensure_track_tags("zzqq", "wwxx", [], [], playlist_genres=[])
    assert genre == ["未分类"]


def test_invalid_llm_genre_filtered(agent, monkeypatch):
    """LLM 返回词表外的风格（如 K-Pop）必须被过滤。"""

    class FakeLLM:
        def generate(self, prompt, **kw):
            return '[{"genre":"K-Pop","mood":"欢快"},{"genre":"R&B","mood":"浪漫"}]'

    monkeypatch.setattr(agent, "llm", FakeLLM())
    out = agent._batch_classify_tracks([("a", "b"), ("c", "d")])
    assert out[0]["genre"] == []  # K-Pop 被过滤
    assert out[1]["genre"] == ["R&B"]


def test_playlist_tags_used_on_import(agent, monkeypatch):
    """整单 tags 为 R&B 时，LLM 分不出的歌应落到 R&B 而非「未分类」。"""

    def _fetch(pid, cookie="", limit=200):
        return {
            "name": "我的 R&B 歌单",
            "tags": ["R&B/Soul", "欧美"],
            "total": 1,
            "tracks": [{"song_id": "9", "title": "Obscure English Track", "artist": "Nobody", "duration": 200}],
        }

    monkeypatch.setattr("app.netease_auth.fetch_playlist_tracks", _fetch)
    result = agent.import_netease_playlist("playlist?id=1", user_id="u1", limit=10)
    asset = result["tracks"][0]
    # mock LLM 分不出 → 歌单 tags 兜底为 R&B（而非「未分类」或假「流行」）
    assert "R&B" in asset.genre
