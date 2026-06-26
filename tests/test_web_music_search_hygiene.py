"""web_music_search hygiene 单测：外部搜索结果只返回真正的歌曲。

对应线上问题：bilibili/netease 把教程/合集当 song 返回，污染候选池。
"""
from __future__ import annotations

from app.models import ExternalTrack
from app.tools.contracts import ToolContext
from app.tools.handlers import _web_music_search


def _t(title, artist="Demo", source="netease"):
    return ExternalTrack(external_id=title, title=title, artist=artist, source=source)


class _Lib:
    def __init__(self):
        self.upserted = []

    def upsert_external(self, track):
        self.upserted.append(track)


class _Agent:
    def __init__(self, tracks):
        self._tracks = tracks
        self.library = _Lib()

    def search_web_music(self, query, top_k=5, relevance_query="", offset=0, variants=None):
        return list(self._tracks)


def test_web_music_search_filters_non_song_results():
    """2 首正常 + 教程/合集/歌单 → 只返回 2 首。"""
    tracks = [
        _t("Ditto", "NewJeans"),
        _t("Firework", "Katy Perry"),
        _t("编曲技巧:怎么做一首流行R&B风格的音乐？", "UP主", "bilibili"),
        _t("独立流行摇滚弹跳全集", "UP主", "bilibili"),
        _t("跑步歌单大合集", "UP主", "bilibili"),
    ]
    agent = _Agent(tracks)
    ctx = ToolContext(thread_id="t", user_id="u", query="流行歌", agent=agent, plan={})
    result = _web_music_search({"query": "流行歌", "top_k": 5}, ctx)

    titles = [c["title"] for c in result.cards]
    assert titles == ["Ditto", "Firework"]
    assert not any("合集" in t or "全集" in t or "技巧" in t for t in titles)
    assert len(result.cards) == 2
    # hygiene report：原始 5、清洗后 2、剔除非歌曲 3
    h = result.data["hygiene"]
    assert h["raw_count"] == 5 and h["cleaned_count"] == 2 and h["removed_invalid_tracks"] == 3
