"""resource library 写入卫生单测：脏数据不得 upsert 进资源库（防跨轮污染扩散）。

对应线上问题：教程/合集被当成 song 写入 library 后，后续多轮继续复用、污染扩散。
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


def test_only_clean_tracks_upserted_to_library():
    """2 首正常 + 2 条教程 + 1 条歌复合集 → 只有 2 首正常歌曲被 upsert。"""
    clean_a = _t("Ditto", "NewJeans")
    clean_b = _t("Firework", "Katy Perry")
    tracks = [
        clean_a, clean_b,
        _t("编曲技巧:怎么做一首流行R&B风格的音乐？", "UP主", "bilibili"),
        _t("独立流行音乐真的好难做", "UP主", "bilibili"),
        _t("纯音乐合集", "UP主", "bilibili"),
    ]
    agent = _Agent(tracks)
    ctx = ToolContext(thread_id="t", user_id="u", query="流行歌", agent=agent, plan={})
    _web_music_search({"query": "流行歌", "top_k": 5}, ctx)

    upserted_titles = [t.title for t in agent.library.upserted]
    assert upserted_titles == ["Ditto", "Firework"]
    assert not any("合集" in t or "技巧" in t or "难做" in t for t in upserted_titles)
