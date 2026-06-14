"""锁住外部源选择 + 播放取流的鸭子类型兼容。

历史 P0 bug：
1. agent.source 写死 MockSource()，推荐永远拉硬编码假歌单；
2. get_audio_url 用 isinstance 严格判类型，Web 前端传的 SimpleNamespace
   两者都不匹配，VIP 登录后仍永远返回 None。
"""
from __future__ import annotations

from types import SimpleNamespace

from app.agent import _build_source
from app.sources.mock_source import MockSource
from app.sources.netease_source import NeteaseSource


class TestSourceFactory:
    def test_default_is_real_netease(self, monkeypatch):
        monkeypatch.setenv("EXTERNAL_SOURCE", "netease")
        # 重新读取 settings 的取值路径：直接构造工厂依赖的 settings 字段
        import app.config
        monkeypatch.setattr(app.config.settings, "external_source", "netease")
        assert isinstance(_build_source(), NeteaseSource)

    def test_mock_only_when_explicit(self, monkeypatch):
        import app.config
        monkeypatch.setattr(app.config.settings, "external_source", "mock")
        assert isinstance(_build_source(), MockSource)


class TestNeteaseSourceShape:
    def test_recommendations_return_real_source_tag(self, monkeypatch):
        """get_recommendations 的候选必须来自真实源（source=netease），非 mock。"""
        from app.sources import netease_source as ns

        monkeypatch.setattr(ns, "search_netease_many", lambda kw, limit=20: [
            {"song_id": "1", "title": f"{kw}-歌", "artist": "歌手", "album": None, "cover": None},
        ])
        src = NeteaseSource()
        recs = src.get_recommendations(["R&B"], ["浪漫"], limit=3)
        assert recs
        assert all(t.source == "netease" for t in recs)
        assert all(t.external_id for t in recs)

    def test_search_maps_metadata(self, monkeypatch):
        from app.sources import netease_source as ns

        monkeypatch.setattr(ns, "search_netease_many", lambda q, limit=20: [
            {"song_id": "42", "title": "Blinding Lights", "artist": "The Weeknd",
             "album": "After Hours", "cover": "http://c.jpg"},
        ])
        tracks = NeteaseSource().search("the weeknd", limit=5)
        assert tracks[0].title == "Blinding Lights"
        assert tracks[0].external_id == "42"
        assert tracks[0].source == "netease"


class TestGetAudioUrlDuckTyping:
    def test_simplenamespace_accepted(self, monkeypatch):
        """Web 前端传的 SimpleNamespace 必须能取流（不再因类型判断直接 None）。"""
        from app.agent import AudioVisualAgent

        agent = AudioVisualAgent.__new__(AudioVisualAgent)
        # 打桩搜索 + 取流，验证调用链通到鸭子类型属性
        monkeypatch.setattr(agent, "_search_netease", lambda q: "999" if "Blinding" in q else None)
        monkeypatch.setattr(agent, "_get_netease_audio_url",
                            lambda sid, cookie="": f"https://audio/{sid}")
        ns = SimpleNamespace(title="Blinding Lights", artist="The Weeknd",
                             source="netease", external_id="", source_url="", cover_url=None)
        url = agent.get_audio_url(ns, netease_cookie="MUSIC_U=x")
        assert url == "https://audio/999"

    def test_source_url_direct_extract(self, monkeypatch):
        from app.agent import AudioVisualAgent

        agent = AudioVisualAgent.__new__(AudioVisualAgent)
        monkeypatch.setattr(agent, "_get_netease_audio_url",
                            lambda sid, cookie="": f"https://audio/{sid}")
        ns = SimpleNamespace(title="x", artist="y", source="netease",
                             external_id="", source_url="https://music.163.com/song?id=555", cover_url=None)
        url = agent.get_audio_url(ns)
        assert url == "https://audio/555"

    def test_no_title_no_url_returns_none(self):
        from app.agent import AudioVisualAgent

        agent = AudioVisualAgent.__new__(AudioVisualAgent)
        ns = SimpleNamespace(title="", artist="", source="netease",
                             external_id="", source_url="", cover_url=None)
        assert agent.get_audio_url(ns) is None
