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


class TestNeteaseAlbumTracks:
    def test_search_album_prefers_exact_artist_match(self, monkeypatch):
        from app.sources import netease as ns

        monkeypatch.setattr(ns, "_fetch_netease_albums", lambda query, limit=8: [
            {"id": 1, "name": "Other Album", "artist": {"name": "Other"}, "size": 2, "picUrl": "x"},
            {"id": 18893, "name": "依然范特西", "artist": {"name": "周杰伦"}, "size": 10, "picUrl": "cover"},
        ])

        album = ns.search_netease_album("周杰伦", "依然范特西")

        assert album is not None
        assert album["id"] == "18893"
        assert album["track_count"] == 10

    def test_fetch_album_tracks_preserves_api_order(self, monkeypatch):
        from app.sources import netease as ns

        payload = {
            "code": 200,
            "album": {"name": "Album", "picUrl": "cover", "artist": {"name": "Artist"}, "size": 3},
            "songs": [
                {"id": 1, "name": "Intro", "ar": [{"name": "Artist"}], "al": {"name": "Album", "picUrl": "cover"}},
                {"id": 2, "name": "Second", "ar": [{"name": "Artist"}], "al": {"name": "Album", "picUrl": "cover"}},
                {"id": 3, "name": "Finale", "ar": [{"name": "Artist"}], "al": {"name": "Album", "picUrl": "cover"}},
            ],
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                import json
                return json.dumps(payload).encode()

        monkeypatch.setattr(ns.urllib.request, "urlopen", lambda *a, **k: FakeResponse())

        detail = ns.fetch_netease_album_tracks("18893", limit=100)

        assert [t["title"] for t in detail["tracks"]] == ["Intro", "Second", "Finale"]
        assert detail["track_count"] == 3

    def test_search_artist_albums_returns_real_ids_and_ranks_artist(self, monkeypatch):
        """歌手页代表专辑要拿到真实 album_id：本歌手专辑排前、同名去重、群星合集靠后。"""
        from app.sources import netease as ns

        monkeypatch.setattr(ns, "_fetch_netease_albums", lambda query, limit=8: [
            {"id": 7, "name": "群星合集", "artist": {"name": "群星"}, "size": 20, "picUrl": "misc"},
            {"id": 18893, "name": "依然范特西", "artist": {"name": "周杰伦"}, "size": 10, "picUrl": "a"},
            {"id": 18894, "name": "叶惠美", "artist": {"name": "周杰伦"}, "size": 11, "picUrl": "b"},
            {"id": 18893, "name": "依然范特西", "artist": {"name": "周杰伦"}, "size": 10, "picUrl": "dup"},
        ])

        albums = ns.search_netease_artist_albums("周杰伦", limit=6)

        names = [a["name"] for a in albums]
        assert "依然范特西" in names
        assert "叶惠美" in names
        # 本歌手专辑(rank 高)排在前，群星合集(rank 0)排在最后
        assert names[-1] == "群星合集"
        # 同名专辑去重：依然范特西 只出现一次
        assert names.count("依然范特西") == 1
        # 每条都带真实 id 与 track_count（点击直达专辑详情，无需二次按名搜索）
        for a in albums:
            assert a["id"]
            assert "track_count" in a

    def test_album_detail_cached_avoids_refetch(self, monkeypatch):
        """同一 album_id 第二次取应命中缓存，不再请求网易云。"""
        from app.sources import netease as ns

        calls = {"n": 0}
        payload = {
            "code": 200,
            "album": {"name": "Album", "picUrl": "cover", "artist": {"name": "Artist"}, "size": 3},
            "songs": [
                {"id": 1, "name": "Intro", "ar": [{"name": "Artist"}], "al": {"name": "Album", "picUrl": "cover"}},
                {"id": 2, "name": "Second", "ar": [{"name": "Artist"}], "al": {"name": "Album", "picUrl": "cover"}},
                {"id": 3, "name": "Finale", "ar": [{"name": "Artist"}], "al": {"name": "Album", "picUrl": "cover"}},
            ],
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                import json
                return json.dumps(payload).encode()

        monkeypatch.setattr(ns.urllib.request, "urlopen", lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), FakeResponse())[1])

        first = ns.fetch_netease_album_tracks("18893", limit=100)
        second = ns.fetch_netease_album_tracks("18893", limit=100)

        assert calls["n"] == 1  # 第二次命中缓存，未再请求网络
        assert [t["title"] for t in first["tracks"]] == ["Intro", "Second", "Finale"]
        assert [t["title"] for t in second["tracks"]] == ["Intro", "Second", "Finale"]

    def test_album_detail_cache_serves_full_regardless_of_first_limit(self, monkeypatch):
        """缓存的是完整曲目：先小 limit 取，再大 limit 取仍能拿到完整列表，不因首次裁剪而丢失。"""
        from app.sources import netease as ns

        payload = {
            "code": 200,
            "album": {"name": "Album", "picUrl": "cover", "artist": {"name": "Artist"}, "size": 3},
            "songs": [
                {"id": i, "name": f"T{i}", "ar": [{"name": "Artist"}], "al": {"name": "Album", "picUrl": "cover"}}
                for i in range(1, 4)
            ],
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                import json
                return json.dumps(payload).encode()

        monkeypatch.setattr(ns.urllib.request, "urlopen", lambda *a, **k: FakeResponse())

        short = ns.fetch_netease_album_tracks("777", limit=1)
        full = ns.fetch_netease_album_tracks("777", limit=100)

        assert len(short["tracks"]) == 1
        assert len(full["tracks"]) == 3  # 缓存存的是完整 3 首，大 limit 不丢

    def test_clear_album_detail_cache_forces_refetch(self, monkeypatch):
        """clear_album_detail_cache 后，再次取同一专辑应重新请求网络。"""
        from app.sources import netease as ns

        calls = {"n": 0}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                import json
                return json.dumps({
                    "code": 200,
                    "album": {"name": "A", "size": 1},
                    "songs": [{"id": 1, "name": "X", "al": {}}],
                }).encode()

        def fake_urlopen(*a, **k):
            calls["n"] += 1
            return FakeResponse()

        monkeypatch.setattr(ns.urllib.request, "urlopen", fake_urlopen)

        ns.fetch_netease_album_tracks("999", limit=100)
        assert calls["n"] == 1
        assert ns.clear_album_detail_cache() == 1
        ns.fetch_netease_album_tracks("999", limit=100)
        assert calls["n"] == 2  # 清空后重新请求网络


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
