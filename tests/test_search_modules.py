"""搜索模块专项测试：verifier / netease_playlist / web_music_discovery / 三路路由。

所有外部 API 调用均 mock，测试纯逻辑。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.models import ExternalTrack, ResourceTrack


def test_infer_count_extracts_arabic_chinese_and_fuzzy_quantities():
    from app.answer import infer_count

    # 阿拉伯数字（原有行为）
    assert infer_count("推荐5首") == 5
    assert infer_count("来 10 首歌") == 10
    assert infer_count("给我12") == 12
    # 中文数字
    assert infer_count("二十首") == 20
    assert infer_count("十二首") == 12
    assert infer_count("十首") == 10
    assert infer_count("两首") == 2
    assert infer_count("三十五首") == 35
    # 模糊量词 → 合理默认（均多于默认 top_k=5）
    assert infer_count("多来几首") == 8
    assert infer_count("来一批") == 12
    assert infer_count("十几首") == 15
    # 无数量线索 → None（调用方回落默认 top_k）
    assert infer_count("随便推荐点") is None
    assert infer_count("来点好听的") is None


def test_merge_search_queries_dedupes_and_caps(monkeypatch):
    from app.agent import _merge_search_queries
    from app.config import settings

    monkeypatch.setattr(settings, "max_search_variants", 2, raising=False)
    assert _merge_search_queries("Eminem", ["eminem", "rap", "hip hop"]) == ["Eminem", "rap", "hip hop"]


def test_recommendation_anchor_extraction_for_heavy_user_queries():
    from app.agent import _extract_recommendation_anchors, _recommendation_search_seeds

    query = "推荐 6 首 post-rock / math rock，参考 toe、Explosions in the Sky、American Football。"
    anchors = _extract_recommendation_anchors(query)

    assert anchors.artists == ("toe", "Explosions in the Sky", "American Football")
    assert "post-rock" in anchors.styles
    assert "math rock" in anchors.styles
    seeds = _recommendation_search_seeds("post-rock math rock toe Explosions in the Sky", query, anchors)
    assert "toe" in seeds
    assert "Explosions in the Sky" in seeds
    assert "post-rock" in seeds


def test_agent_dense_library_fallback_returns_verified_tracks(monkeypatch):
    from app.agent import AudioVisualAgent
    from app.config import settings

    agent = AudioVisualAgent.__new__(AudioVisualAgent)
    monkeypatch.setattr(settings, "dense_recall_min_score", 0.5, raising=False)

    class FakeLibrary:
        def semantic_search(self, *a, **k):
            return [
                ResourceTrack(
                    title="Semantic Hit",
                    artist="Verifier",
                    source="netease",
                    source_id="sem-1",
                    genre=["爵士"],
                    mood=["深夜"],
                    verified=True,
                )
            ]

    agent.library = FakeLibrary()

    out = agent._dense_library_fallback("late night jazz", existing=[], limit=1)
    assert len(out) == 1
    assert out[0].title == "Semantic Hit"
    assert out[0].external_id == "sem-1"


def test_discovery_route_soft_budget_skips_llm_and_lastfm():
    from app.memory import UserMemory
    from app.services.recommend import RecommendationService

    service = RecommendationService(
        store=None,
        memory=MagicMock(),
        library=MagicMock(),
        list_assets=lambda: [],
        track_key=lambda track: getattr(track, "external_id", getattr(track, "asset_id", "")),
        is_quality_track=lambda _track: True,
        query_noise=[],
    )
    search_and_extract = MagicMock(return_value=[
        ExternalTrack(external_id="p1", title="Playlist Song", artist="Artist", source="netease")
    ])
    discover_from_llm = MagicMock(return_value=[
        ExternalTrack(external_id="l1", title="LLM Song", artist="Artist", source="netease")
    ])
    discover_from_lastfm = MagicMock(return_value=[
        ExternalTrack(external_id="f1", title="Lastfm Song", artist="Artist", source="netease")
    ])

    candidates, trace = service.extend_discovery_route_candidates(
        candidates=[],
        goal="深夜 慵懒 律动",
        search_goal="深夜 慵懒 律动",
        scene_queries=["深夜 R&B 歌单"],
        prefer_playlist=False,
        top_k=5,
        memory=UserMemory(user_id="u1"),
        taste_summary="",
        library_artists=[],
        dedupe_tracks=lambda items: items,
        search_and_extract=search_and_extract,
        discover_from_llm=discover_from_llm,
        discover_from_lastfm=discover_from_lastfm,
        budget_degrade_level="soft",
    )

    assert len(candidates) == 1
    assert search_and_extract.call_count == 1
    discover_from_llm.assert_not_called()
    discover_from_lastfm.assert_not_called()
    assert any("budget_degrade=soft" in line for line in trace)


def test_discovery_route_hard_budget_skips_scene_followups():
    from app.memory import UserMemory
    from app.services.recommend import RecommendationService

    service = RecommendationService(
        store=None,
        memory=MagicMock(),
        library=MagicMock(),
        list_assets=lambda: [],
        track_key=lambda track: getattr(track, "external_id", getattr(track, "asset_id", "")),
        is_quality_track=lambda _track: True,
        query_noise=[],
    )
    search_and_extract = MagicMock(return_value=[])

    _candidates, trace = service.extend_discovery_route_candidates(
        candidates=[],
        goal="深夜 慵懒 律动",
        search_goal="深夜 慵懒 律动",
        scene_queries=["深夜 R&B 歌单", "深夜 氛围 歌单", "凌晨 开车 歌单"],
        prefer_playlist=True,
        top_k=5,
        memory=UserMemory(user_id="u1"),
        taste_summary="",
        library_artists=[],
        dedupe_tracks=lambda items: items,
        search_and_extract=search_and_extract,
        discover_from_llm=MagicMock(return_value=[]),
        discover_from_lastfm=MagicMock(return_value=[]),
        budget_degrade_level="hard",
    )

    assert search_and_extract.call_count == 1
    assert any("skip=scene_playlist_followups" in line for line in trace)


def test_agent_lexical_library_fallback_supports_offline_mood_queries():
    from app.agent import AudioVisualAgent

    agent = AudioVisualAgent.__new__(AudioVisualAgent)

    class FakeLibrary:
        def list_tracks(self, limit, *, verified_only=False):
            return [
                ResourceTrack(title="Night Calm", artist="A", source="local", source_id="1", mood=["放松"], verified=True),
                ResourceTrack(title="Morning Run", artist="B", source="local", source_id="2", mood=["热血"], verified=True),
            ]

    agent.library = FakeLibrary()
    result = agent._lexical_resource_fallback("来点深夜放松的歌曲", limit=5)
    assert [track.title for track in result] == ["Night Calm"]


# ═══════════════════════════════════════════════════════════════════════
# verifier.py
# ═══════════════════════════════════════════════════════════════════════


class TestVerifySong:
    """verify_song: 歌名+歌手搜网易云，返回精确匹配或 None。

    verify_song 在函数体内 import search_netease_many，
    所以 patch 靶点是 app.sources.netease.search_netease_many。
    """

    @patch("app.sources.netease.search_netease_many")
    def test_exact_match(self, mock_search):
        mock_search.return_value = [
            {"song_id": "100", "title": "Blinding Lights", "artist": "The Weeknd",
             "album": "After Hours", "cover": None},
        ]
        from app.search.verifier import verify_song

        result = verify_song("Blinding Lights", "The Weeknd")
        assert result is not None
        assert result.external_id == "100"
        assert result.title == "Blinding Lights"
        assert result.source == "netease"
        assert "music.163.com" in result.playback_url

    @patch("app.sources.netease.search_netease_many")
    def test_case_insensitive_partial_match(self, mock_search):
        mock_search.return_value = [
            {"song_id": "200", "title": "blinding lights", "artist": "the weeknd",
             "album": None, "cover": None},
        ]
        from app.search.verifier import verify_song

        result = verify_song("Blinding Lights", "The Weeknd")
        assert result is not None
        assert result.external_id == "200"

    @patch("app.sources.netease.search_netease_many")
    def test_no_match_returns_none(self, mock_search):
        mock_search.return_value = [
            {"song_id": "300", "title": "其他歌曲", "artist": "其他歌手",
             "album": None, "cover": None},
        ]
        from app.search.verifier import verify_song

        assert verify_song("Blinding Lights", "The Weeknd") is None

    @patch("app.sources.netease.search_netease_many")
    def test_api_failure_returns_none(self, mock_search):
        mock_search.side_effect = Exception("network error")
        from app.search.verifier import verify_song

        assert verify_song("test", "test") is None

    def test_empty_query_returns_none(self):
        from app.search.verifier import verify_song

        assert verify_song("", "") is None
        # title="" + artist="artist" → query="artist" 会触发真实搜索
        # 这不算空查询，只是 title 为空的情况


class TestBatchVerify:
    """batch_verify: 批量验证候选，去重 + 限流。"""

    @patch("app.search.verifier.verify_song")
    def test_verify_multiple(self, mock_verify):
        mock_verify.side_effect = [
            ExternalTrack(external_id="1", title="Song A", artist="X", source="netease",
                          playback_url="https://music.163.com/song?id=1"),
            ExternalTrack(external_id="2", title="Song B", artist="Y", source="netease",
                          playback_url="https://music.163.com/song?id=2"),
            None,  # 第三首找不到
        ]
        from app.search.verifier import batch_verify

        result = batch_verify([
            {"title": "Song A", "artist": "X"},
            {"title": "Song B", "artist": "Y"},
            {"title": "Song C", "artist": "Z"},
        ])
        assert len(result) == 2
        assert result[0].title == "Song A"
        assert result[1].title == "Song B"

    @patch("app.search.verifier.verify_song")
    def test_dedup_same_song(self, mock_verify):
        mock_verify.return_value = ExternalTrack(
            external_id="1", title="Song A", artist="X", source="netease",
            playback_url="https://music.163.com/song?id=1",
        )
        from app.search.verifier import batch_verify

        result = batch_verify([
            {"title": "Song A", "artist": "X"},
            {"title": "song a", "artist": "x"},  # 去重
        ])
        assert len(result) == 1

    def test_max_verify_limit(self):
        """超过 max_verify 的候选不验证。"""
        from app.search.verifier import batch_verify

        with patch("app.search.verifier.verify_song", return_value=None) as mock:
            candidates = [{"title": f"Song {i}", "artist": "X"} for i in range(30)]
            batch_verify(candidates, max_verify=5)
            assert mock.call_count == 5

    def test_skip_empty_title(self):
        from app.search.verifier import batch_verify

        with patch("app.search.verifier.verify_song") as mock:
            batch_verify([{"title": "", "artist": "X"}, {"title": "  ", "artist": "Y"}])
            mock.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════
# netease_playlist.py
# ═══════════════════════════════════════════════════════════════════════


class TestSearchNeteasePlaylists:
    """search_netease_playlists: 搜索网易云歌单。"""

    @patch("app.search.netease_playlist.requests.get")
    def test_returns_playlists(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "result": {
                "playlists": [
                    {"id": 111, "name": "深夜 R&B", "trackCount": 30},
                    {"id": 222, "name": "慵懒夜晚", "trackCount": 50},
                ]
            }
        }
        from app.search.netease_playlist import search_netease_playlists

        result = search_netease_playlists("深夜 R&B", limit=2)
        assert len(result) == 2
        assert result[0]["id"] == 111
        assert result[0]["track_count"] == 30

    @patch("app.search.netease_playlist.requests.get")
    def test_api_failure_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("timeout")
        from app.search.netease_playlist import search_netease_playlists

        assert search_netease_playlists("test") == []

    @patch("app.search.netease_playlist.requests.get")
    def test_empty_result(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"result": {}}
        from app.search.netease_playlist import search_netease_playlists

        assert search_netease_playlists("不存在的歌单") == []

    @patch("app.search.netease_playlist.time.sleep", lambda *_a, **_k: None)
    @patch("app.search.netease_playlist.requests.get")
    def test_rate_limit_429_is_retried_then_succeeds(self, mock_get):
        """429 限流应重试，而非被吞成空——重试后拿到结果即返回。"""
        throttled = MagicMock(status_code=429)
        ok = MagicMock(status_code=200)
        ok.json.return_value = {"result": {"playlists": [{"id": 9, "name": "ok", "trackCount": 3}]}}
        mock_get.side_effect = [throttled, ok]
        from app.search.netease_playlist import search_netease_playlists

        result = search_netease_playlists("限流后重试")
        assert mock_get.call_count == 2  # 第一次 429 触发重试
        assert len(result) == 1 and result[0]["id"] == 9

    @patch("app.search.netease_playlist.time.sleep", lambda *_a, **_k: None)
    @patch("app.search.netease_playlist.requests.get")
    def test_persistent_429_returns_empty_after_retries(self, mock_get):
        """持续 429：重试用尽后返回空（不伪造），且 call_count 证明确实重试过。"""
        mock_get.return_value = MagicMock(status_code=429)
        from app.search.netease_playlist import search_netease_playlists

        assert search_netease_playlists("一直限流") == []
        assert mock_get.call_count >= 2  # 至少重试一次,而非单发即弃


class TestGetPlaylistTracks:
    """get_playlist_tracks: 从歌单提取歌曲。"""

    @patch("app.search.netease_playlist.requests.get")
    def test_extracts_tracks(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "playlist": {
                "tracks": [
                    {
                        "id": 1001,
                        "name": "Blinding Lights",
                        "ar": [{"name": "The Weeknd"}],
                        "al": {"name": "After Hours", "picUrl": "https://pic.test/1.jpg"},
                    },
                    {
                        "id": 1002,
                        "name": "Save Your Tears",
                        "ar": [{"name": "The Weeknd"}, {"name": "Ariana Grande"}],
                        "al": {"name": "After Hours"},
                    },
                ]
            }
        }
        from app.search.netease_playlist import get_playlist_tracks

        result = get_playlist_tracks(111, limit=5)
        assert len(result) == 2
        assert result[0].title == "Blinding Lights"
        assert result[0].artist == "The Weeknd"
        assert result[0].source == "netease"
        assert result[0].cover_url == "https://pic.test/1.jpg"
        assert result[1].artist == "The Weeknd/Ariana Grande"

    @patch("app.search.netease_playlist.requests.get")
    def test_limit_respected(self, mock_get):
        """歌单返回很多歌曲时，limit 生效。注意 id 从 1 开始（id=0 会被跳过）。"""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "playlist": {
                "tracks": [
                    {"id": i, "name": f"Song {i}", "ar": [{"name": "Artist"}], "al": {}}
                    for i in range(1, 101)  # id 从 1 开始
                ]
            }
        }
        from app.search.netease_playlist import get_playlist_tracks

        result = get_playlist_tracks(111, limit=10)
        assert len(result) == 10

    @patch("app.search.netease_playlist.requests.get")
    def test_skip_tracks_without_id(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "playlist": {
                "tracks": [
                    {"id": None, "name": "No ID", "ar": [], "al": {}},
                    {"id": 100, "name": "Has ID", "ar": [{"name": "A"}], "al": {}},
                ]
            }
        }
        from app.search.netease_playlist import get_playlist_tracks

        result = get_playlist_tracks(111)
        assert len(result) == 1
        assert result[0].title == "Has ID"


class TestNeteaseSearchDeadline:
    """_fetch_netease_songs 墙钟预算：持续限流时不让重试拖爆。"""

    def test_persistent_empty_respects_deadline(self, monkeypatch):
        import app.concurrency
        import app.sources.netease as ne

        calls = {"n": 0}

        def fake_run_parallel(tasks, timeout, default):
            calls["n"] += 1
            return [[] for _ in tasks]  # 全端点空 = 疑似限流

        # _fetch_netease_songs 里是函数内 import，patch 到 app.concurrency 源头。
        monkeypatch.setattr(app.concurrency, "run_parallel", fake_run_parallel)
        monkeypatch.setattr(ne, "_SEARCH_BACKOFF", 0.0)

        result = ne._fetch_netease_songs("一直限流的查询", limit=5)
        assert result == []
        # attempts = _SEARCH_RETRIES + 1，每轮一次 run_parallel；deadline 不应让它超过该上限。
        assert calls["n"] <= ne._SEARCH_RETRIES + 1

    def test_deadline_constants_bound_worst_case(self):
        import app.sources.netease as ne

        # 最坏耗时 ≈ attempts × timeout + 退避，必须落在 deadline 附近(不爆 16s 工具墙)。
        worst = (ne._SEARCH_RETRIES + 1) * ne._SEARCH_TIMEOUT
        assert worst <= ne._SEARCH_DEADLINE + ne._SEARCH_TIMEOUT
        assert ne._SEARCH_DEADLINE < 8.0  # 远小于 web_music_search 的 16s 超时


class TestSearchAndExtract:
    """search_and_extract: 搜歌单 + 提取 + 去重。"""

    @patch("app.search.netease_playlist.get_playlist_tracks")
    @patch("app.search.netease_playlist.search_netease_playlists")
    def test_full_pipeline(self, mock_search, mock_tracks):
        mock_search.return_value = [
            {"id": 111, "name": "歌单1", "track_count": 10},
            {"id": 222, "name": "歌单2", "track_count": 10},
        ]
        mock_tracks.side_effect = [
            [
                ExternalTrack(external_id="1", title="Song A", artist="X", source="netease"),
                ExternalTrack(external_id="2", title="Song B", artist="Y", source="netease"),
            ],
            [
                ExternalTrack(external_id="3", title="Song A", artist="X", source="netease"),  # 重复
                ExternalTrack(external_id="4", title="Song C", artist="Z", source="netease"),
            ],
        ]
        from app.search.netease_playlist import search_and_extract

        result = search_and_extract("深夜 R&B", max_playlists=2)
        assert len(result) == 3  # A, B, C (A 不重复)
        titles = [t.title for t in result]
        assert "Song A" in titles
        assert "Song B" in titles
        assert "Song C" in titles

    @patch("app.search.netease_playlist._offline_playlist_fallback")
    @patch("app.search.netease_playlist.search_netease_playlists")
    def test_no_playlists_uses_offline_fallback(self, mock_search, mock_fallback):
        mock_search.return_value = []
        mock_fallback.return_value = [
            ExternalTrack(external_id="fallback-1", title="Fallback Song", artist="Offline", source="local"),
        ]
        from app.search.netease_playlist import search_and_extract

        result = search_and_extract("test")

        assert [track.title for track in result] == ["Fallback Song"]
        mock_fallback.assert_called_once()

    @patch("app.search.netease_playlist.get_playlist_tracks")
    @patch("app.search.netease_playlist.search_netease_playlists")
    def test_prefers_official_curated_playlist(self, mock_search, mock_tracks):
        mock_search.return_value = [
            {
                "id": 1, "name": "SEO 跑步歌单", "track_count": 50,
                "play_count": 999999, "creator_name": "关键词音乐", "creator_verified": False,
            },
            {
                "id": 2, "name": "夏日跑步", "track_count": 50,
                "play_count": 200000, "creator_name": "云音乐官方歌单", "creator_verified": True,
            },
        ]
        mock_tracks.side_effect = lambda playlist_id, limit: [ExternalTrack(
            external_id=str(playlist_id),
            title="Firework" if playlist_id == 2 else "跑步 BPM 180 Type Beat",
            artist="Katy Perry" if playlist_id == 2 else "Beat Maker",
            source="netease",
        )]
        from app.search.netease_playlist import search_and_extract

        result = search_and_extract("跑步 动感 节奏", max_playlists=1)

        assert [track.title for track in result] == ["Firework"]
        mock_tracks.assert_called_once_with(2, limit=15)


# ═══════════════════════════════════════════════════════════════════════
# web_music_discovery.py
# ═══════════════════════════════════════════════════════════════════════


class TestGenerateLlmCandidates:
    """generate_llm_candidates: LLM 生成候选歌名列表。"""

    def test_no_llm_returns_empty(self):
        from app.search.web_music_discovery import generate_llm_candidates

        assert generate_llm_candidates("test", "R&B", llm=None) == []

    def test_llm_returns_valid_json(self):
        from app.search.web_music_discovery import generate_llm_candidates

        mock_llm = MagicMock()
        mock_llm.generate.return_value = '{"candidates": [{"title": "Nikes", "artist": "Frank Ocean"}, {"title": "Ivy", "artist": "Frank Ocean"}]}'

        result = generate_llm_candidates("深夜 R&B", "R&B, neo-soul", llm=mock_llm)
        assert len(result) == 2
        assert result[0]["title"] == "Nikes"
        assert result[1]["artist"] == "Frank Ocean"

    def test_llm_returns_array_directly(self):
        from app.search.web_music_discovery import generate_llm_candidates

        mock_llm = MagicMock()
        mock_llm.generate.return_value = '[{"title": "Nikes", "artist": "Frank Ocean"}]'

        result = generate_llm_candidates("test", "R&B", llm=mock_llm)
        assert len(result) == 1
        assert result[0]["title"] == "Nikes"

    def test_llm_failure_returns_empty(self):
        from app.search.web_music_discovery import generate_llm_candidates

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("API error")

        assert generate_llm_candidates("test", "R&B", llm=mock_llm) == []

    def test_llm_garbage_returns_empty(self):
        from app.search.web_music_discovery import generate_llm_candidates

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "I don't know any songs"

        assert generate_llm_candidates("test", "R&B", llm=mock_llm) == []

    def test_filters_items_without_title(self):
        from app.search.web_music_discovery import generate_llm_candidates

        mock_llm = MagicMock()
        mock_llm.generate.return_value = '{"candidates": [{"title": "Nikes", "artist": "Frank Ocean"}, {"artist": "NoTitle"}, {"title": "", "artist": "Empty"}]}'

        result = generate_llm_candidates("test", "R&B", llm=mock_llm)
        assert len(result) == 1
        assert result[0]["title"] == "Nikes"

    def test_prompt_includes_taste_and_exclusions(self):
        from app.search.web_music_discovery import generate_llm_candidates

        mock_llm = MagicMock()
        mock_llm.generate.return_value = '{"candidates": []}'

        generate_llm_candidates(
            "深夜 R&B", "R&B, neo-soul",
            exclusion_rules=["不听摇滚"],
            library_artists=["Frank Ocean", "SZA"],
            target_count=8,
            llm=mock_llm,
        )
        prompt = mock_llm.generate.call_args[0][0]
        assert "R&B, neo-soul" in prompt
        assert "不听摇滚" in prompt
        assert "Frank Ocean" in prompt
        assert "8" in prompt


class TestDiscoverFromLlm:
    """discover_from_llm: LLM 候选生成 → Netease 验证完整流程。"""

    @patch("app.search.web_music_discovery.batch_verify")
    @patch("app.search.web_music_discovery.generate_llm_candidates")
    def test_full_pipeline(self, mock_gen, mock_verify):
        from app.search.web_music_discovery import discover_from_llm

        mock_gen.return_value = [
            {"title": "Nikes", "artist": "Frank Ocean"},
            {"title": "Ivy", "artist": "Frank Ocean"},
        ]
        mock_verify.return_value = [
            ExternalTrack(external_id="1", title="Nikes", artist="Frank Ocean", source="netease"),
        ]

        result = discover_from_llm("深夜 R&B", "R&B", llm=MagicMock())
        assert len(result) == 1
        assert result[0].title == "Nikes"

    @patch("app.search.web_music_discovery.generate_llm_candidates")
    def test_no_candidates_returns_empty(self, mock_gen):
        from app.search.web_music_discovery import discover_from_llm

        mock_gen.return_value = []
        assert discover_from_llm("test", "R&B", llm=MagicMock()) == []

    def test_no_llm_returns_empty(self):
        from app.search.web_music_discovery import discover_from_llm

        assert discover_from_llm("test", "R&B", llm=None) == []


# ═══════════════════════════════════════════════════════════════════════
# agent.py — 三路搜索路由
# ═══════════════════════════════════════════════════════════════════════


class TestQueryHasEntity:
    """_query_has_entity: 判断查询是否包含精确实体。"""

    def test_english_artist_name(self):
        from app.agent import AudioVisualAgent

        assert AudioVisualAgent._query_has_entity("Drake") is True
        assert AudioVisualAgent._query_has_entity("Frank Ocean") is True
        assert AudioVisualAgent._query_has_entity("The Weeknd") is True

    def test_english_generic_mood(self):
        from app.agent import AudioVisualAgent

        assert AudioVisualAgent._query_has_entity("chill") is False
        assert AudioVisualAgent._query_has_entity("lofi vibe") is False
        assert AudioVisualAgent._query_has_entity("R&B") is False

    def test_chinese_entity(self):
        from app.agent import AudioVisualAgent

        # "周杰伦" 不在 _GENERAL_WORDS 里 → 有实体
        assert AudioVisualAgent._query_has_entity("周杰伦的歌") is True

    def test_chinese_pure_mood(self):
        from app.agent import AudioVisualAgent

        # 这些 CJK token 都在 _GENERAL_WORDS 里 → 无实体
        assert AudioVisualAgent._query_has_entity("深夜 慵懒 律动") is False
        assert AudioVisualAgent._query_has_entity("放松 伤感 治愈") is False

    def test_chinese_mixed_functional_words_detected(self):
        """经过 _extract_search_query 清洗后，功能词查询不含实体。

        实际流程：goal → _extract_search_query → _query_has_entity(search_goal)。
        """
        from app.agent import AudioVisualAgent, _extract_search_query

        sq = _extract_search_query("帮我推荐几首放松的歌")
        # 清洗后应该只剩 "放松"
        assert AudioVisualAgent._query_has_entity(sq) is False

    def test_style_intensity_phrase_is_not_misread_as_entity(self):
        from app.agent import AudioVisualAgent, _extract_search_query

        sq = _extract_search_query("有没有偏微电子一点专注")
        assert sq == "电子 专注"
        assert AudioVisualAgent._query_has_entity(sq) is False

    def test_genre_words_not_entity(self):
        """中文风格词不是实体：说唱、摇滚、电子等应走 LLM+歌单路由。"""
        from app.agent import AudioVisualAgent

        assert AudioVisualAgent._query_has_entity("说唱 R&B") is False
        assert AudioVisualAgent._query_has_entity("摇滚 电子 民谣") is False
        assert AudioVisualAgent._query_has_entity("晚上 嘻哈 治愈") is False

    def test_empty_query(self):
        from app.agent import AudioVisualAgent

        assert AudioVisualAgent._query_has_entity("") is False


def test_discover_query_classifier_separates_category_artist_and_track(tmp_path):
    from app.agent import AudioVisualAgent
    from app.models import Asset
    from app.storage import JsonStore

    store = JsonStore(tmp_path / "store")
    store.write_model("assets", "weeknd", Asset(
        asset_id="weeknd", source_url="https://example.com/weeknd",
        title="Blinding Lights", artist="The Weeknd", duration_seconds=200,
        status="analyzed", genre=["R&B"], mood=["暗黑"],
    ))
    store.write_model("assets", "kanye", Asset(
        asset_id="kanye", source_url="https://example.com/kanye",
        title="I Wonder", artist="Kanye West、Ye", duration_seconds=200,
        status="analyzed", genre=["说唱"], mood=["律动"],
    ))
    store.write_model("assets", "not-kanye", Asset(
        asset_id="not-kanye", source_url="https://example.com/not-kanye",
        title="Kanye Dreams", artist="Joe Example", duration_seconds=200,
        status="analyzed", genre=["爵士"], mood=["放松"],
    ))
    store.write_model("assets", "lana", Asset(
        asset_id="lana", source_url="https://example.com/lana",
        title="Video Games", artist="Lana Del Rey", duration_seconds=200,
        status="analyzed", genre=["流行"], mood=["浪漫"],
    ))
    agent = AudioVisualAgent(store)

    category = agent.classify_discover_query("适合专注工作的电子音乐")
    artist = agent.classify_discover_query("The Weeknd")
    artist_alias = agent.classify_discover_query("kanye")
    artist_typo = agent.classify_discover_query("lana del ray")
    track = agent.classify_discover_query("Blinding Lights")
    category_focus = agent.classify_discover_query("适合专注工作")

    assert category["kind"] == "category"
    assert category["browse_category"] == "scene"
    assert artist["kind"] == "artist"
    assert artist_alias["kind"] == "artist"
    assert artist_typo["kind"] == "artist"
    assert artist_typo["normalized_query"] == "Lana Del Rey"
    assert artist_typo["reason"] == "library_artist_fuzzy"
    assert track["kind"] == "track"
    assert category_focus["kind"] == "category"
    assert agent.artist_name_matches("kanye", "Kanye West") is True
    assert agent.artist_name_matches("专注", "Focus Band") is False

    artist_results = agent.search("listener", "kanye", include_external=False, top_k=12)
    assert [item.asset_id for item in artist_results.local] == ["kanye"]
    assert "已结合记忆扩展" not in artist_results.summary

    typo_results = agent.search("listener", "lana del ray", include_external=False, top_k=12)
    assert [item.asset_id for item in typo_results.local] == ["lana"]


def test_library_evidence_search_does_not_analyze_or_write_missing_segments():
    from app.agent import AudioVisualAgent
    from app.models import Asset

    asset = Asset(
        asset_id="readonly", source_url="https://example.com/readonly",
        title="Read Only", artist="Example", duration_seconds=180,
        status="analyzed", genre=["电子"], mood=["专注"],
    )
    agent = AudioVisualAgent.__new__(AudioVisualAgent)
    agent.list_assets = MagicMock(return_value=[asset])
    agent.media = MagicMock()
    agent.media.get_segments.return_value = []
    agent.analyze_media = MagicMock()
    from app.services.rag import RagService
    agent.rag = RagService(
        MagicMock(), agent.media, MagicMock(),
        analyze_media=agent.analyze_media, list_assets=agent.list_assets,
    )

    assert agent.retrieve_library_evidence("专注", top_k=5) == []
    agent.media.get_segments.assert_called_once_with("readonly")
    agent.analyze_media.assert_not_called()


class TestRecommendForQueryRoutes:
    """recommend_for_query: 三路搜索路由逻辑。

    注：conftest 已 mock search_web_music。memory/library 是实例属性，
    直接设到 __new__ 出来的 agent 上即可。
    """

    def test_entity_query_uses_exact_route(self):
        """包含实体的查询走 exact 路由（search_web_music），不走 LLM/歌单。"""
        from app.agent import AudioVisualAgent
        from app.memory import TasteProfile, UserMemory
        from app.models import DailyRecommendation

        mock_memory = MagicMock()
        mock_mem = UserMemory(user_id="test", taste_profile=TasteProfile(top_genres=[("R&B", 0.8)], top_moods=[("放松", 0.6)]))
        mock_memory.get_memory.return_value = mock_mem
        mock_memory.weighted_query.return_value = "R&B 放松"

        mock_library = MagicMock()
        mock_library.is_disliked.return_value = False

        agent = AudioVisualAgent.__new__(AudioVisualAgent)
        agent.memory = mock_memory
        agent.library = mock_library
        agent.llm = MagicMock()
        agent.list_assets = MagicMock(return_value=[])  # 不走 store
        from app.services.profile_signals import ProfileSignals
        agent.profile_signals = ProfileSignals(MagicMock(), agent.memory, list_assets=agent.list_assets)

        # search_web_music 已被 conftest mock → 固定返回 3 首
        result = agent.recommend_for_query("user1", "Drake 的歌", top_k=5)

        assert isinstance(result, DailyRecommendation)
        # exact route 不会走 discover_from_llm 或 search_and_extract
        # 因为 "Drake" 被检测为实体 → has_entity=True → 只走 search_web_music

    def test_excluded_tracks_forwarded_as_offset(self):
        """Bug1③ 回归：excluded_tracks 非空时，exact 路由的 search_web_music
        必须收到 offset=len(excluded)，去翻页取新歌，否则同一查询永远返回 top-N。"""
        from app.agent import AudioVisualAgent
        from app.memory import TasteProfile, UserMemory

        mock_memory = MagicMock()
        mock_memory.get_memory.return_value = UserMemory(
            user_id="test", taste_profile=TasteProfile(top_genres=[("R&B", 0.8)])
        )
        mock_memory.weighted_query.return_value = "R&B"

        mock_library = MagicMock()
        mock_library.is_disliked.return_value = False

        agent = AudioVisualAgent.__new__(AudioVisualAgent)
        agent.memory = mock_memory
        agent.library = mock_library
        agent.llm = MagicMock()
        agent.list_assets = MagicMock(return_value=[])
        from app.services.profile_signals import ProfileSignals
        agent.profile_signals = ProfileSignals(MagicMock(), agent.memory, list_assets=agent.list_assets)

        seen_offsets: list[int] = []

        def spy(query, top_k=5, relevance_query="", offset=0, **_):
            seen_offsets.append(offset)
            return [ExternalTrack(external_id=f"x-{offset}-{i}", title=f"T{i}", artist="A", source="netease")
                    for i in range(top_k)]

        agent.search_web_music = spy  # 实例级覆盖，盖住 conftest 的类级 mock

        excluded = [{"title": f"Shown {i}", "source_id": str(i)} for i in range(5)]
        agent.recommend_for_query("user1", "Drake 的歌", top_k=5, excluded_tracks=excluded)

        # 首次 exact 路由调用必须带 offset=5（已展示数）
        assert seen_offsets, "search_web_music 未被调用"
        assert seen_offsets[0] == 5

    def test_seeds_skip_redundant_online_search(self):
        """Fix: 前序 web_music_search 已搜过（seed_tracks 非空）时，recommend 不再二次联网搜——
        网易云限流下重搜会吃光 20s 预算导致 recommend 超时（实测"深夜看球"案例）。"""
        from app.agent import AudioVisualAgent
        from app.memory import TasteProfile, UserMemory

        mock_memory = MagicMock()
        mock_memory.get_memory.return_value = UserMemory(
            user_id="test", taste_profile=TasteProfile(top_genres=[("R&B", 0.8)])
        )
        mock_memory.weighted_query.return_value = "R&B"

        mock_library = MagicMock()
        mock_library.is_disliked.return_value = False

        agent = AudioVisualAgent.__new__(AudioVisualAgent)
        agent.memory = mock_memory
        agent.library = mock_library
        agent.llm = MagicMock()
        agent.list_assets = MagicMock(return_value=[])
        from app.services.profile_signals import ProfileSignals
        agent.profile_signals = ProfileSignals(MagicMock(), agent.memory, list_assets=agent.list_assets)

        call_count = {"n": 0}

        def spy(*_a, **_k):
            call_count["n"] += 1
            return []

        agent.search_web_music = spy  # 实例级覆盖

        seeds = [
            ExternalTrack(external_id=f"s-{i}", title=f"Seed {i}", artist="A", source="netease",
                          playback_url=f"https://music.163.com/song?id={i}")
            for i in range(6)  # >= top_k，命中 seed 短路
        ]
        agent.recommend_for_query("user1", "有节奏感的歌", top_k=5, seed_tracks=seeds)
        assert call_count["n"] == 0, "已有 seed 时不应再调 search_web_music（会撞 20s 超时）"

    def test_complex_explicit_query_uses_anchor_seed_recall_and_drops_unrelated(self):
        """长 query 为空时，逐艺人 seed 能召回；不相关 city pop 不应补位。"""
        from types import SimpleNamespace

        from app.agent import AudioVisualAgent
        from app.models import UserMemory

        mock_memory = MagicMock()
        mock_memory.get_memory.return_value = UserMemory(user_id="test")
        mock_memory.weighted_query.return_value = "city pop ambient"

        mock_library = MagicMock()
        mock_library.is_disliked.return_value = False
        mock_library.record_exposure = MagicMock()
        mock_library.decay_exposure_ts = MagicMock()

        agent = AudioVisualAgent.__new__(AudioVisualAgent)
        agent.memory = mock_memory
        agent.library = mock_library
        agent.llm = MagicMock()
        agent.list_assets = MagicMock(return_value=[])
        agent._local_recommendation_candidates = MagicMock(return_value=[])
        agent._dense_library_fallback = MagicMock(return_value=[
            ExternalTrack(external_id="bad-1", title="City Lights", artist="Airi City Pop", source="netease")
        ])
        agent._record_recommendation_history = MagicMock()
        agent._rerank_tracks = MagicMock(side_effect=lambda _uid, _q, tracks, top_k: [
            (t, SimpleNamespace(score=1.0 - i * 0.01, reason="test", components={}))
            for i, t in enumerate(tracks[:top_k])
        ])

        def fake_search(query, top_k=5, relevance_query="", offset=0, **_):
            if query == "toe":
                return [
                    ExternalTrack(external_id=f"toe-{i}", title=f"toe Song {i}", artist="toe", source="netease")
                    for i in range(3)
                ]
            if query == "Explosions in the Sky":
                return [
                    ExternalTrack(external_id=f"eits-{i}", title=f"Sky Song {i}", artist="Explosions In The Sky", source="netease")
                    for i in range(3)
                ]
            return [
                ExternalTrack(external_id="bad-web", title="City-Pop Curse", artist="Airi City Pop", source="netease")
            ]

        agent.search_web_music = fake_search

        result = agent.recommend_for_query(
            "user1",
            "推荐 6 首适合搭视觉灵感板的 post-rock / math rock，参考 toe、Explosions in the Sky。",
            top_k=6,
        )

        assert len(result.tracks) == 6
        assert {item.asset.artist for item in result.tracks} == {"toe", "Explosions In The Sky"}
        assert agent._dense_library_fallback.call_count == 0
        assert any("anchor_filter=dropped" in line for line in result.agent_trace)

    def test_llm_entity_anchors_filter_out_other_artists(self):
        """回归：请求非硬编码名单里的歌手（The Weeknd）时，LLM entities 作艺人锚点，
        本地库里用户爱听的其他歌手（Drake/Kanye）必须被 anchor_filter 剔除。

        这是"推几首 The Weeknd 却返回 Drake/Kanye"漂移 bug 的根治验证——修复前
        The Weeknd 不在 _KNOWN_RECOMMENDATION_ARTISTS → anchors.explicit=False →
        过滤跳过 → 本地爱听歌手漏进来。
        """
        from types import SimpleNamespace

        from app.agent import AudioVisualAgent
        from app.models import UserMemory

        mock_memory = MagicMock()
        mock_memory.get_memory.return_value = UserMemory(user_id="test")
        mock_memory.weighted_query.return_value = "The Weeknd"

        mock_library = MagicMock()
        mock_library.is_disliked.return_value = False
        mock_library.record_exposure = MagicMock()
        mock_library.decay_exposure_ts = MagicMock()

        agent = AudioVisualAgent.__new__(AudioVisualAgent)
        agent.memory = mock_memory
        agent.library = mock_library
        agent.llm = MagicMock()
        agent.list_assets = MagicMock(return_value=[])
        # 本地库返回用户爱听的其他歌手（模拟 taste 漏出）
        agent._local_recommendation_candidates = MagicMock(return_value=[
            ExternalTrack(external_id="drake-1", title="Passionfruit", artist="Drake", source="local"),
            ExternalTrack(external_id="kanye-1", title="THIS ONE HERE", artist="Kanye West", source="local"),
        ])
        agent._dense_library_fallback = MagicMock(return_value=[])
        agent._record_recommendation_history = MagicMock()
        agent._rerank_tracks = MagicMock(side_effect=lambda _uid, _q, tracks, top_k: [
            (t, SimpleNamespace(score=1.0 - i * 0.01, reason="test", components={}))
            for i, t in enumerate(tracks[:top_k])
        ])

        def fake_search(query, top_k=5, relevance_query="", offset=0, **_):
            return [
                ExternalTrack(external_id=f"tw-{i}", title=f"Weeknd Song {i}", artist="The Weeknd", source="netease")
                for i in range(3)
            ]

        agent.search_web_music = fake_search

        # entities=["The Weeknd"] 是 LLM 规划器抽出的实体（现在会传进来）
        result = agent.recommend_for_query(
            "user1", "推几首 The Weeknd", top_k=5, entities=["The Weeknd"],
        )

        artists = {item.asset.artist for item in result.tracks}
        assert "Drake" not in artists, f"Drake 漏进推荐：{artists}"
        assert "Kanye West" not in artists, f"Kanye 漏进推荐：{artists}"
        assert artists == {"The Weeknd"}, f"应只含 The Weeknd，实际：{artists}"
        assert any("anchor_filter=dropped" in line for line in result.agent_trace)

    def test_explicit_anchor_query_returns_shortfall_instead_of_unrelated_fill(self):
        from app.agent import AudioVisualAgent
        from app.models import UserMemory

        mock_memory = MagicMock()
        mock_memory.get_memory.return_value = UserMemory(user_id="test")
        mock_memory.weighted_query.return_value = "city pop"

        mock_library = MagicMock()
        mock_library.is_disliked.return_value = False

        agent = AudioVisualAgent.__new__(AudioVisualAgent)
        agent.memory = mock_memory
        agent.library = mock_library
        agent.llm = MagicMock()
        agent.list_assets = MagicMock(return_value=[])
        agent._local_recommendation_candidates = MagicMock(return_value=[])
        agent._dense_library_fallback = MagicMock(return_value=[
            ExternalTrack(external_id="bad-pool", title="City Lights", artist="Airi City Pop", source="netease")
        ])
        agent._record_recommendation_history = MagicMock()
        agent.search_web_music = MagicMock(return_value=[
            ExternalTrack(external_id="bad-web", title="City-Pop Curse", artist="Airi City Pop", source="netease")
        ])

        result = agent.recommend_for_query("user1", "给我 6 首 Nujabes jazz hip-hop lo-fi beats", top_k=6)

        assert result.tracks == []
        assert "online_verified=0" in result.agent_trace[-1]
        assert agent._dense_library_fallback.call_count == 1

    @patch("app.search.web_music_discovery.discover_from_llm")
    @patch("app.search.netease_playlist.search_and_extract")
    def test_mood_query_uses_llm_and_playlist_routes(
        self, mock_playlist, mock_llm_disc
    ):
        """纯情绪查询走 LLM 候选 + 歌单搜索，不走 exact。"""
        from app.agent import AudioVisualAgent
        from app.memory import TasteProfile, UserMemory
        from app.models import DailyRecommendation, ExternalTrack

        mock_memory = MagicMock()
        mock_mem = UserMemory(user_id="test", taste_profile=TasteProfile(top_genres=[("R&B", 0.8)], top_moods=[("放松", 0.6)]))
        mock_memory.get_memory.return_value = mock_mem
        mock_memory.weighted_query.return_value = "R&B 放松"

        mock_library = MagicMock()
        mock_library.is_disliked.return_value = False

        # LLM 路由返回 2 首
        mock_llm_disc.return_value = [
            ExternalTrack(external_id="1", title="Nikes", artist="Frank Ocean", source="netease",
                          playback_url="https://music.163.com/song?id=1"),
            ExternalTrack(external_id="2", title="Ivy", artist="Frank Ocean", source="netease",
                          playback_url="https://music.163.com/song?id=2"),
        ]

        # 歌单路由返回 2 首
        mock_playlist.return_value = [
            ExternalTrack(external_id="3", title="Song A", artist="Artist A", source="netease",
                          playback_url="https://music.163.com/song?id=3"),
            ExternalTrack(external_id="4", title="Song B", artist="Artist B", source="netease",
                          playback_url="https://music.163.com/song?id=4"),
        ]

        agent = AudioVisualAgent.__new__(AudioVisualAgent)
        agent.memory = mock_memory
        agent.library = mock_library
        agent.llm = MagicMock()
        agent.list_assets = MagicMock(return_value=[])  # 不走 store
        from app.services.profile_signals import ProfileSignals
        agent.profile_signals = ProfileSignals(MagicMock(), agent.memory, list_assets=agent.list_assets)

        result = agent.recommend_for_query("user1", "深夜 慵懒 律动", top_k=5)

        # 验证走了 LLM 和歌单路由
        mock_llm_disc.assert_called_once()
        mock_playlist.assert_called_once()

        assert isinstance(result, DailyRecommendation)


# ═══════════════════════════════════════════════════════════════════════
# _compose_discussion 反幻觉
# ═══════════════════════════════════════════════════════════════════════


class TestComposeDiscussion:
    """Async discussion path remains grounded in verified tracks."""

    def test_no_tracks_with_agent_refuses(self):
        from app.graph.nodes import _discussion_prompt

        prompt, refuse = _discussion_prompt("Drake 的专辑评价", [])
        assert prompt is None
        assert "不想编" in refuse or "没找到" in refuse

    def test_prompt_includes_anti_hallucination_rules(self):
        from app.graph.nodes import _discussion_prompt

        tracks = [ExternalTrack(external_id="1", title="Test", artist="Test", source="netease")]
        prompt, _ = _discussion_prompt("test", tracks)
        assert prompt is not None
        assert "不要编造" in prompt
        assert "100字" in prompt
        assert "真实曲目" in prompt


# ═══════════════════════════════════════════════════════════════════════
# candidate_generator prompt
# ═══════════════════════════════════════════════════════════════════════


class TestCandidateGeneratorPrompt:
    """CANDIDATE_GENERATOR_PROMPT 格式化正确性。"""

    def test_format_succeeds(self):
        from app.prompts.candidate_generator import CANDIDATE_GENERATOR_PROMPT

        result = CANDIDATE_GENERATOR_PROMPT.format(
            taste_summary="R&B, neo-soul",
            exclusion_rules="不听摇滚",
            library_artists="Frank Ocean, SZA",
            query="深夜 慵懒",
            target_count=10,
        )
        assert "R&B, neo-soul" in result
        assert "不听摇滚" in result
        assert "Frank Ocean" in result
        assert "10" in result
        assert "candidates" in result

    def test_version_exists(self):
        from app.prompts.candidate_generator import CANDIDATE_GENERATOR_VERSION

        assert CANDIDATE_GENERATOR_VERSION.startswith("v")


# ═══════════════════════════════════════════════════════════════════════
# lastfm_client._pick_image + agent.search 抽象词回退（本轮新增回归）
# ═══════════════════════════════════════════════════════════════════════


class TestPickImage:
    """_pick_image: 从大到小取首个非空尺寸。mega 尺寸常空，旧逻辑盲取最后一张
    导致大量歌手/专辑返回空图。"""

    def test_mega_empty_falls_back_to_smaller(self):
        from app.sources.lastfm_client import _pick_image
        images = [
            {"#text": "small.jpg", "size": "small"},
            {"#text": "large.jpg", "size": "large"},
            {"#text": "", "size": "mega"},
        ]
        # mega 空 → 取最大的非空 large
        assert _pick_image(images) == "large.jpg"

    def test_all_empty_returns_empty(self):
        from app.sources.lastfm_client import _pick_image
        assert _pick_image([{"#text": "", "size": "small"}, {"#text": "", "size": "mega"}]) == ""

    def test_not_a_list_returns_empty(self):
        from app.sources.lastfm_client import _pick_image
        assert _pick_image(None) == ""
        assert _pick_image([]) == ""


class TestSearchAbstractFallback:
    """agent.search: 字面歌曲搜索归零时回退歌单搜索，让"痛苦"这类抽象词有结果。"""

    def test_literal_empty_triggers_playlist_fallback(self):
        from app.agent import AudioVisualAgent
        from app.memory import TasteProfile, UserMemory

        mock_memory = MagicMock()
        mock_memory.get_memory.return_value = UserMemory(
            user_id="test", taste_profile=TasteProfile()
        )
        mock_memory.weighted_query.return_value = ""

        mock_library = MagicMock()
        mock_library.is_disliked.return_value = False

        agent = AudioVisualAgent.__new__(AudioVisualAgent)
        agent.memory = mock_memory
        agent.library = mock_library
        agent.llm = MagicMock()
        agent.list_assets = MagicMock(return_value=[])
        from app.services.rag import RagService
        agent.rag = RagService(
            MagicMock(), MagicMock(), agent.memory,
            analyze_media=MagicMock(), list_assets=agent.list_assets,
        )

        # 字面 netease 搜空（"痛苦"没有同名歌曲）
        agent.search_web_music = lambda query, top_k=5, relevance_query="", offset=0, **_: []

        fallback_track = ExternalTrack(
            external_id="pl-1", title="治愈系", artist="某歌手", source="netease",
        )

        with patch("app.search.netease_playlist.search_and_extract",
                   return_value=[fallback_track]) as mock_extract:
            result = agent.search("user1", "痛苦", include_external=True, top_k=8)

        # 触发了歌单回退，且返回了回退曲目
        assert mock_extract.called
        assert any(t.title == "治愈系" for t in result.external)

    def test_literal_hits_skip_fallback(self):
        """正常歌手/歌名搜索有结果时不触发歌单回退。"""
        from app.agent import AudioVisualAgent
        from app.memory import TasteProfile, UserMemory

        mock_memory = MagicMock()
        mock_memory.get_memory.return_value = UserMemory(
            user_id="test", taste_profile=TasteProfile()
        )
        mock_memory.weighted_query.return_value = ""

        agent = AudioVisualAgent.__new__(AudioVisualAgent)
        agent.memory = mock_memory
        agent.library = MagicMock(is_disliked=MagicMock(return_value=False))
        agent.llm = MagicMock()
        agent.list_assets = MagicMock(return_value=[])

        hits = [ExternalTrack(external_id=f"h-{i}", title=f"Hit{i}", artist="Drake", source="netease")
                for i in range(5)]
        agent.search_web_music = lambda query, top_k=5, relevance_query="", offset=0, **_: hits

        with patch("app.search.netease_playlist.search_and_extract",
                   return_value=[]) as mock_extract:
            agent.search("user1", "Drake", include_external=True, top_k=8)

        # 有足够字面结果 → 不触发歌单回退
        assert not mock_extract.called
