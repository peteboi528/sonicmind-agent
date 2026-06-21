"""候选质量过滤：合集/连播视频识别 + _valid_external_track。

修复 P0：推荐 The Weeknd 只出 B站/YouTube 视频合集的污染问题。
"""
from __future__ import annotations

from app.agent import (
    _classify_candidate_kind,
    _is_playlist_context_compatible,
    _is_recommendation_quality_track,
    _query_requests_variant_content,
    _valid_external_track,
)
from app.models import ExternalTrack


class TestCandidateKindClassifier:
    def test_compilation_chinese(self):
        for title in [
            "The Weeknd 歌曲合集 经典回顾",
            "周杰伦热门歌曲连播",
            "欧美流行串烧 100首",
            "华语金曲合集 纯享",
        ]:
            assert _classify_candidate_kind(title, "bilibili") == "compilation", title

    def test_compilation_english(self):
        for title in [
            "The Weeknd Greatest Hits Full Album",
            "The Best of Coldplay",
            "Coldplay Compilation 2024",
        ]:
            assert _classify_candidate_kind(title, "youtube") == "compilation", title

    def test_compilation_count_signal(self):
        assert _classify_candidate_kind("精选 50首 一次听个够", "bilibili") == "compilation"
        assert _classify_candidate_kind("Top 20 songs", "youtube") == "compilation"

    def test_official_mv_kept(self):
        assert _classify_candidate_kind("Blinding Lights (Official Video)", "youtube") == "official_mv"
        assert _classify_candidate_kind("周杰伦 - 晴天 MV", "bilibili") == "official_mv"
        assert _classify_candidate_kind("Adele Live at the BBC", "youtube") == "official_mv"

    def test_lyrics_video(self):
        for title in [
            "Blinding Lights (Lyrics)",
            "周杰伦 晴天 动态歌词",
            "Someone Like You Lyric Video",
        ]:
            assert _classify_candidate_kind(title, "youtube") == "lyrics_video", title

    def test_playlist(self):
        for title in [
            "欧美流行 精选歌单",
            "Chill Vibes Playlist 2024",
            "华语流行排行榜",
        ]:
            assert _classify_candidate_kind(title, "bilibili") == "playlist", title

    def test_long_mix(self):
        for title in [
            "Non-stop EDM Mix 2024",
            "Deep House DJ Mix",
            "Lofi 2 hours 连续播放",
        ]:
            assert _classify_candidate_kind(title, "youtube") == "long_mix", title

    def test_remix_is_track_not_long_mix(self):
        # 单曲 Remix 不应误判为 long_mix
        assert _classify_candidate_kind("Blinding Lights (Chromatics Remix)", "netease") == "track"

    def test_plain_track(self):
        assert _classify_candidate_kind("Blinding Lights", "netease") == "track"
        assert _classify_candidate_kind("晴天", "netease") == "track"


class TestValidExternalTrackFiltersCompilation:
    def test_compilation_dropped(self):
        track = ExternalTrack(
            external_id="bv1", title="The Weeknd 歌曲合集",
            artist="", source="bilibili", candidate_kind="compilation",
        )
        assert _valid_external_track(track, "The Weeknd") is False

    def test_playlist_dropped(self):
        track = ExternalTrack(
            external_id="bv2", title="Chill Playlist",
            artist="", source="bilibili", candidate_kind="playlist",
        )
        assert _valid_external_track(track, "Chill") is False

    def test_lyrics_video_dropped(self):
        track = ExternalTrack(
            external_id="yt3", title="Blinding Lights Lyrics",
            artist="The Weeknd", source="youtube", candidate_kind="lyrics_video",
        )
        assert _valid_external_track(track, "The Weeknd") is False

    def test_long_mix_dropped(self):
        track = ExternalTrack(
            external_id="yt4", title="EDM Non-stop Mix",
            artist="", source="youtube", candidate_kind="long_mix",
        )
        assert _valid_external_track(track, "EDM") is False

    def test_track_kept(self):
        track = ExternalTrack(
            external_id="n1", title="Blinding Lights",
            artist="The Weeknd", source="netease", candidate_kind="track",
        )
        assert _valid_external_track(track, "The Weeknd") is True

    def test_official_mv_kept(self):
        track = ExternalTrack(
            external_id="yt1", title="Blinding Lights (Official Video)",
            artist="The Weeknd", source="youtube", candidate_kind="official_mv",
        )
        assert _valid_external_track(track, "The Weeknd") is True

    def test_typo_artist_fuzzy_match_kept(self):
        track = ExternalTrack(
            external_id="n2", title="Lose Yourself",
            artist="Eminem", source="netease", candidate_kind="track",
        )
        assert _valid_external_track(track, "Emenem") is True

    def test_rnb_normalization_kept(self):
        track = ExternalTrack(
            external_id="n3", title="Late Night RnB",
            artist="SZA", source="netease", candidate_kind="track",
        )
        assert _valid_external_track(track, "R&B") is True

    def test_unrelated_entity_rejected(self):
        track = ExternalTrack(
            external_id="n4", title="Lose Yourself",
            artist="Eminem", source="netease", candidate_kind="track",
        )
        assert _valid_external_track(track, "Taylor Swift") is False


class TestRecommendationQualityGate:
    def test_filters_keyword_spam_and_production_assets(self):
        noisy = [
            ("跑步音乐 180步频（033）动感节奏 卡点", "音符糖"),
            ('（FREE）“Only You” R&B+Drake+Trapsoul - Type beat', "ZN Kill This Vibe"),
            ("青春的狂欢（伴奏）", "律动R&B"),
            ("Neo Soul Beat", "Chill5"),
            ("#梦核 #新风格探索 #trap", "SAAC"),
            ("热门歌曲", "热门歌曲"),
            ("学习专注", "学习能量"),
            ("学习专注力读书音乐", "休闲音乐"),
        ]
        for index, (title, artist) in enumerate(noisy):
            track = ExternalTrack(
                external_id=str(index), title=title, artist=artist, source="netease"
            )
            assert _is_recommendation_quality_track(track) is False, title

    def test_keeps_normal_official_tracks(self):
        for index, (title, artist) in enumerate([
            ("Blinding Lights", "The Weeknd"),
            ("搬家", "张震岳"),
            ("Firework", "Katy Perry"),
            ("Butter (Hotter Remix)", "BTS"),
        ]):
            track = ExternalTrack(
                external_id=str(index), title=title, artist=artist, source="netease"
            )
            assert _is_recommendation_quality_track(track) is True, title

    def test_explicit_type_beat_request_can_bypass_gate(self):
        track = ExternalTrack(
            external_id="beat-1", title="Drake Type Beat", artist="Producer", source="netease"
        )
        assert _query_requests_variant_content("给我一些 Drake Type Beat") is True
        assert _is_recommendation_quality_track(track, allow_variants=True) is True


class TestPlaylistContextGate:
    def test_running_playlist_rejects_sleep_study_and_noise_candidates(self):
        noisy = [
            ("高音质/白噪音/雨水声/雷声/放松/催眠/学习/睡前音乐/工作学习必备", "纯音乐馆", "netease"),
            ("工作学习时听的音乐 提高专注力", "曹哲瀚", "netease"),
            ("雨林/下雨声/自然白噪音/工作/学习/睡觉", "纯音乐馆", "netease"),
            ("推荐歌曲：《Chasing Tonight(Slowed)》Richz一定要带上耳机", "桥尼娜臂样", "bilibili"),
            ("日推 |“绝对不能错过的慵懒调调！”|《踊り子》Vaundy", "漁小孩", "bilibili"),
        ]
        for index, (title, artist, source) in enumerate(noisy):
            track = ExternalTrack(
                external_id=f"bad-{index}", title=title, artist=artist, source=source
            )
            assert _is_playlist_context_compatible("帮我做 20 首跑步歌单", track) is False, title

    def test_running_playlist_keeps_normal_song_candidates(self):
        clean = [
            ("Blinding Lights", "The Weeknd"),
            ("Lose Yourself", "Eminem"),
            ("Firework", "Katy Perry"),
        ]
        for index, (title, artist) in enumerate(clean):
            track = ExternalTrack(
                external_id=f"ok-{index}", title=title, artist=artist, source="netease"
            )
            assert _is_playlist_context_compatible("帮我做 20 首跑步歌单", track) is True, title

    def test_non_running_playlist_does_not_apply_running_gate(self):
        track = ExternalTrack(
            external_id="sleep-1", title="雨声 白噪音", artist="纯音乐馆", source="netease"
        )
        assert _is_playlist_context_compatible("帮我做睡前放松歌单", track) is True
