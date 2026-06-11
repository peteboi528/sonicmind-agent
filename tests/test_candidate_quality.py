"""候选质量过滤：合集/连播视频识别 + _valid_external_track。

修复 P0：推荐 The Weeknd 只出 B站/YouTube 视频合集的污染问题。
"""
from __future__ import annotations

from app.agent import _classify_candidate_kind, _valid_external_track
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
