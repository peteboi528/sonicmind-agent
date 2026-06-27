from __future__ import annotations

import re
from typing import Any

from app.media.pipeline import netease_song_id
from app.models import Asset, ExternalTrack
from app.sources import bilibili as bilibili_source
from app.sources import netease as netease_source
from app.sources import youtube as youtube_source


class PlaybackService:
    def __init__(
        self,
        *,
        extract_youtube_id: Any | None = None,
        extract_bilibili_id: Any | None = None,
        search_youtube_video: Any | None = None,
        search_bilibili_video: Any | None = None,
        search_netease: Any | None = None,
        get_netease_audio_url: Any | None = None,
        artist_name_matches: Any | None = None,
    ) -> None:
        self._extract_youtube_id = extract_youtube_id or youtube_source.extract_youtube_id
        self._extract_bilibili_id = extract_bilibili_id or bilibili_source.extract_bilibili_id
        self._search_youtube_video = search_youtube_video or youtube_source.search_youtube_video
        self._search_bilibili_video = search_bilibili_video or bilibili_source.search_bilibili_video
        self._search_netease = search_netease or netease_source.search_netease
        self._get_netease_audio_url = get_netease_audio_url or netease_source.get_netease_audio_url
        self._artist_name_matches = artist_name_matches or self._default_artist_name_matches

    def get_playback_url(self, track: Asset | ExternalTrack, netease_cookie: str = "") -> str | None:
        if isinstance(track, Asset) and track.source_url:
            video_id = self.extract_youtube_id(track.source_url)
            if video_id:
                return f"https://www.youtube.com/embed/{video_id}?autoplay=1&rel=0"
            bilibili = self.extract_bilibili_id(track.source_url)
            if bilibili:
                param, value = bilibili
                return f"https://player.bilibili.com/player.html?{param}={value}&autoplay=0&high_quality=1&danmaku=0"
            netease_id = netease_song_id(track.source_url)
            if netease_id:
                audio = self.get_netease_audio_url(netease_id, netease_cookie)
                if audio:
                    return audio
            video_id = self.search_youtube_video(f"{track.title} {track.artist or ''} official")
            if video_id:
                return f"https://www.youtube.com/embed/{video_id}?autoplay=1&rel=0"
            return track.source_url
        if isinstance(track, ExternalTrack):
            if track.playback_url and "listType=search" not in track.playback_url:
                return track.playback_url
            netease_id = self.search_netease(f"{track.title} {track.artist}")
            if netease_id:
                audio = self.get_netease_audio_url(netease_id, netease_cookie)
                if audio:
                    return audio
            video_id = self.search_youtube_video(f"{track.title} {track.artist}")
            if video_id:
                return f"https://www.youtube.com/embed/{video_id}?autoplay=1&rel=0"
            return None
        return None

    def get_audio_url(self, track: Any, netease_cookie: str = "") -> str | None:
        source_url = getattr(track, "source_url", "") or ""
        title = getattr(track, "title", "") or ""
        artist = getattr(track, "artist", "") or ""
        if source_url:
            netease_id = netease_song_id(source_url)
            if netease_id:
                return self.get_netease_audio_url(netease_id, netease_cookie)
        if title:
            netease_id = self.search_netease(f"{title} {artist}".strip())
            if netease_id:
                return self.get_netease_audio_url(netease_id, netease_cookie)
        return None

    def get_mv_url(self, track: Asset | ExternalTrack) -> str | None:
        if isinstance(track, Asset) and track.source_url:
            bilibili = self.extract_bilibili_id(track.source_url)
            if bilibili:
                param, value = bilibili
                return f"https://player.bilibili.com/player.html?{param}={value}&autoplay=0&high_quality=1&danmaku=0"
            video_id = self.extract_youtube_id(track.source_url)
            if video_id:
                return f"https://www.youtube-nocookie.com/embed/{video_id}?autoplay=1&rel=0"

        title = track.title
        artist = getattr(track, "artist", "") or ""
        bvid = self.search_bilibili_video(f"{title} {artist} MV".strip())
        if bvid:
            return f"https://player.bilibili.com/player.html?bvid={bvid}&autoplay=0&high_quality=1&danmaku=0"
        video_id = self.search_youtube_video(f"{title} {artist} MV official".strip())
        if video_id:
            return f"https://www.youtube-nocookie.com/embed/{video_id}?autoplay=1&rel=0"
        return None

    def get_lyrics(self, title: str, artist: str, source_id: str = "") -> dict[str, Any]:
        from app.sources.netease import fetch_netease_lyrics_timed, search_netease_many

        def norm(value: str) -> str:
            return re.sub(r"[^a-z0-9一-鿿㐀-䶿]+", "", (value or "").lower())

        song_id = str(source_id or "").strip()
        if not song_id.isdigit():
            song_id = ""
        resolved_title = title
        if not song_id and title:
            for candidate in search_netease_many(" ".join(filter(None, [title, artist])), limit=8):
                if norm(title) == norm(candidate.get("title", "")) and (
                    not artist or self.artist_name_matches(artist, candidate.get("artist", ""))
                ):
                    song_id = str(candidate.get("song_id", ""))
                    resolved_title = candidate.get("title", title)
                    break
        lines = fetch_netease_lyrics_timed(song_id) if song_id else []
        return {
            "title": resolved_title,
            "artist": artist,
            "song_id": song_id,
            "lines": lines,
            "found": bool(lines),
        }

    def extract_youtube_id(self, url: str) -> str | None:
        return self._extract_youtube_id(url)

    def extract_bilibili_id(self, url: str) -> tuple[str, str] | None:
        return self._extract_bilibili_id(url)

    def search_youtube_video(self, query: str) -> str | None:
        return self._search_youtube_video(query)

    def search_bilibili_video(self, query: str) -> str | None:
        return self._search_bilibili_video(query)

    def search_netease(self, query: str) -> str | None:
        return self._search_netease(query)

    def get_netease_audio_url(self, song_id: str, cookie: str = "") -> str | None:
        return self._get_netease_audio_url(song_id, cookie)

    def artist_name_matches(self, query: str, artist: str) -> bool:
        return self._artist_name_matches(query, artist)

    @staticmethod
    def _default_artist_name_matches(query: str, artist: str) -> bool:
        from app.rules.discover import _artist_query_matches

        return _artist_query_matches(query, artist, allow_fuzzy=True)
