"""Spotify Web API 客户端：genres/popularity/audio_features/related。

需 OAuth client credentials（SPOTIFY_CLIENT_ID/SECRET）。access_token 约 1 小时有效，
本 client 缓存并按过期时间自动刷新。凭证缺失或调用失败时返回空，知识链路降级。
与 lastfm/musicbrainz 同风格：同步 urllib，零新依赖。
"""
from __future__ import annotations

import base64
import json
import logging
import time
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_TOKEN_URL = "https://accounts.spotify.com/api/token"
_API_BASE = "https://api.spotify.com/v1"
_HEADERS = {"User-Agent": "MusicAgent/1.0"}
_TIMEOUT = 6


class SpotifyClient:
    def __init__(self, client_id: str = "", client_secret: str = "") -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self._token = ""
        self._token_expires_at = 0.0

    @property
    def available(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _ensure_token(self) -> str:
        if self._token and time.monotonic() < self._token_expires_at - 5:
            return self._token
        if not self.available:
            return ""
        try:
            data = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode()
            cred = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
            req = urllib.request.Request(_TOKEN_URL, data=data, method="POST", headers={
                **_HEADERS, "Authorization": f"Basic {cred}",
                "Content-Type": "application/x-www-form-urlencoded",
            })
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                payload = json.loads(resp.read().decode())
            self._token = payload.get("access_token", "")
            self._token_expires_at = time.monotonic() + int(payload.get("expires_in", 3600))
            return self._token
        except Exception:
            logger.debug("Spotify token fetch failed", exc_info=True)
            return ""

    def _get(self, path: str, **params: Any) -> dict:
        token = self._ensure_token()
        if not token:
            return {}
        url = f"{_API_BASE}/{path.lstrip('/')}?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(url, headers={**_HEADERS, "Authorization": f"Bearer {token}"})
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            logger.debug("Spotify API call failed: path=%s params=%s", path, params, exc_info=True)
            return {}

    def search_artist(self, name: str, limit: int = 1) -> dict | None:
        """返回 {id, name, genres, popularity, image}。无结果 None。"""
        data = self._get("search", q=name, type="artist", limit=limit)
        items = (data.get("artists") or {}).get("items") or []
        if not items:
            return None
        a = items[0]
        images = a.get("images") or []
        return {
            "id": a.get("id", ""),
            "name": a.get("name", ""),
            "genres": a.get("genres") or [],
            "popularity": a.get("popularity", 0),
            "image": (images[0].get("url", "") if images else ""),
        }

    def search_album(self, title: str, artist: str = "", limit: int = 1) -> dict | None:
        """返回 {id, name, artist, release_date, total_tracks, image}。无结果 None。"""
        q = f"album:{title}" if not artist else f"album:{title} artist:{artist}"
        data = self._get("search", q=q, type="album", limit=limit)
        items = (data.get("albums") or {}).get("items") or []
        if not items:
            return None
        al = items[0]
        artists = al.get("artists") or []
        images = al.get("images") or []
        return {
            "id": al.get("id", ""),
            "name": al.get("name", ""),
            "artist": (artists[0].get("name", "") if artists else ""),
            "release_date": al.get("release_date", ""),
            "total_tracks": al.get("total_tracks", 0),
            "image": (images[0].get("url", "") if images else ""),
        }

    def artist_top_track_ids(self, artist_id: str, limit: int = 3) -> list[str]:
        data = self._get(f"artists/{artist_id}/top-tracks", market="US")
        tracks = data.get("tracks") or []
        return [t.get("id", "") for t in tracks[:limit] if t.get("id")]

    def audio_features(self, track_ids: list[str]) -> list[dict]:
        """批量取音频特征（danceability/energy/valence/tempo/...）。"""
        if not track_ids:
            return []
        data = self._get("audio-features", ids=",".join(track_ids[:5]))
        return [f for f in (data.get("audio_features") or []) if isinstance(f, dict)]

    def audio_features_description(self, artist_id: str) -> str:
        """取艺人 top track 的平均音频特征，转成自然语言声音描述。

        这是 Phase 2 的核心价值：把 Spotify 的声学数据（推荐四锚里缺的声学锚）
        变成 dossier 可读的声音描述。
        """
        ids = self.artist_top_track_ids(artist_id, limit=3)
        feats = self.audio_features(ids)
        if not feats:
            return ""
        n = len(feats)
        avg = {k: sum(f.get(k, 0) for f in feats) / n
               for k in ("danceability", "energy", "valence", "acousticness")}
        tempo = sum(f.get("tempo", 0) for f in feats) / n
        bits: list[str] = []
        if avg["danceability"] > 0.7:
            bits.append("律动感强")
        elif avg["danceability"] < 0.4:
            bits.append("节奏内敛")
        if avg["energy"] > 0.7:
            bits.append("能量充沛")
        elif avg["energy"] < 0.4:
            bits.append("气质安静")
        if avg["valence"] > 0.6:
            bits.append("情绪明亮")
        elif avg["valence"] < 0.4:
            bits.append("情绪偏暗")
        if avg["acousticness"] > 0.5:
            bits.append("原声质感")
        if tempo:
            bits.append(f"约{int(tempo)}bpm")
        return "、".join(bits)
