"""Last.fm 发现 + Netease 验证：用品味档案驱动高质量音乐发现。

流程：
1. top_artists → artist.getSimilar → 相似艺人
2. 相似艺人 → artist.getTopTracks → 代表曲目
3. top_genres → tag.getTopTracks → 风格热门曲目
4. 合并去重 → verifier.batch_verify → 网易云逐首验证
5. 返回有真实播放链接的 ExternalTrack

Last.fm 只负责发现歌名+歌手，网易云负责验证和播放链接。
"""
from __future__ import annotations

import logging

from app.models import ExternalTrack

logger = logging.getLogger(__name__)

# 中文风格 → Last.fm tags
GENRE_TO_LASTFM_TAG: dict[str, list[str]] = {
    "说唱": ["hip-hop", "rap"],
    "R&B": ["rnb", "soul", "neo soul"],
    "摇滚": ["rock", "indie rock", "alternative"],
    "电子": ["electronic", "house", "techno"],
    "流行": ["pop"],
    "爵士": ["jazz", "blues"],
    "民谣": ["folk", "indie", "singer-songwriter"],
    "古典": ["classical"],
    "金属": ["metal", "heavy metal"],
    "国风": ["c-pop", "mandopop", "chinese"],
}


def discover_from_lastfm(
    top_artists: list[str] | None = None,
    top_genres: list[str] | None = None,
    target_count: int = 12,
) -> list[ExternalTrack]:
    """Last.fm 发现 + Netease 验证完整流程。

    返回有真实播放链接的 ExternalTrack，找不到的丢弃。
    """
    from app.config import settings
    from app.search.verifier import batch_verify
    from app.sources.lastfm_client import LastfmClient

    if not settings.lastfm_api_key:
        return []

    client = LastfmClient(settings.lastfm_api_key)
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()

    def _add(title: str, artist: str) -> None:
        key = f"{title.lower().strip()}|{artist.lower().strip()}"
        if key not in seen and title:
            seen.add(key)
            candidates.append({"title": title, "artist": artist})

    # ── 策略 1：从品味 top_artists 发现相似艺人 → 拿代表曲目 ──
    if top_artists:
        for artist in top_artists[:3]:
            similar = client.get_similar_artists(artist, limit=5)
            for sim_artist in similar[:3]:
                tracks = client.get_artist_top_tracks(sim_artist, limit=5)
                for t in tracks:
                    _add(t["title"], t["artist"])

    # ── 策略 2：从品味 top_genres 搜风格热门曲目 ──
    if top_genres:
        for genre in top_genres[:3]:
            tags = GENRE_TO_LASTFM_TAG.get(genre, [genre.lower()])
            for tag in tags[:1]:  # 每个风格只取第一个 tag
                tracks = client.get_tag_top_tracks(tag, limit=10)
                for t in tracks:
                    _add(t["title"], t["artist"])

    # ── 策略 3：从品味 top_artists 的热门曲目发现相似曲目 ──
    if top_artists and len(candidates) < target_count:
        for artist in top_artists[:2]:
            top_tracks = client.get_artist_top_tracks(artist, limit=3)
            for t in top_tracks[:1]:
                similar = client.get_similar_tracks(artist, t["title"], limit=5)
                for s in similar:
                    _add(s["title"], s["artist"])

    if not candidates:
        logger.debug("Last.fm discovery found 0 candidates")
        return []

    logger.debug("Last.fm discovery found %d candidates, verifying against Netease...", len(candidates))

    # ── 网易云验证 ──
    verified = batch_verify(candidates[:target_count * 2], max_verify=target_count * 2)
    logger.debug("Last.fm → Netease verified: %d/%d", len(verified), len(candidates))
    return verified
