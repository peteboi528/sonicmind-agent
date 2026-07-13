"""候选验证器：将 LLM 生成的或歌单提取的候选歌曲拿到网易云验证，
返回有真实播放链接的 ExternalTrack。找不到的丢弃。

对齐 SoulTuner 的 _extract_and_fetch_web_songs 思路：
LLM 提取歌名 → Netease 搜索验证 → 只保留有真实播放链接的。
"""

from __future__ import annotations

import logging

from app.models import ExternalTrack

logger = logging.getLogger(__name__)


def verify_song(title: str, artist: str) -> ExternalTrack | None:
    """用歌名+歌手搜网易云，返回第一个精确匹配的真实 ExternalTrack，找不到返回 None。"""
    from app.sources.netease import search_netease_many

    query = f"{artist} {title}".strip()
    if not query:
        return None

    try:
        results = search_netease_many(query, limit=5)
    except Exception:
        logger.debug("verify_song: Netease search failed for %r", query, exc_info=True)
        return None

    title_lower = title.lower().strip()
    artist_lower = artist.lower().strip()

    for meta in results:
        r_title = (meta.get("title") or "").lower().strip()
        r_artist = (meta.get("artist") or "").lower().strip()
        # 精确匹配：歌名或歌手至少有一边命中
        if title_lower in r_title or r_title in title_lower or artist_lower in r_artist or r_artist in artist_lower:
            return ExternalTrack(
                external_id=meta["song_id"],
                title=meta["title"],
                artist=meta.get("artist", ""),
                album=meta.get("album"),
                cover_url=meta.get("cover"),
                source="netease",
                playback_url=f"https://music.163.com/song?id={meta['song_id']}",
            )
    return None


def batch_verify(
    candidates: list[dict[str, str]],
    max_verify: int = 20,
) -> list[ExternalTrack]:
    """批量验证候选歌曲。每个候选是 {"title": ..., "artist": ...}。

    逐首调用 verify_song，返回验证通过的列表。找不到的静默丢弃。
    限制最大验证数量避免 API 限流。
    """
    verified: list[ExternalTrack] = []
    seen_queries: set[str] = set()

    for candidate in candidates[:max_verify]:
        title = candidate.get("title", "").strip()
        artist = candidate.get("artist", "").strip()
        if not title:
            continue
        # 去重
        key = f"{title.lower()}|{artist.lower()}"
        if key in seen_queries:
            continue
        seen_queries.add(key)

        track = verify_song(title, artist)
        if track is not None:
            verified.append(track)

    return verified
