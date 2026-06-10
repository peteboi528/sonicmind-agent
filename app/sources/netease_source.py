"""真实网易云外部源：取代 MockSource，让推荐/歌单补位走真实曲目而非硬编码假歌单。

实现 ExternalSource 协议（search / get_track / get_recommendations）。
所有候选都来自网易云搜索 API 的真实数据，不再返回 mock 目录里的假歌。
get_recommendations 用 seed_genres/seed_moods 拼成搜索词，捞同风格真实曲目。
"""
from __future__ import annotations

import logging

from app.models import ExternalTrack
from app.sources.netease import (
    fetch_netease_song_detail,
    search_netease_many,
)

logger = logging.getLogger(__name__)


def _to_external(meta: dict) -> ExternalTrack:
    return ExternalTrack(
        external_id=meta["song_id"],
        title=meta["title"],
        artist=meta.get("artist", ""),
        album=meta.get("album"),
        cover_url=meta.get("cover"),
        source="netease",
        playback_url=f"https://music.163.com/song?id={meta['song_id']}",
    )


class NeteaseSource:
    """基于网易云搜索 API 的真实外部源。"""

    def search(self, query: str, limit: int = 20) -> list[ExternalTrack]:
        metas = search_netease_many(query, limit=limit)
        return [_to_external(m) for m in metas]

    def get_track(self, external_id: str) -> ExternalTrack | None:
        detail = fetch_netease_song_detail(external_id)
        if not detail:
            return None
        return _to_external(detail)

    def get_recommendations(
        self, seed_genres: list[str], seed_moods: list[str], limit: int = 20
    ) -> list[ExternalTrack]:
        """用风格/情绪关键词搜真实曲目作推荐候选。

        逐个 seed 关键词搜索并去重，凑够 limit 即停。避免一次搜太多拖慢。
        """
        seen: set[str] = set()
        out: list[ExternalTrack] = []
        # 风格优先，情绪兜底；都搜中文关键词，网易云中文召回更准
        keywords = [g for g in seed_genres if g] + [m for m in seed_moods if m]
        per_kw = max(limit // max(len(keywords), 1), 5)
        for kw in keywords:
            if len(out) >= limit:
                break
            for meta in search_netease_many(kw, limit=per_kw):
                sid = meta["song_id"]
                if sid in seen:
                    continue
                seen.add(sid)
                out.append(_to_external(meta))
                if len(out) >= limit:
                    break
        return out
