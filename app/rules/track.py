from __future__ import annotations

import re
from typing import Any

from app.config import settings
from app.rules.discover import _query_matches_track
from app.models import Asset, ExternalTrack


def _generic_metadata_title(title: str | None) -> bool:
    if not title:
        return True
    normalized = title.strip().lower()
    generic = {
        "网易云音乐",
        "qq音乐",
        "bilibili",
        "哔哩哔哩",
        "youtube",
        "cinesonic demo asset",
    }
    return normalized in {item.lower() for item in generic} or normalized.startswith("网易云音乐 -")


def _has_reliable_metadata(asset: Asset) -> bool:
    if _generic_metadata_title(asset.title):
        return False
    if asset.title.startswith("网易云歌曲 ") or asset.title == "CineSonic Demo Asset":
        return False
    return bool(asset.artist or asset.album or asset.genre or asset.mood)


def _query_needs_asset_context(query: str) -> bool:
    lowered = query.lower()
    media_terms = [
        "片段", "segment", "video", "素材", "场景", "镜头", "画面",
        "当前视频", "当前素材", "这个视频", "这个素材", "相似片段",
    ]
    return any(term in lowered for term in media_terms)


def _playlist_match_score(track: Asset | ExternalTrack, query: str) -> int:
    searchable = (
        f"{track.title} {getattr(track, 'artist', '') or ''} "
        f"{' '.join(getattr(track, 'genre', []) or [])} "
        f"{' '.join(getattr(track, 'mood', []) or [])}"
    ).lower()
    score = 0
    for term in query.lower().split():
        if term and term in searchable:
            score += 1
    return score


def _track_key(track: Asset | ExternalTrack | dict[str, Any]) -> str:
    if isinstance(track, Asset):
        return f"asset:{track.asset_id}"
    if isinstance(track, ExternalTrack):
        if track.external_id:
            return f"{track.source}:{track.external_id}"
        return f"title:{track.title.lower()}:{track.artist.lower()}"
    title = str(track.get("title", "")).lower().strip()
    artist = str(track.get("artist", "")).lower().strip()
    aid = str(track.get("asset_id", "")).strip()
    return f"asset:{aid}" if aid else f"title:{title}:{artist}"


def _is_verified_online_track(track: Asset | ExternalTrack) -> bool:
    return isinstance(track, ExternalTrack) and track.source == "netease"


def _is_local_recommendation_track(track: Asset | ExternalTrack) -> bool:
    return isinstance(track, Asset) or (
        isinstance(track, ExternalTrack)
        and track.source == "local"
        and bool(track.external_id or track.playback_url)
    )


def _is_verified_recommendation_track(track: Asset | ExternalTrack) -> bool:
    return _is_local_recommendation_track(track) or _is_verified_online_track(track)


def _is_fallback_track(track: Asset | ExternalTrack) -> bool:
    source = getattr(track, "source", "local")
    return "fallback" in source or source in {"mock", "llm"}


_ALLOWED_CANDIDATE_KINDS = {"track", "official_mv", "unknown"}


def _classify_candidate_kind(title: str, source: str) -> str:
    t = (title or "").lower()

    lyrics_signals = ["动态歌词", "歌词版", "歌词视频", "lyric video", "lyrics video", "(lyrics)", "[lyrics]"]
    if any(sig in t for sig in lyrics_signals):
        return "lyrics_video"

    playlist_signals = ["歌单", "playlist", "排行榜", "top chart", "网易云歌单"]
    if any(sig in t for sig in playlist_signals):
        return "playlist"

    long_mix_signals = [
        "non-stop", "nonstop", "megamix", "mega mix", "dj mix",
        "连续播放", "一直播放", "纯音乐合集", "睡眠歌单",
    ]
    if "remix" not in t:
        if any(sig in t for sig in long_mix_signals):
            return "long_mix"
        if re.search(r"\bmix\b\s*\d*\)?\s*$", t):
            return "long_mix"
    if re.search(r"\d+\s*(?:小时|hours?|hrs?)\b", t):
        return "long_mix"

    compilation_signals = [
        "合集", "连播", "串烧", "歌曲合集", "精选集", "全部歌曲",
        "全部曲目", "经典回顾", "最全", "歌曲大全", "纯享合集", "金曲合集",
        "full album", "all songs", "greatest hits", "compilation",
        "best of", "歌曲串烧",
    ]
    if any(sig in t for sig in compilation_signals):
        return "compilation"
    if re.search(r"\d+\s*首", t) or re.search(r"\d+\s*songs?\b", t):
        return "compilation"

    mv_signals = ["mv", "live", "现场", "演唱会", "官方视频", "official video", "official music video", "music video"]
    if any(sig in t for sig in mv_signals):
        return "official_mv"

    return "track"


def _valid_external_track(track: ExternalTrack, query: str) -> bool:
    title = (track.title or "").strip()
    if not title:
        return False
    lowered_title = title.lower()
    lowered_query = query.lower().strip()
    if lowered_title == lowered_query:
        return False
    if lowered_title in {"网易云音乐", "bilibili", "youtube", "搜索结果"}:
        return False
    if len(title) > 80 and " - " not in title:
        return False
    if getattr(track, "candidate_kind", "track") not in _ALLOWED_CANDIDATE_KINDS:
        return False
    if not _query_matches_track(query, track):
        return False
    return True


def _online_candidate_reason(track: ExternalTrack, memory_query: str) -> str:
    source_label = {
        "netease": "网易云真实曲目",
        "bilibili": "B 站真实视频/MV",
        "youtube": "YouTube 真实视频",
    }.get(track.source, "真实线上候选")
    if memory_query:
        return f"online_candidate：来自{source_label}，并结合你的记忆偏好「{memory_query[:40]}」排序。"
    return f"online_candidate：来自{source_label}，不是本地 mock 结果。"


def _dedupe_tracks(tracks: list[Asset | ExternalTrack]) -> list[Asset | ExternalTrack]:
    seen: set[str] = set()
    seen_titles: set[str] = set()
    unique: list[Asset | ExternalTrack] = []
    for track in tracks:
        key = _track_key(track)
        title_key = f"{track.title.strip().lower()}|{(track.artist or '').strip().lower()}"
        if key in seen or title_key in seen_titles:
            continue
        seen.add(key)
        seen_titles.add(title_key)
        unique.append(track)
    return unique


def _merge_search_queries(query: str, variants: list[str] | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in [query, *(variants or [])]:
        value = (item or "").strip()
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        out.append(value)
        if len(out) >= settings.max_search_variants + 1:
            break
    return out


def _filter_excluded_tracks(
    tracks: list[Asset | ExternalTrack],
    excluded: list[dict[str, str]],
) -> list[Asset | ExternalTrack]:
    if not excluded:
        return tracks
    seen_keys: set[tuple[str, str]] = set()
    seen_titles: set[str] = set()
    for ex in excluded:
        title = ex.get("title", "").lower().strip()
        sid = ex.get("source_id", "").strip()
        if title:
            seen_titles.add(title)
            if sid:
                seen_keys.add((title, sid))
    filtered: list[Asset | ExternalTrack] = []
    for track in tracks:
        track_title = (getattr(track, "title", "") or "").lower().strip()
        track_sid = getattr(track, "external_id", "") or getattr(track, "asset_id", "") or ""
        if track_title and track_sid and (track_title, track_sid) in seen_keys:
            continue
        if track_title and track_title in seen_titles:
            continue
        filtered.append(track)
    return filtered


def _fill_tracks(
    tracks: list[Asset | ExternalTrack],
    candidates: list[Asset | ExternalTrack],
    target_count: int,
) -> list[Asset | ExternalTrack]:
    merged = _dedupe_tracks(tracks)
    seen = {_track_key(track) for track in merged}
    for candidate in candidates:
        if len(merged) >= target_count:
            break
        key = _track_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        merged.append(candidate)
    return merged[:target_count]
