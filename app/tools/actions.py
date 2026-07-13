"""Shared implementations for auxiliary Agent tools.

The LangGraph executor calls this module so side
effects, validation, and degradation behavior cannot drift between paths.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from app.models import DislikeRequest, MemoryUpdateRequest

if TYPE_CHECKING:
    from app.agent import AudioVisualAgent

logger = logging.getLogger(__name__)

AUX_TOOL_NAMES = {
    "feedback",
    "listening_history",
    "list_my_playlists",
    "find_on_platform",
    "lyrics",
    "audio_features",
    "save_to_playlist",
    "favorite_track",
    "concert_events",
}


def execute_aux_tool(
    agent: AudioVisualAgent,
    name: str,
    user_id: str,
    args: dict[str, Any],
    *,
    query: str = "",
    prior_tracks: list[Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Execute an auxiliary tool and return ``(structured_result, summary)``."""
    started = time.monotonic()
    try:
        if name == "feedback":
            result, summary = _feedback(agent, user_id, args, prior_tracks or [])
        elif name == "listening_history":
            result, summary = _listening_history(agent, user_id, args)
        elif name == "list_my_playlists":
            result, summary = _list_my_playlists(user_id)
        elif name == "find_on_platform":
            result, summary = _find_on_platform(agent, args)
        elif name == "lyrics":
            result, summary = _lyrics(agent, args)
        elif name == "audio_features":
            result, summary = _audio_features(agent, args)
        elif name in {"save_to_playlist", "favorite_track"}:
            result, summary = _preview_account_write(name, user_id, args)
        elif name == "concert_events":
            result, summary = _concert_events(agent, args)
        else:
            raise ValueError(f"Unsupported auxiliary tool: {name}")
        return result, summary
    finally:
        logger.info("tool=%s user=%s elapsed_ms=%d", name, user_id, int((time.monotonic() - started) * 1000))


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9一-鿿㐀-䶿]+", "", (value or "").lower())


def _resolve_track(agent: AudioVisualAgent, title: str, artist: str, prior_tracks: list[Any]) -> Any | None:
    title_key, artist_key = _norm(title), _norm(artist)
    candidates = [*prior_tracks, *agent.list_assets()]
    for track in candidates:
        candidate_title = _norm(getattr(track, "title", ""))
        candidate_artist = _norm(getattr(track, "artist", "") or "")
        if title_key and candidate_title != title_key:
            continue
        if artist_key and artist_key not in candidate_artist:
            continue
        if title_key or artist_key:
            return track
    return None


def _feedback(agent: AudioVisualAgent, user_id: str, args: dict[str, Any], prior_tracks: list[Any]):
    action = str(args.get("action", "")).lower().strip()
    if action not in {"like", "dislike", "skip", "played"}:
        raise ValueError("action must be like/dislike/skip/played")
    title = str(args.get("title", "")).strip()
    artist = str(args.get("artist", "")).strip()
    reason = str(args.get("reason", "")).strip()
    track = _resolve_track(agent, title, artist, prior_tracks)
    target = title or artist or "当前曲目"

    if action == "dislike":
        source = getattr(track, "source", "") if track else ""
        source_id = getattr(track, "external_id", "") or getattr(track, "asset_id", "") if track else ""
        agent.record_dislike(
            DislikeRequest(
                user_id=user_id,
                title=title,
                artist=artist,
                source=source,
                source_id=source_id,
                reason=reason,
            )
        )
        if artist and not title:
            agent.memory.add_exclusion(user_id, artist)
        summary = f"已记录不喜欢：{target}，后续推荐将排除或显著降权。"
    elif action == "like":
        if track:
            agent.library.update_ts_feedback(track, positive=True, weight=1.0)
        event = " ".join(part for part in ["我喜欢", artist, title] if part)
        agent.update_memory(MemoryUpdateRequest(user_id=user_id, event=event or f"喜欢 {target}"))
        summary = f"已记录喜欢：{target}。"
    elif action == "skip":
        if track:
            agent.library.update_ts_feedback(track, positive=False, weight=0.3)
        summary = f"已记录跳过：{target}（弱负反馈）。"
    else:
        track_id = getattr(track, "asset_id", "") or getattr(track, "external_id", "") if track else ""
        if track_id:
            agent.memory.record_listen(user_id, track_id, 0, False, context="agent_feedback")
            summary = f"已记录播放：{target}。"
        else:
            summary = f"未定位到真实曲目，仅保留了播放意图：{target}。"

    return {
        "type": "feedback",
        "action": action,
        "title": title,
        "artist": artist,
        "resolved": bool(track),
        "summary": summary,
    }, summary


def _listening_history(agent: AudioVisualAgent, user_id: str, args: dict[str, Any]):
    window = str(args.get("window", "recent"))
    group_by = str(args.get("group_by", "track"))
    top_k = max(1, min(int(args.get("top_k", 10) or 10), 50))
    memory = agent.memory.get_memory(user_id)
    events = list(memory.listening_history)
    now = datetime.now(UTC)
    cutoff = now - timedelta(days=7 if window == "week" else 30) if window in {"week", "month"} else None
    if cutoff:
        events = [event for event in events if _event_time(event.timestamp) >= cutoff]
    elif window == "recent":
        events = events[-50:]

    assets = {asset.asset_id: asset for asset in agent.list_assets()}
    counts: Counter[str] = Counter()
    labels: dict[str, dict[str, str]] = {}
    for event in events:
        asset = assets.get(event.asset_id)
        title = asset.title if asset else event.asset_id
        artist = (asset.artist or "") if asset else ""
        key = _norm(artist) if group_by == "artist" and artist else event.asset_id
        label = artist if group_by == "artist" and artist else title
        counts[key] += 1
        labels[key] = {"label": label, "title": title, "artist": artist, "asset_id": event.asset_id}
    items = [{**labels[key], "count": count} for key, count in counts.most_common(top_k)]
    summary = (
        f"{window} 听歌历史共 {len(events)} 次，按{('歌手' if group_by == 'artist' else '曲目')}汇总 {len(items)} 项。"
    )
    return {"type": "listening_history", "window": window, "group_by": group_by, "items": items}, summary


def _event_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=UTC)


def _list_my_playlists(user_id: str):
    from app import netease_auth

    info = netease_auth.load_cookie(user_id)
    if not info or not info.get("cookie"):
        summary = "尚未登录网易云，请先扫码登录。"
        return {"type": "auth_required", "what": "netease_login", "playlists": []}, summary
    playlists = netease_auth.fetch_user_playlists(info["cookie"])
    summary = f"读取到 {len(playlists)} 个网易云歌单。"
    return {"type": "my_playlists", "playlists": playlists}, summary


def _find_on_platform(agent: AudioVisualAgent, args: dict[str, Any]):
    title = str(args.get("title", "")).strip()
    artist = str(args.get("artist", "")).strip()
    platform = str(args.get("platform", "")).lower().strip()
    query = " ".join(part for part in [title, artist] if part)
    if platform == "netease":
        candidates = agent.search_web_music(query, top_k=8, relevance_query=title or query)
    elif platform in {"youtube", "bilibili"}:
        candidates = [track for track in agent.search_videos(query, top_k=8) if track.source == platform]
    else:
        raise ValueError("platform must be netease/youtube/bilibili")
    title_key = _norm(title)
    matched = [track for track in candidates if title_key and title_key in _norm(track.title)]
    result_tracks = matched[:3]
    summary = (
        f"已在 {platform} 找到 {len(result_tracks)} 个可追溯结果。"
        if result_tracks
        else f"未在 {platform} 找到可核实的《{title}》结果。"
    )
    return {"type": "find_on_platform", "platform": platform, "tracks": result_tracks}, summary


def _lyrics(agent: AudioVisualAgent, args: dict[str, Any]):
    from app.sources.netease import fetch_netease_lyrics, search_netease_many

    song_id = str(args.get("song_id", "")).strip()
    title = str(args.get("title", "")).strip()
    artist = str(args.get("artist", "")).strip()
    resolved: dict[str, Any] | None = None
    if not song_id and title:
        for candidate in search_netease_many(" ".join(filter(None, [title, artist])), limit=8):
            if _norm(title) == _norm(candidate.get("title", "")) and (
                not artist or agent.artist_name_matches(artist, candidate.get("artist", ""))
            ):
                resolved = candidate
                song_id = str(candidate.get("song_id", ""))
                break
    lines = fetch_netease_lyrics(song_id) if song_id else []
    summary = (
        f"已获取《{title or (resolved or {}).get('title', song_id)}》歌词 {len(lines)} 行。"
        if lines
        else "没有获取到可核实的歌词。"
    )
    return {"type": "lyrics", "song_id": song_id, "title": title, "artist": artist, "lines": lines}, summary


def _audio_features(agent: AudioVisualAgent, args: dict[str, Any]):
    asset_id = str(args.get("asset_id", "")).strip()
    track = None
    if asset_id:
        track = next((item for item in agent.list_assets() if item.asset_id == asset_id), None)
    if not track:
        track = _resolve_track(agent, str(args.get("title", "")), str(args.get("artist", "")), [])
    features = {
        "bpm": getattr(track, "tempo_bpm", None) if track else None,
        "energy": getattr(track, "energy_level", None) if track else None,
        "key": None,
        "danceability": None,
    }
    known = any(value is not None for value in features.values())
    # 诚实化：tempo/energy 可能是「基于标签估算」而非真实测量。只有 features_source=='measured'
    # 才算真实音频特征（目前没有任何路径产出 measured——DemoAnalyzer 不做真实分析，估算走
    # app/recommend/features.py 并标 'estimated'）。绝不把估算值冒充测量值告诉用户。
    source = getattr(track, "features_source", None) if track else None
    measured = known and source == "measured"
    if measured:
        summary = "已读取曲库中的真实音频特征。"
    elif known:
        summary = "基于曲风/情绪标签估算的能量与节奏（非真实测量）；曲库暂无音频文件可做真实分析。"
    else:
        summary = "当前没有可用的能量/节奏特征；不会用随机值补齐。"
    return {
        "type": "audio_features",
        "asset_id": getattr(track, "asset_id", ""),
        "features": features,
        "measured": measured,
        "features_source": source,
    }, summary


def _preview_account_write(name: str, user_id: str, args: dict[str, Any]):
    confirm = bool(args.get("confirm", False))
    action = {
        "tool": name,
        "playlist_id": str(args.get("playlist_id", "")),
        "track_ids": [str(item) for item in args.get("track_ids", [])],
        "track_id": str(args.get("track_id", "")),
    }
    if not confirm:
        summary = "这是账号写操作预览；请明确确认后再执行。"
        return {"type": "confirmation_required", "confirmed": False, "action": action}, summary
    from app import netease_auth

    info = netease_auth.load_cookie(user_id)
    if not info or not info.get("cookie"):
        return {
            "type": "auth_required",
            "what": "netease_login",
            "action": action,
        }, "尚未登录网易云，未执行任何写操作。"
    return {
        "type": "unsupported_write",
        "confirmed": True,
        "executed": False,
        "action": action,
    }, "当前版本尚无经过验证的网易云写接口，已安全拒绝，账号未发生变化。"


def _concert_events(agent: AudioVisualAgent, args: dict[str, Any]):
    artist = str(args.get("artist", "")).strip()
    city = str(args.get("city", "")).strip()
    query = " ".join(filter(None, [artist, city, "演出 巡演 官方"])).strip()
    sources = agent.search_artist_info(query)
    events: list[dict[str, Any]] = []
    unverified_sources: list[dict[str, Any]] = []
    seen_verified: set[tuple[str, str]] = set()
    seen_weak: set[tuple[str, str]] = set()
    for item in sources:
        title = str(item.get("title", "") or "")
        content = str(item.get("content", "") or "")
        url = item.get("url", "")
        host = _source_label(url)
        event = {
            "title": title or "未命名演出信息",
            "venue": _extract_venue_text(f"{title} {content}"),
            "date_text": _extract_date_text(f"{title} {content}"),
            "city": city or _extract_city_text(f"{title} {content}"),
            "source_name": host,
            "source_url": url,
            "summary": content,
        }
        event["kind"] = "event" if any(event.get(k) for k in ("date_text", "city", "venue")) else "tour_page"
        event["source_tier"] = _concert_source_tier(url, artist)
        verified, weak_reason = _looks_like_verified_event_signal(title, content, url, artist=artist, city=city)
        dedupe_key = (_norm(event["title"]), event["source_url"] or host)
        if verified:
            if dedupe_key in seen_verified:
                continue
            seen_verified.add(dedupe_key)
            events.append(event)
        else:
            if dedupe_key in seen_weak:
                continue
            seen_weak.add(dedupe_key)
            unverified_sources.append(
                {
                    "title": title or "未命名线索页",
                    "source_name": host,
                    "source_url": url,
                    "reason": weak_reason,
                }
            )
    events.sort(key=_concert_event_sort_key)
    concrete = [event for event in events if event.get("kind") == "event"]
    pages = [event for event in events if event.get("kind") != "event"]
    events = [*concrete, *pages]
    summary = (
        f"整理出 {len(events)} 条可核实演出事件，另有 {len(unverified_sources)} 条弱线索来源。"
        if events or unverified_sources
        else "暂未找到可核实的演出信息。"
    )
    return {
        "type": "concert_events",
        "artist": artist,
        "city": city,
        "events": events,
        "unverified_sources": unverified_sources,
    }, summary


def _extract_date_text(text: str) -> str:
    import re

    match = re.search(r"(20\d{2}[./-]\d{1,2}[./-]\d{1,2})", text)
    if match:
        return match.group(1)
    match = re.search(r"\b(20\d{2})(\d{2})(\d{2})\b", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    match = re.search(r"\b(20\d{2})\b", text)
    return match.group(1) if match else ""


def _extract_city_text(text: str) -> str:
    for city in ("上海", "北京", "广州", "深圳", "杭州", "成都", "东京", "首尔", "London", "New York", "Los Angeles"):
        if city.lower() in text.lower():
            return city
    return ""


def _extract_venue_text(text: str) -> str:
    for venue in ("启德体育园", "体育馆", "体育场", "Arena", "Stadium", "Hall", "Center"):
        if venue.lower() in text.lower():
            return venue
    return ""


def _looks_like_verified_event_signal(
    title: str, content: str, url: str, *, artist: str = "", city: str = ""
) -> tuple[bool, str]:
    from urllib.parse import urlparse

    text = " ".join([title or "", content or "", url or ""]).lower()
    host = urlparse(url or "").netloc.lower().removeprefix("www.")
    weak_hosts = {
        "music.apple.com",
        "open.spotify.com",
        "threads.com",
        "instagram.com",
        "x.com",
        "twitter.com",
        "facebook.com",
        "wikipedia.org",
        "en.wikipedia.org",
    }
    weak_terms = ("歌单", "playlist", "粉丝团", "threads", "forum", "讨论")
    event_terms = ("tour", "concert", "live", "巡演", "演出", "场馆", "门票", "tickets", "stadium", "arena")
    if host in weak_hosts:
        return False, "weak_host"
    if any(term in text for term in weak_terms) and not any(term in text for term in event_terms):
        return False, "weak_term"
    if _is_stale_event_text(text):
        return False, "stale_event"
    if artist and not _concert_artist_match(text, artist):
        return False, "artist_mismatch"
    extracted_city = _extract_city_text(text)
    if city and extracted_city and extracted_city.lower() != city.lower():
        return False, "city_mismatch"
    if not any(term in text for term in event_terms):
        return False, "no_event_signal"
    return True, ""


def _is_stale_event_text(text: str) -> bool:
    date_text = _extract_date_text(text)
    if not date_text:
        return False
    current_year = datetime.now(UTC).year
    year_match = re.search(r"20\d{2}", date_text)
    if not year_match:
        return False
    year = int(year_match.group(0))
    if year < current_year:
        return True
    if re.match(r"20\d{2}-\d{2}-\d{2}$", date_text):
        try:
            event_date = datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=UTC)
            return event_date.date() < datetime.now(UTC).date()
        except ValueError:
            return False
    return False


def _source_label(url: str) -> str:
    from urllib.parse import urlparse

    host = urlparse(url).netloc.lower()
    if not host:
        return "web"
    return host.removeprefix("www.")


def _concert_source_tier(url: str, artist: str) -> str:
    host = _source_label(url)
    artist_key = _norm(artist)
    if artist_key and artist_key in _norm(host):
        return "official"
    if any(token in host for token in ("ticketmaster", "livenation", "axs", "bandsintown", "songkick")):
        return "ticketing"
    if any(token in host for token in ("stadium", "arena", "theater", "theatre", "center", "sportspark", "venue")):
        return "venue"
    if any(token in host for token in ("trip.com", "shazam", "setlist", "eventbrite")):
        return "aggregator"
    return "web"


def _concert_artist_match(text: str, artist: str) -> bool:
    raw_artist = (artist or "").strip()
    if not raw_artist:
        return True
    if re.fullmatch(r"[A-Za-z0-9 .&'_-]+", raw_artist):
        return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(raw_artist)}(?![A-Za-z0-9])", text or "", flags=re.I))
    return _norm(raw_artist) in _norm(text)


def _concert_event_sort_key(event: dict[str, Any]) -> tuple[int, int, int, str]:
    tier_rank = {"official": 0, "ticketing": 1, "venue": 2, "aggregator": 3, "web": 4}
    kind_rank = 0 if event.get("kind") == "event" else 1
    date_rank = 99999999
    parsed = _parse_concert_date(event.get("date_text", ""))
    if parsed is not None:
        date_rank = int(parsed.strftime("%Y%m%d"))
    return (
        kind_rank,
        tier_rank.get(str(event.get("source_tier") or "web"), 9),
        date_rank,
        str(event.get("title") or ""),
    )


def _parse_concert_date(text: str) -> datetime | None:
    value = str(text or "").strip()
    if not value:
        return None
    normalized = value.replace(".", "-").replace("/", "-")
    if re.match(r"20\d{2}-\d{1,2}-\d{1,2}$", normalized):
        try:
            return datetime.strptime(normalized, "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError:
            return None
    return None
