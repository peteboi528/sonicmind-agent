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
    "feedback", "listening_history", "list_my_playlists", "find_on_platform",
    "lyrics", "audio_features", "save_to_playlist", "favorite_track", "concert_events",
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
        source_id = (
            getattr(track, "external_id", "") or getattr(track, "asset_id", "")
            if track else ""
        )
        agent.record_dislike(DislikeRequest(
            user_id=user_id, title=title, artist=artist,
            source=source, source_id=source_id, reason=reason,
        ))
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
        track_id = (
            getattr(track, "asset_id", "") or getattr(track, "external_id", "")
            if track else ""
        )
        if track_id:
            agent.memory.record_listen(user_id, track_id, 0, False, context="agent_feedback")
            summary = f"已记录播放：{target}。"
        else:
            summary = f"未定位到真实曲目，仅保留了播放意图：{target}。"

    return {
        "type": "feedback", "action": action, "title": title, "artist": artist,
        "resolved": bool(track), "summary": summary,
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
    summary = f"{window} 听歌历史共 {len(events)} 次，按{('歌手' if group_by == 'artist' else '曲目')}汇总 {len(items)} 项。"
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
        if result_tracks else f"未在 {platform} 找到可核实的《{title}》结果。"
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
    summary = f"已获取《{title or (resolved or {}).get('title', song_id)}》歌词 {len(lines)} 行。" if lines else "没有获取到可核实的歌词。"
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
    summary = "已读取曲库中的真实音频特征。" if known else "当前没有经过测量的音频特征；不会用随机值补齐。"
    return {"type": "audio_features", "asset_id": getattr(track, "asset_id", ""), "features": features, "measured": known}, summary


def _preview_account_write(name: str, user_id: str, args: dict[str, Any]):
    confirm = bool(args.get("confirm", False))
    action = {
        "tool": name, "playlist_id": str(args.get("playlist_id", "")),
        "track_ids": [str(item) for item in args.get("track_ids", [])],
        "track_id": str(args.get("track_id", "")),
    }
    if not confirm:
        summary = "这是账号写操作预览；请明确确认后再执行。"
        return {"type": "confirmation_required", "confirmed": False, "action": action}, summary
    from app import netease_auth
    info = netease_auth.load_cookie(user_id)
    if not info or not info.get("cookie"):
        return {"type": "auth_required", "what": "netease_login", "action": action}, "尚未登录网易云，未执行任何写操作。"
    return {
        "type": "unsupported_write", "confirmed": True, "executed": False, "action": action,
    }, "当前版本尚无经过验证的网易云写接口，已安全拒绝，账号未发生变化。"


def _concert_events(agent: AudioVisualAgent, args: dict[str, Any]):
    artist = str(args.get("artist", "")).strip()
    city = str(args.get("city", "")).strip()
    query = " ".join(filter(None, [artist, city, "演出 巡演 官方"])).strip()
    sources = agent.search_artist_info(query)
    summary = f"找到 {len(sources)} 条可追溯的演出信息来源。" if sources else "暂未找到可核实的演出信息。"
    return {"type": "concert_events", "artist": artist, "city": city, "events": sources}, summary
