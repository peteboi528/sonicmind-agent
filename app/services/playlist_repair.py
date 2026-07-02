from __future__ import annotations

from collections import Counter
from typing import Any

from app.answer import collect_tracks
from app.recommend.hygiene import is_valid_music_track
from app.recommend.rerank import detect_language


def analyze_playlist_repair(
    *,
    agent: Any,
    user_id: str,
    query: str,
    instruction: str | None,
    target: str | None,
    prior_results: list[dict[str, Any]],
) -> dict[str, Any]:
    tracks = _resolve_target_tracks(prior_results)
    if not tracks:
        return {
            "type": "playlist_repair",
            "target": target or "",
            "issues": [],
            "repair_actions": [],
            "suggested_replacements": [],
            "missing_context": True,
            "message": "缺少待修复的歌单或候选结果；请先给我一轮推荐或歌单。",
        }

    issues: list[dict[str, Any]] = []
    existing_keys = {_track_key(track) for track in tracks}
    search_text = " ".join(part for part in [instruction, target, query] if part).strip()

    duplicates = _duplicate_titles(tracks)
    if duplicates:
        issues.append({
            "kind": "duplicate_tracks",
            "severity": "high",
            "summary": f"存在 {len(duplicates)} 组重复曲目。",
            "items": duplicates,
        })

    invalid = [track for track in tracks if not is_valid_music_track(track)]
    if invalid:
        issues.append({
            "kind": "invalid_tracks",
            "severity": "high",
            "summary": f"发现 {len(invalid)} 首像教程/合集/节目而非真实歌曲的条目。",
            "items": [_track_brief(track) for track in invalid],
        })

    style_jumps = _style_jumps(tracks)
    if style_jumps:
        issues.append({
            "kind": "style_jump",
            "severity": "medium",
            "summary": f"有 {len(style_jumps)} 处相邻歌曲风格跨度过大。",
            "items": style_jumps,
        })

    energy_gaps = _energy_gaps(tracks)
    if energy_gaps:
        issues.append({
            "kind": "energy_gap",
            "severity": "medium",
            "summary": f"有 {len(energy_gaps)} 处能量断层，听感可能不连贯。",
            "items": energy_gaps,
        })

    language_mix = _language_mix(tracks)
    if language_mix:
        issues.append(language_mix)

    target_mismatch = _target_mismatch(tracks, search_text)
    if target_mismatch:
        issues.append(target_mismatch)

    repair_actions = _repair_actions(issues)
    suggested = _suggest_replacements(agent, user_id, search_text or query, existing_keys)
    return {
        "type": "playlist_repair",
        "target": target or search_text,
        "issues": issues,
        "repair_actions": repair_actions,
        "suggested_replacements": suggested,
        "missing_context": False,
        "message": "",
    }


def _resolve_target_tracks(prior_results: list[dict[str, Any]]) -> list[Any]:
    for result in reversed(prior_results or []):
        if result.get("type") == "playlist":
            return list(result.get("playlist").tracks or [])
        if result.get("type") == "daily_recommend":
            recommendation = result.get("recommendation")
            if recommendation is not None and hasattr(recommendation, "tracks"):
                return [item.asset for item in recommendation.tracks]
        if result.get("type") == "journey":
            tracks = []
            for phase in (result.get("journey") or {}).get("phases", []):
                tracks.extend(collect_tracks([{"type": "phase_tracks", "tracks": phase.get("tracks", [])}]))
            if tracks:
                return tracks
    return collect_tracks(prior_results or [])


def _track_key(track: Any) -> tuple[str, str]:
    return (
        str(getattr(track, "title", "") or "").strip().lower(),
        str(getattr(track, "artist", "") or "").strip().lower(),
    )


def _track_brief(track: Any) -> dict[str, str]:
    return {
        "title": str(getattr(track, "title", "") or ""),
        "artist": str(getattr(track, "artist", "") or ""),
        "source": str(getattr(track, "source", "") or ""),
    }


def _duplicate_titles(tracks: list[Any]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[Any]] = {}
    for track in tracks:
        key = _track_key(track)
        buckets.setdefault(key, []).append(track)
    duplicates = []
    for (title, artist), items in buckets.items():
        if title and len(items) > 1:
            duplicates.append({
                "title": title,
                "artist": artist,
                "count": len(items),
            })
    return duplicates


def _style_jumps(tracks: list[Any]) -> list[dict[str, Any]]:
    jumps: list[dict[str, Any]] = []
    for left, right in zip(tracks, tracks[1:], strict=False):
        left_tags = {str(item).lower() for item in [*(getattr(left, "genre", []) or []), *(getattr(left, "mood", []) or [])] if str(item).strip()}
        right_tags = {str(item).lower() for item in [*(getattr(right, "genre", []) or []), *(getattr(right, "mood", []) or [])] if str(item).strip()}
        if left_tags and right_tags and not left_tags.intersection(right_tags):
            jumps.append({
                "from": _track_brief(left),
                "to": _track_brief(right),
                "reason": "相邻歌曲没有共享的曲风或情绪标签。",
            })
    return jumps[:5]


_ENERGY_HIGH = {"跑步", "热血", "高能", "爆发", "冲刺", "舞曲", "硬核", "激情"}
_ENERGY_LOW = {"放松", "治愈", "chill", "深夜", "舒缓", "安静", "慵懒", "ambient"}


def _energy_bucket(track: Any) -> int:
    tags = {str(item).lower() for item in [*(getattr(track, "genre", []) or []), *(getattr(track, "mood", []) or [])]}
    if any(tag in _ENERGY_HIGH for tag in tags):
        return 2
    if any(tag in _ENERGY_LOW for tag in tags):
        return 0
    return 1


def _energy_gaps(tracks: list[Any]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for left, right in zip(tracks, tracks[1:], strict=False):
        if abs(_energy_bucket(left) - _energy_bucket(right)) >= 2:
            gaps.append({
                "from": _track_brief(left),
                "to": _track_brief(right),
                "reason": "能量级别从高到低或从低到高跳变过大。",
            })
    return gaps[:5]


def _language_mix(tracks: list[Any]) -> dict[str, Any] | None:
    counts = Counter(detect_language(track) for track in tracks if detect_language(track) in {"zh", "en"})
    if len(counts) < 2:
        return None
    total = sum(counts.values()) or 1
    dominant, dominant_count = counts.most_common(1)[0]
    minority = total - dominant_count
    if minority / total < 0.25:
        return None
    return {
        "kind": "language_mix",
        "severity": "medium",
        "summary": "歌单里中英文混杂比例较高，可能破坏统一听感。",
        "items": [{"language": lang, "count": count} for lang, count in counts.items()],
        "dominant_language": dominant,
    }


def _target_mismatch(tracks: list[Any], query: str) -> dict[str, Any] | None:
    lowered = (query or "").lower()
    if not lowered:
        return None
    tags = Counter(
        str(item).lower()
        for track in tracks
        for item in [*(getattr(track, "genre", []) or []), *(getattr(track, "mood", []) or [])]
        if str(item).strip()
    )
    if any(token in lowered for token in ("跑步", "冲刺", "高能")):
        energetic = sum(count for tag, count in tags.items() if tag in _ENERGY_HIGH)
        if energetic < max(1, len(tracks) // 3):
            return {
                "kind": "target_mismatch",
                "severity": "medium",
                "summary": "目标是高能/跑步，但当前候选里高能标签占比偏低。",
                "items": [{"target": "run_high_energy", "matched": energetic, "total": len(tracks)}],
            }
    if any(token in lowered for token in ("放松", "深夜", "chill", "治愈")):
        calming = sum(count for tag, count in tags.items() if tag in _ENERGY_LOW)
        if calming < max(1, len(tracks) // 3):
            return {
                "kind": "target_mismatch",
                "severity": "medium",
                "summary": "目标是放松/深夜，但当前候选里相关情绪标签占比偏低。",
                "items": [{"target": "calm_night", "matched": calming, "total": len(tracks)}],
            }
    return None


def _repair_actions(issues: list[dict[str, Any]]) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    for issue in issues:
        kind = issue.get("kind")
        if kind == "duplicate_tracks":
            actions.append({"action": "dedupe", "reason": "先删重复曲目，每组只保留一首。"})
        elif kind == "invalid_tracks":
            actions.append({"action": "remove_invalid", "reason": "移除教程/合集/节目类非歌曲条目。"})
        elif kind == "style_jump":
            actions.append({"action": "regroup_by_style", "reason": "按相近曲风或情绪重新排序，减少突兀切换。"})
        elif kind == "energy_gap":
            actions.append({"action": "smooth_energy_curve", "reason": "在高低能量曲之间插入过渡曲。"})
        elif kind == "language_mix":
            actions.append({"action": "normalize_language", "reason": "统一语言或分拆成双语版本歌单。"})
        elif kind == "target_mismatch":
            actions.append({"action": "reseed_with_goal", "reason": "按用户目标重新补召回候选。"})
    return actions


def _suggest_replacements(agent: Any, user_id: str, query: str, existing_keys: set[tuple[str, str]]) -> list[Any]:
    if not query:
        return []
    try:
        response = agent.search(user_id, query, include_external=False, top_k=8)
    except Exception:
        return []
    candidates = []
    for track in list(getattr(response, "local", []) or []):
        if _track_key(track) in existing_keys:
            continue
        candidates.append(track)
        if len(candidates) >= 5:
            break
    return candidates


# is_valid_music_track 及其黑名单常量已统一到 app.recommend.hygiene（与 is_structural_reject/
# classify_candidate 同源，单一事实来源）。此处 import 复用，下方 analyze_playlist_repair 直接调。
