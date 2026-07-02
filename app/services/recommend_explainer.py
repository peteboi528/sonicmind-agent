from __future__ import annotations

from collections import Counter
from typing import Any


def build_recommend_explanation(
    *,
    agent: Any,
    user_id: str,
    query: str | None,
    prior_results: list[dict[str, Any]],
) -> dict[str, Any]:
    recommendation = _latest_recommendation(prior_results)
    if recommendation is None:
        return {
            "type": "recommend_explainer",
            "global_reasons": [],
            "per_track_reasons": [],
            "evidence_buckets": {},
            "message": "还没有最近一轮推荐结果；先让我给你推荐一轮，再解释原因。",
            "missing_context": True,
        }
    memory = agent.memory.get_memory(user_id)
    top_genres = [name for name, _ in ((memory.taste_profile.top_genres if memory.taste_profile else []) or [])][:3]
    top_moods = [name for name, _ in ((memory.taste_profile.top_moods if memory.taste_profile else []) or [])][:3]
    global_reasons: list[str] = []
    if top_genres:
        global_reasons.append(f"推荐整体贴近你最近常听的曲风：{'、'.join(top_genres)}。")
    if top_moods:
        global_reasons.append(f"也延续了你偏好的情绪标签：{'、'.join(top_moods)}。")
    if query:
        global_reasons.append(f"当前这一轮还额外对齐了你的即时需求：{query}。")

    evidence = Counter()
    per_track = []
    suggested_tracks = []
    for item in recommendation.tracks:
        track = item.asset
        tags = [*(getattr(track, "genre", []) or []), *(getattr(track, "mood", []) or [])]
        source = str(getattr(track, "source", "") or "unknown")
        reasons = []
        if item.reason:
            reasons.append(item.reason)
        if tags:
            reasons.append("标签匹配：" + " / ".join(str(tag) for tag in tags[:3]))
        if top_genres and any(tag in top_genres for tag in getattr(track, "genre", []) or []):
            evidence["taste_profile"] += 1
        if top_moods and any(tag in top_moods for tag in getattr(track, "mood", []) or []):
            evidence["scene_or_mood"] += 1
        if source != "local":
            evidence["source_trust"] += 1
        if getattr(track, "artist", ""):
            evidence["artist_or_style_similarity"] += 1
        per_track.append({
            "title": track.title,
            "artist": track.artist,
            "source": source,
            "reasons": reasons,
        })
        suggested_tracks.append(track)
    return {
        "type": "recommend_explainer",
        "global_reasons": global_reasons,
        "per_track_reasons": per_track,
        "evidence_buckets": dict(evidence),
        "message": "",
        "missing_context": False,
        "tracks": suggested_tracks,
    }


def _latest_recommendation(prior_results: list[dict[str, Any]]) -> Any | None:
    for result in reversed(prior_results or []):
        if result.get("type") == "daily_recommend":
            return result.get("recommendation")
    return None
