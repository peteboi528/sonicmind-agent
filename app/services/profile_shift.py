from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any


def detect_taste_shift(
    *,
    agent: Any,
    user_id: str,
    recent_days: int = 30,
    baseline_days: int = 90,
) -> dict[str, Any]:
    memory = agent.memory.get_memory(user_id)
    events = list(memory.listening_history or [])
    if not events:
        return {
            "type": "taste_shift_detector",
            "recent_profile": {},
            "baseline_profile": {},
            "shift_signals": [],
            "emerging_artists": [],
            "emerging_genres": [],
            "emerging_moods": [],
            "message": "还没有足够的听歌历史来分析口味迁移。",
        }
    asset_map = {asset.asset_id: asset for asset in agent.list_assets()}
    now = datetime.now(UTC)
    recent_cutoff = now - timedelta(days=recent_days)
    baseline_cutoff = recent_cutoff - timedelta(days=baseline_days)

    recent_assets: list[Any] = []
    baseline_assets: list[Any] = []
    for event in events:
        asset = asset_map.get(getattr(event, "asset_id", ""))
        if asset is None:
            continue
        ts = _parse_iso(getattr(event, "timestamp", ""))
        if ts is None:
            baseline_assets.append(asset)
            continue
        if ts >= recent_cutoff:
            recent_assets.append(asset)
        elif ts >= baseline_cutoff:
            baseline_assets.append(asset)
    if not recent_assets:
        recent_assets = baseline_assets[-10:]
    if not baseline_assets:
        baseline_assets = [
            asset_map.get(getattr(event, "asset_id", "")) for event in events[: -len(recent_assets) or None]
        ]
        baseline_assets = [asset for asset in baseline_assets if asset is not None]

    recent_profile = _profile_snapshot(recent_assets)
    baseline_profile = _profile_snapshot(baseline_assets)
    shift_signals = _diff_profiles(recent_profile, baseline_profile)
    return {
        "type": "taste_shift_detector",
        "recent_profile": recent_profile,
        "baseline_profile": baseline_profile,
        "shift_signals": shift_signals,
        "emerging_artists": [
            name
            for name, _ in recent_profile.get("top_artists", [])
            if name not in {x for x, _ in baseline_profile.get("top_artists", [])}
        ][:5],
        "emerging_genres": [
            name
            for name, _ in recent_profile.get("top_genres", [])
            if name not in {x for x, _ in baseline_profile.get("top_genres", [])}
        ][:5],
        "emerging_moods": [
            name
            for name, _ in recent_profile.get("top_moods", [])
            if name not in {x for x, _ in baseline_profile.get("top_moods", [])}
        ][:5],
        "message": "",
    }


def _parse_iso(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
    except Exception:
        return None


def _profile_snapshot(assets: list[Any]) -> dict[str, Any]:
    genre_counts = Counter()
    mood_counts = Counter()
    artist_counts = Counter()
    for asset in assets:
        artist = str(getattr(asset, "artist", "") or "").strip()
        if artist:
            artist_counts[artist] += 1
        for genre in getattr(asset, "genre", []) or []:
            if str(genre).strip():
                genre_counts[str(genre)] += 1
        for mood in getattr(asset, "mood", []) or []:
            if str(mood).strip():
                mood_counts[str(mood)] += 1
    return {
        "event_count": len(assets),
        "top_genres": genre_counts.most_common(5),
        "top_moods": mood_counts.most_common(5),
        "top_artists": artist_counts.most_common(5),
    }


def _diff_profiles(recent: dict[str, Any], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for key in ("top_genres", "top_moods", "top_artists"):
        recent_map = dict(recent.get(key) or [])
        baseline_map = dict(baseline.get(key) or [])
        for name, count in sorted(recent_map.items(), key=lambda item: -item[1]):
            diff = count - baseline_map.get(name, 0)
            if diff <= 0:
                continue
            signals.append(
                {
                    "dimension": key.removeprefix("top_"),
                    "name": name,
                    "direction": "up",
                    "recent_count": count,
                    "baseline_count": baseline_map.get(name, 0),
                    "delta": diff,
                }
            )
    return signals[:8]
