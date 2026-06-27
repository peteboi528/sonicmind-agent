from __future__ import annotations

import re
from typing import Any

from app.models import ExternalTrack, TasteExperiment, TasteExperimentTrack, utc_now_iso


def filter_taste_experiment_candidates(
    *,
    library: Any,
    user_id: str,
    candidates: list[tuple[Any, dict[str, float], str, float]],
    exclusion_rules: list[str],
    is_quality_track: Any,
) -> list[tuple[Any, dict[str, float], str, float]]:
    seen: set[str] = set()
    artist_counts: dict[str, int] = {}
    filtered: list[tuple[Any, dict[str, float], str, float]] = []
    rules = [rule.lower().strip() for rule in exclusion_rules if rule.strip()]
    for track, components, reason, score in candidates:
        if library.is_disliked(user_id, track):
            continue
        if not is_quality_track(track):
            continue
        searchable = " ".join([
            getattr(track, "title", "") or "",
            getattr(track, "artist", "") or "",
            " ".join(getattr(track, "genre", []) or []),
            " ".join(getattr(track, "mood", []) or []),
        ]).lower()
        if any(rule and rule in searchable for rule in rules):
            continue
        title = (getattr(track, "title", "") or "").strip().lower()
        artist = (getattr(track, "artist", "") or "").strip().lower()
        title_artist_key = f"title:{title}:{artist}"
        base_title = re.sub(r"\s*[\[(（].*?[\])）]", "", title).strip()
        base_title_artist_key = f"title:{base_title}:{artist}"
        external_id = getattr(track, "external_id", "") or getattr(track, "source_id", "") or ""
        external_key = f"external:{external_id}" if external_id else ""
        if title_artist_key in seen or base_title_artist_key in seen or (external_key and external_key in seen):
            continue
        primary_artist = re.split(r"[、,/&]| feat\\.? | ft\\.? ", artist, maxsplit=1)[0].strip() or artist
        if primary_artist and artist_counts.get(primary_artist, 0) >= 4:
            continue
        seen.add(title_artist_key)
        seen.add(base_title_artist_key)
        if external_key:
            seen.add(external_key)
        if primary_artist:
            artist_counts[primary_artist] = artist_counts.get(primary_artist, 0) + 1
        filtered.append((track, components or {}, reason, float(score or 0.0)))
    return filtered


def taste_familiarity(item: tuple[Any, dict[str, float], str, float]) -> float:
    _, components, _, _ = item
    per = components.get("personalize", 0.0)
    sem = components.get("semantic", 0.0)
    beh = components.get("behavior", 0.0)
    return per * 0.6 + sem * 0.3 + beh * 0.1


def slice_for_bucket(
    ranked: list[tuple[Any, dict[str, float], str, float]],
    bucket: str,
    per_bucket: int,
) -> list[tuple[Any, dict[str, float], str, float]]:
    if bucket == "safe":
        return ranked[0:per_bucket]
    if bucket == "stretch":
        return ranked[per_bucket:2 * per_bucket]
    return ranked[2 * per_bucket:3 * per_bucket]


def bucket_taste_experiment_candidates(
    candidates: list[tuple[Any, dict[str, float], str, float]],
    per_bucket: int,
) -> dict[str, list[tuple[Any, dict[str, float], str, float]]]:
    buckets: dict[str, list[tuple[Any, dict[str, float], str, float]]] = {"safe": [], "stretch": [], "bold": []}
    if not candidates:
        return buckets
    ranked = sorted(candidates, key=taste_familiarity, reverse=True)
    familiarities = [taste_familiarity(item) for item in ranked]
    if len(ranked) < per_bucket * 3 or max(familiarities) - min(familiarities) < 0.08:
        buckets["stretch"] = ranked[:per_bucket * 3]
        return buckets
    buckets["safe"] = slice_for_bucket(ranked, "safe", per_bucket)
    buckets["stretch"] = slice_for_bucket(ranked, "stretch", per_bucket)
    buckets["bold"] = slice_for_bucket(ranked, "bold", per_bucket)
    return buckets


def candidate_key(item: tuple[Any, dict[str, float], str, float]) -> str:
    track, _, _, _ = item
    source = getattr(track, "source", "netease") or "netease"
    external_id = getattr(track, "external_id", "") or getattr(track, "source_id", "") or ""
    if external_id:
        return f"{source}:{external_id}"
    title = (getattr(track, "title", "") or "").strip().lower()
    artist = (getattr(track, "artist", "") or "").strip().lower()
    return f"title:{title}:{artist}"


def taste_experiment_track_key(item: TasteExperimentTrack) -> str:
    source_id = item.track.source_id.strip()
    if source_id:
        return f"{item.track.source}:{source_id}"
    return f"title:{item.track.title.lower()}:{item.track.artist.lower()}"


def find_taste_experiment_track(experiment: TasteExperiment, track_key: str) -> TasteExperimentTrack | None:
    for segment in experiment.segments:
        for item in segment.tracks:
            if taste_experiment_track_key(item) == track_key:
                return item
    return None


def apply_taste_experiment_ts_feedback(
    *,
    library: Any,
    item: TasteExperimentTrack,
    signal: str,
    score: float | None,
) -> None:
    if not item.track.source_id:
        return
    positive = signal in {"completed", "liked", "saved"} or (signal == "rated" and (score or 0) >= 7)
    negative = signal in {"skipped", "disliked"} or (signal == "rated" and (score or 10) <= 4)
    if not positive and not negative:
        return
    track = ExternalTrack(
        external_id=item.track.source_id,
        title=item.track.title,
        artist=item.track.artist or "",
        genre=item.track.genre,
        mood=item.track.mood,
        source=item.track.source,
    )
    library.update_ts_feedback(track, positive=positive, weight=1.0 if positive else 0.6)


def record_taste_experiment_listen(
    *,
    memory: Any,
    user_id: str,
    item: TasteExperimentTrack,
    signal: str,
    score: float | None,
) -> None:
    source_id = (item.track.source_id or "").strip()
    if not source_id:
        return
    if signal in {"completed", "liked", "saved"}:
        completed, duration = True, 180
    elif signal in {"skipped", "disliked", "too_far", "too_safe"}:
        completed, duration = False, 0
    elif signal == "rated":
        if (score or 0) >= 7:
            completed, duration = True, 180
        elif (score or 10) <= 4:
            completed, duration = False, 0
        else:
            return
    else:
        return
    memory.record_listen(user_id, source_id, duration, completed, context=f"taste_lab:{signal}")


def taste_experiment_feedback_count(experiment: TasteExperiment) -> int:
    total = 0
    for segment in experiment.segments:
        for item in segment.tracks:
            fb = item.feedback
            total += fb.completed + fb.skipped + fb.liked + fb.disliked + fb.saved + fb.rated + fb.too_safe + fb.too_far
    return total


def taste_experiment_bucket_stats(experiment: TasteExperiment) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    for segment in experiment.segments:
        total_tracks = len(segment.tracks)
        completed = skipped = liked = disliked = saved = too_safe = too_far = rated = 0
        scores: list[float] = []
        for item in segment.tracks:
            fb = item.feedback
            completed += fb.completed
            skipped += fb.skipped
            liked += fb.liked
            disliked += fb.disliked
            saved += fb.saved
            too_safe += fb.too_safe
            too_far += fb.too_far
            rated += fb.rated
            scores.extend(fb.scores)
        feedback_count = completed + skipped + liked + disliked + saved + too_safe + too_far + rated
        denom = max(feedback_count, 1)
        stats[segment.name] = {
            "tracks": total_tracks,
            "feedback_count": feedback_count,
            "completed": completed,
            "skipped": skipped,
            "liked": liked,
            "disliked": disliked,
            "saved": saved,
            "too_safe": too_safe,
            "too_far": too_far,
            "completed_rate": round(completed / denom, 3),
            "skip_rate": round(skipped / denom, 3),
            "liked_rate": round((liked + saved) / denom, 3),
            "avg_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        }
    return stats


def bucket_label(bucket: str) -> str:
    return {"safe": "安全区", "stretch": "轻微越界", "bold": "大胆探索"}.get(bucket, bucket)
