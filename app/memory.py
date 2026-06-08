from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from app.models import (
    Asset,
    ListeningEvent,
    MemoryEntry,
    MemoryUpdateRequest,
    RatingEntry,
    Segment,
    TasteProfile,
    UserMemory,
    utc_now_iso,
)
from app.recommend.engine import compute_taste_profile
from app.storage import JsonStore


PREFERENCE_PATTERNS = [
    re.compile(r"(?:i\s+)?(?:like|love|prefer)\s+(.+)", re.IGNORECASE),
    re.compile(r"(?:喜欢|偏好|更想要|更喜欢|爱听)(.+)"),
]


class MemoryManager:
    def __init__(self, store: JsonStore) -> None:
        self.store = store

    def get_memory(self, user_id: str) -> UserMemory:
        memory = self.store.read_model("memory", user_id, UserMemory)
        if memory is None:
            memory = UserMemory(user_id=user_id)
        self._migrate_preferences(memory)
        return memory

    def update_memory(self, request: MemoryUpdateRequest) -> tuple[UserMemory, bool]:
        memory = self.get_memory(request.user_id)
        changed = False

        preference = extract_preference(request.event)
        if preference:
            if self._upsert_entry(memory, preference, "user_event"):
                changed = True
            if preference not in memory.preferences:
                memory.preferences.append(preference)

        goal = extract_goal(request.event)
        if goal and goal not in memory.common_goals:
            memory.common_goals.append(goal)
            changed = True

        if request.segment_id and request.segment_id not in memory.confirmed_segments:
            memory.confirmed_segments.append(request.segment_id)
            changed = True

        if request.asset_id:
            note = f"{request.asset_id}: {request.event}"
            if note not in memory.project_notes:
                memory.project_notes.append(note)
                changed = True

        if changed:
            memory.updated_at = utc_now_iso()
            self.store.write_model("memory", request.user_id, memory)
        return memory, changed

    def record_feedback(self, user_id: str, segment: Segment, accepted: bool) -> UserMemory:
        memory = self.get_memory(user_id)
        tags = segment.audio_tags + segment.visual_tags
        for entry in memory.structured_preferences:
            if any(tag in entry.text.lower() for tag in tags):
                if accepted:
                    entry.frequency = min(entry.frequency + 1, 20)
                else:
                    entry.frequency = max(entry.frequency - 1, 1)
                entry.last_used = utc_now_iso()
        memory.updated_at = utc_now_iso()
        self.store.write_model("memory", user_id, memory)
        return memory

    def record_listen(self, user_id: str, asset_id: str, duration: int, completed: bool, context: str | None = None) -> UserMemory:
        memory = self.get_memory(user_id)
        event = ListeningEvent(
            asset_id=asset_id,
            duration_listened=duration,
            completed=completed,
            context=context,
        )
        memory.listening_history.append(event)
        if len(memory.listening_history) > 200:
            memory.listening_history = memory.listening_history[-200:]
        memory.updated_at = utc_now_iso()
        self.store.write_model("memory", user_id, memory)
        return memory

    def record_rating(self, user_id: str, asset: Asset, score: float) -> UserMemory:
        memory = self.get_memory(user_id)
        # 更新或新增评分
        existing = next((r for r in memory.ratings if r.asset_id == asset.asset_id), None)
        if existing:
            existing.score = score
            existing.timestamp = utc_now_iso()
        else:
            memory.ratings.append(RatingEntry(
                asset_id=asset.asset_id,
                score=score,
                title=asset.title,
                artist=asset.artist or "",
                genre=asset.genre,
                mood=asset.mood,
            ))
        memory.updated_at = utc_now_iso()
        self.store.write_model("memory", user_id, memory)
        return memory

    def remove_asset_references(self, asset_id: str, user_id: str | None = None) -> dict[str, int]:
        user_ids = [user_id] if user_id else self.store.list_keys("memory")
        removed = {"ratings": 0, "listening_history": 0, "project_notes": 0, "users": 0}
        note_prefix = f"{asset_id}:"

        for uid in user_ids:
            memory = self.get_memory(uid)
            before_ratings = len(memory.ratings)
            before_history = len(memory.listening_history)
            before_notes = len(memory.project_notes)

            memory.ratings = [r for r in memory.ratings if r.asset_id != asset_id]
            memory.listening_history = [ev for ev in memory.listening_history if ev.asset_id != asset_id]
            memory.project_notes = [note for note in memory.project_notes if not note.startswith(note_prefix)]

            delta_ratings = before_ratings - len(memory.ratings)
            delta_history = before_history - len(memory.listening_history)
            delta_notes = before_notes - len(memory.project_notes)
            if delta_ratings or delta_history or delta_notes:
                memory.updated_at = utc_now_iso()
                self.store.write_model("memory", uid, memory)
                removed["ratings"] += delta_ratings
                removed["listening_history"] += delta_history
                removed["project_notes"] += delta_notes
                removed["users"] += 1

        return removed

    def refresh_taste_profile(self, user_id: str, library: list[Asset]) -> UserMemory:
        memory = self.get_memory(user_id)
        listened_ids = {ev.asset_id for ev in memory.listening_history}
        listened_assets = [a for a in library if a.asset_id in listened_ids]
        if not listened_assets:
            listened_assets = library
        memory.taste_profile = compute_taste_profile(listened_assets, memory.listening_history, memory.ratings)
        memory.updated_at = utc_now_iso()
        self.store.write_model("memory", user_id, memory)
        return memory

    def weighted_query(self, memory: UserMemory) -> str:
        scored = score_entries(memory.structured_preferences)
        parts: list[str] = []
        for entry, weight in scored[:8]:
            repeat = max(1, int(weight * 2))
            parts.extend([entry.text] * repeat)
        parts.extend(memory.common_goals[-3:])
        return " ".join(parts)

    def _upsert_entry(self, memory: UserMemory, text: str, source: str) -> bool:
        normalized = text.lower().strip()
        for entry in memory.structured_preferences:
            if entry.text.lower().strip() == normalized:
                entry.frequency += 1
                entry.last_used = utc_now_iso()
                return True
        memory.structured_preferences.append(
            MemoryEntry(text=text, frequency=1, source=source)
        )
        return True

    def _migrate_preferences(self, memory: UserMemory) -> None:
        if memory.preferences and not memory.structured_preferences:
            for pref in memory.preferences:
                memory.structured_preferences.append(
                    MemoryEntry(text=pref, frequency=1, source="migrated")
                )


def score_entries(entries: list[MemoryEntry]) -> list[tuple[MemoryEntry, float]]:
    now = datetime.now(timezone.utc)
    scored: list[tuple[MemoryEntry, float]] = []
    for entry in entries:
        try:
            last = datetime.fromisoformat(entry.last_used)
            age_days = max(0, (now - last).days)
        except (ValueError, TypeError):
            age_days = 0
        decay = math.exp(-0.05 * age_days)
        weight = entry.frequency * decay
        scored.append((entry, weight))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def extract_preference(event: str) -> str | None:
    normalized = event.strip()
    for pattern in PREFERENCE_PATTERNS:
        match = pattern.search(normalized)
        if match:
            value = cleanup(match.group(1))
            if len(value) >= 3:
                return value
    return None


def extract_goal(event: str) -> str | None:
    lowered = event.lower()
    if any(word in lowered for word in ["trailer", "预告片", "宣传片", "recommend", "推荐"]):
        return "trailer_or_promo_selection"
    if any(word in lowered for word in ["summary", "report", "摘要", "报告"]):
        return "content_analysis_report"
    return None


def cleanup(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip("。.!? ，,")).strip()
