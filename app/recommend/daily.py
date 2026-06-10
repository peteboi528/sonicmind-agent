from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from app.config import settings
from app.llm.protocol import LLMProvider
from app.llm.structured import parse_json_list_safe
from app.models import (
    Asset,
    DailyRecommendation,
    ExternalTrack,
    RagEvidence,
    RecommendedTrack,
    TasteProfile,
    UserMemory,
)
from app.prompts import DAILY_RECOMMEND_USER_TEMPLATE, DAILY_SUMMARY_TEMPLATE
from app.recommend.engine import RecommendEngine
from app.sources.protocol import ExternalSource


logger = logging.getLogger(__name__)


class _LLMTrackItem(BaseModel):
    title: str = ""
    artist: str = ""
    genre: str = "流行"
    mood: str = "放松"
    reason: str = ""


TIME_MOODS: dict[str, list[str]] = {
    "morning": ["欢快", "治愈", "放松"],
    "focus": ["宁静", "放松", "梦幻"],
    "afternoon": ["欢快", "激昂", "浪漫"],
    "evening": ["热血", "激昂", "欢快"],
    "night": ["宁静", "放松", "治愈", "梦幻"],
}


def get_time_bucket(hour: int | None = None) -> str:
    if hour is None:
        hour = datetime.now(timezone.utc).hour + 8
        hour = hour % 24
    if 6 <= hour < 10:
        return "morning"
    if 10 <= hour < 14:
        return "focus"
    if 14 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 22:
        return "evening"
    return "night"


class DailyRecommender:
    def __init__(self, engine: RecommendEngine, source: ExternalSource, llm: LLMProvider) -> None:
        self.engine = engine
        self.source = source
        self.llm = llm

    def generate(
        self,
        user: UserMemory,
        library: list[Asset],
        time_of_day: str | None = None,
        count: int | None = None,
        evidences: list[RagEvidence] | None = None,
        trace: list[str] | None = None,
    ) -> DailyRecommendation:
        count = count or settings.daily_rec_count
        taste = user.taste_profile or TasteProfile()
        bucket = time_of_day or get_time_bucket()

        # 优先用真实外部源推荐（不编造歌曲），失败再用 LLM
        tracks = self._fallback_recommend(user, library, taste, bucket, count)

        if not tracks:
            # 兜底：用 LLM 生成（标记 source="llm"，用户可见这是未核实的）
            library_desc = self._describe_library(library)
            prefs_desc = ", ".join(user.preferences[-5:]) if user.preferences else "暂无明确偏好"
            taste_desc = self._describe_taste(taste)
            prompt = DAILY_RECOMMEND_USER_TEMPLATE(
                count=count,
                prefs_desc=prefs_desc,
                taste_desc=taste_desc,
                library_desc=library_desc,
                bucket=bucket,
            )
            tracks = self._llm_recommend(prompt, count, bucket, openness=taste.discovery_openness)

        tracks = [self._with_reason(track, user, library, taste) for track in tracks]

        summary = self._generate_summary(tracks, bucket, taste)

        return DailyRecommendation(
            user_id=user.user_id,
            tracks=tracks,
            reason_summary=summary,
            evidences=evidences or [],
            agent_trace=trace or [],
        )

    def _describe_library(self, library: list[Asset]) -> str:
        if not library:
            return "空"
        items = []
        for a in library[:10]:
            artist = a.artist or "未知"
            genre = ", ".join(a.genre) if a.genre else "未知"
            items.append(f"{a.title} - {artist} ({genre})")
        return "; ".join(items)

    def _describe_taste(self, taste: TasteProfile) -> str:
        genres = [g for g, _ in taste.top_genres[:5]]
        moods = [m for m, _ in taste.top_moods[:5]]
        if not genres and not moods:
            return "暂无数据"
        return f"风格偏好: {genres}, 情绪偏好: {moods}, 能量偏好: {taste.preferred_energy}"

    def _llm_recommend(self, prompt: str, count: int, bucket: str, openness: float = 0.3) -> list[RecommendedTrack]:
        try:
            result = self.llm.generate(prompt)
            items = parse_json_list_safe(result, _LLMTrackItem)
            tracks: list[RecommendedTrack] = []
            # Phase 2：explore/exploit 配比由 discovery_openness 驱动。
            # openness=0.3 → 前 70% familiar（利用），后 30% discovery（探索）。
            familiar_cutoff = max(1, round(count * (1 - openness)))
            for i, item in enumerate(items[:count]):
                track = ExternalTrack(
                    external_id=f"llm-rec-{i:03d}",
                    title=item.title or "未知",
                    artist=item.artist or "未知",
                    genre=[item.genre],
                    mood=[item.mood],
                    # source="llm" 标记未核实：这些曲目由 LLM 生成、未经真实回查。
                    source="llm",
                )
                cat = "familiar" if i < familiar_cutoff else "discovery"
                tracks.append(RecommendedTrack(
                    asset=track,
                    score=round(1.0 - i * 0.03, 4),
                    reason=item.reason or f"{item.genre}风格推荐",
                    category=cat,
                ))
            return tracks
        except Exception:
            logger.debug("LLM recommendation generation failed; using fallback recommender", exc_info=True)
            return []

    def _fallback_recommend(self, user: UserMemory, library: list[Asset], taste: TasteProfile, bucket: str, count: int) -> list[RecommendedTrack]:
        moods = TIME_MOODS.get(bucket, TIME_MOODS["afternoon"])
        seed_genres = [g for g, _ in taste.top_genres[:3]] or ["流行"]
        external = self.source.get_recommendations(seed_genres, moods[:2], limit=count)
        recent_ids = {ev.asset_id for ev in user.listening_history[-50:]}
        # Phase 1：把真实收听行为回灌成排序奖励（Spotify BaRT 思想）
        from app.memory import compute_behavior_scores
        asset_durations = {a.asset_id: a.duration_seconds for a in library}
        behavior_scores = compute_behavior_scores(user.listening_history, asset_durations)
        ranked = self.engine.rank_tracks(external, taste, moods, recent_ids, behavior_scores)
        tracks: list[RecommendedTrack] = []
        for track, score in ranked[:count]:
            tracks.append(RecommendedTrack(
                asset=track, score=round(score, 4),
                reason=f"{track.genre[0] if track.genre else '流行'}风格推荐",
                category="discovery",
            ))
        return tracks

    def _generate_summary(self, tracks: list[RecommendedTrack], bucket: str, taste: TasteProfile) -> str:
        if not tracks:
            return "暂无推荐"
        genres = set()
        for t in tracks[:10]:
            genres.update(t.asset.genre if hasattr(t.asset, "genre") else [])
        genre_str = "、".join(list(genres)[:3]) or "多种风格"
        try:
            return self.llm.generate(
                DAILY_SUMMARY_TEMPLATE(count=len(tracks), genre_str=genre_str, bucket=bucket)
            )
        except Exception:
            logger.debug("Daily summary generation failed; using template fallback", exc_info=True)
            return f"今日为你精选{len(tracks)}首{genre_str}音乐，适合{bucket}聆听。"

    def _with_reason(
        self,
        track: RecommendedTrack,
        user: UserMemory,
        library: list[Asset],
        taste: TasteProfile,
    ) -> RecommendedTrack:
        asset = track.asset
        track_genres = set(getattr(asset, "genre", []) or [])
        track_moods = set(getattr(asset, "mood", []) or [])

        memory_pref = next(
            (
                pref
                for pref in reversed(user.preferences)
                if any(token in pref for token in list(track_genres) + list(track_moods))
            ),
            None,
        )
        if memory_pref:
            reason = f"延续你明确表达的偏好：{memory_pref}"
            return track.model_copy(update={"reason": reason})

        top_genres = [genre for genre, _ in taste.top_genres[:3]]
        top_moods = [mood for mood, _ in taste.top_moods[:3]]
        genre_hit = next((genre for genre in top_genres if genre in track_genres), None)
        mood_hit = next((mood for mood in top_moods if mood in track_moods), None)
        if genre_hit or mood_hit:
            pieces = []
            if genre_hit:
                pieces.append(f"{genre_hit}风格")
            if mood_hit:
                pieces.append(f"{mood_hit}情绪")
            reason = "匹配你的品味档案：" + "、".join(pieces)
            return track.model_copy(update={"reason": reason})

        similar_asset = next(
            (
                lib
                for lib in library
                if set(lib.genre) & track_genres or set(lib.mood) & track_moods
            ),
            None,
        )
        if similar_asset is not None:
            reason = f"与你库里的《{similar_asset.title}》在风格或情绪上相近"
            return track.model_copy(update={"reason": reason})

        fallback = track.reason or f"{next(iter(track_genres), '当前')}方向的补充推荐"
        return track.model_copy(update={"reason": fallback})
