from __future__ import annotations

import hashlib
import logging
import re
from datetime import UTC, datetime
from typing import Any

from app.concurrency import run_parallel
from app.models import (
    Asset,
    ExternalTrack,
    TasteExperiment,
    TasteExperimentFeedbackRequest,
    TasteExperimentReport,
    TasteExperimentSegment,
    TasteExperimentTrack,
    TasteProfile,
    TrackRef,
    UserMemory,
    utc_now_iso,
)
from app.rules.content import filter_content_exclusions
from app.rules.taste_experiment import (
    exploration_artist_anchors,
    exploration_artist_match,
    exploration_direction,
    exploration_matches,
    exploration_track_keys,
    is_default_exploration,
)

logger = logging.getLogger(__name__)


class TasteExperimentService:
    def __init__(
        self,
        *,
        store: Any,
        memory: Any,
        library: Any,
        recommend_for_query: Any,
        search_web_music: Any,
        rerank_tracks: Any,
        dedupe_tracks: Any,
        list_assets: Any,
        is_recommendation_quality_track: Any,
    ) -> None:
        self.store = store
        self.memory = memory
        self.library = library
        self._recommend_for_query = recommend_for_query
        self._search_web_music = search_web_music
        self._rerank_tracks = rerank_tracks
        self._dedupe_tracks = dedupe_tracks
        self._list_assets = list_assets
        self._is_recommendation_quality_track = is_recommendation_quality_track

    def generate_taste_experiment(
        self,
        user_id: str,
        prompt: str,
        total: int = 12,
        *,
        online_only: bool = False,
        taste_experiment_hypothesis: Any,
        taste_experiment_search_seeds: Any,
        collect_taste_candidates: Any,
        taste_prompt_exclusions: Any,
        filter_taste_experiment_candidates: Any,
        bucket_taste_experiment_candidates: Any,
        taste_experiment_track: Any,
        new_taste_experiment_id: Any,
        save_taste_experiment: Any,
    ) -> TasteExperiment:
        total = max(3, min(total or 12, 30))
        per_bucket = max(1, total // 3)
        memory = self.memory.get_memory(user_id)
        hypothesis = taste_experiment_hypothesis(memory)
        seeds = taste_experiment_search_seeds(memory, prompt)
        if online_only:
            recent = [
                item
                for experiment in self.list_taste_experiments(user_id)[-3:]
                if experiment.prompt.strip().casefold() == prompt.strip().casefold()
                for item in [
                    *experiment.previous_tracks,
                    *(item.track for segment in experiment.segments for item in segment.tracks),
                ]
            ]
            candidates = collect_taste_candidates(
                user_id, seeds, total, online_only=True, prompt=prompt, excluded_tracks=recent
            )
        else:
            candidates = collect_taste_candidates(user_id, seeds, total)
        prompt_rules = taste_prompt_exclusions(prompt)
        candidates = filter_taste_experiment_candidates(
            user_id,
            candidates,
            [*memory.exclusion_rules, *prompt_rules],
        )
        buckets = bucket_taste_experiment_candidates(candidates, per_bucket)
        if online_only and (exploration_artist_anchors(exploration_direction(prompt)) or not memory.listening_history):
            buckets = {"safe": [], "stretch": candidates[:total], "bold": []}
        confidence_ok = bool(buckets["safe"] and buckets["bold"])
        segments = [
            TasteExperimentSegment(
                name="safe",
                label="安全区",
                description=(
                    "命中你的核心风格或艺人，用来验证稳定偏好。"
                    if confidence_ok
                    else "本轮证据不足，暂不强行标记安全区。"
                ),
                tracks=[
                    taste_experiment_track(track, "safe", components, reason, score)
                    for track, components, reason, score in buckets["safe"]
                ],
            ),
            TasteExperimentSegment(
                name="stretch",
                label="轻微越界",
                description=(
                    "和你的画像相邻，但至少在一个维度上有所变化。" if confidence_ok else "先试听这批符合方向的候选。"
                ),
                tracks=[
                    taste_experiment_track(track, "stretch", components, reason, score)
                    for track, components, reason, score in buckets["stretch"]
                ],
            ),
            TasteExperimentSegment(
                name="bold",
                label="大胆探索",
                description=(
                    "有可解释连接点，同时明显超出你的主画像。"
                    if confidence_ok
                    else "本轮证据不足，暂不强行标记大胆探索。"
                ),
                tracks=[
                    taste_experiment_track(track, "bold", components, reason, score)
                    for track, components, reason, score in buckets["bold"]
                ],
            ),
        ]
        actual_total = sum(len(segment.tracks) for segment in segments)
        if not confidence_ok and actual_total:
            shortfall = f"找到 {actual_total} 首，先集中试听。"
        else:
            shortfall = "" if actual_total >= total else f"符合条件的新歌只有 {actual_total}/{total} 首。"
        if online_only and not actual_total:
            shortfall = "这一轮没有找到符合条件且未推荐过的曲目，可以缩短描述或换个方向。"
        experiment = TasteExperiment(
            experiment_id=new_taste_experiment_id(user_id, prompt),
            user_id=user_id,
            prompt=prompt,
            online_only=online_only,
            hypothesis=hypothesis,
            segments=segments,
            result_summary=shortfall,
        )
        save_taste_experiment(experiment)
        return experiment

    def collect_taste_candidates(
        self,
        user_id: str,
        seeds: list[str],
        total: int,
        *,
        online_only: bool = False,
        prompt: str = "",
        excluded_tracks: list[TrackRef] | None = None,
    ) -> list[tuple[Any, dict[str, float], str, float]]:
        """收集候选曲目。

        online_only=True（探索页用）：跳过库内推荐路径，只走 web 搜索拉库外新歌/新
        歌手，并剔除本地曲与已在库/已听过的曲目——否则本地曲 personalize 分天然偏高，
        三档会被库内歌占满，探索失去意义。默认 False 保持品味实验旧行为。
        """
        raw_tracks: list[Asset | ExternalTrack] = []
        if seeds and not online_only:
            try:
                rec = self._recommend_for_query(user_id, seeds[0], top_k=max(total * 3, 18))
                for item in rec.tracks:
                    raw_tracks.append(item.asset)
            except Exception:
                logger.debug("taste_experiment recommend failed for %s", seeds[0], exc_info=True)
        direction = exploration_direction(prompt) if online_only else []
        anchors = exploration_artist_anchors(direction)
        goal = prompt.strip() if online_only and prompt else " ".join(seeds[:6])
        if online_only:
            if prompt and not is_default_exploration(prompt):
                clean_prompt = re.split(r"[，。,.；;]\s*(?:不要|别|不想听|no\s|without\s)", prompt, maxsplit=1)[0]
                clean_prompt = re.sub(r"^(?:推荐|给我|找|想听|听听|来点)\s*", "", clean_prompt).strip()
                terms = [group[0] for group in direction if group[0] not in {"日语", "韩语", "华语", "英语"}]
                seeds = self.dedupe_seeds(
                    [
                        *(name for aliases in anchors for name in aliases[:1]),
                        clean_prompt,
                        " ".join(group[0] for group in direction),
                        " ".join(terms),
                        " ".join(group[1] for group in direction),
                    ]
                )
            else:
                seeds = seeds[:6] + ["neo soul", "indie pop", "爵士钢琴", "电子氛围"]
            tasks = [
                (seed, lambda seed=seed: self._search_web_music(seed, top_k=max(18, total * 2), catalog_only=True))
                for seed in self.dedupe_seeds(seeds)[:10]
            ]
            for batch in run_parallel(tasks, timeout=8.0, default=[]):
                raw_tracks.extend(batch or [])
        else:
            for search_goal in seeds[:16]:
                try:
                    raw_tracks.extend(self._search_web_music(search_goal, top_k=6, relevance_query=search_goal))
                except Exception:
                    logger.debug("taste_experiment seed search failed for %s", search_goal, exc_info=True)
        raw_tracks = [
            track for track in self._dedupe_tracks(raw_tracks) if self._is_recommendation_quality_track(track)
        ]
        if online_only:
            known = {
                f"id:{event.asset_id}" for event in self.memory.get_memory(user_id).listening_history if event.asset_id
            }
            for track in [*self._list_assets(), *(excluded_tracks or [])]:
                known.update(exploration_track_keys(track))
            raw_tracks = [
                track
                for track in raw_tracks
                if track.source == "netease"
                and str(track.external_id or "").isascii()
                and str(track.external_id or "").isdigit()
                and not exploration_track_keys(track) & known
                and exploration_matches(track, direction, anchored=bool(exploration_artist_match(track, anchors)))
            ]
            raw_tracks = filter_content_exclusions(raw_tracks, self.taste_prompt_exclusions(prompt))
        if not raw_tracks:
            return []
        ranked = self._rerank_tracks(user_id, goal, raw_tracks, top_k=len(raw_tracks))
        if anchors:
            # Representative artists first. Keep each artist from taking over the first screen.
            counts: dict[str, int] = {}
            preferred = []
            others = []
            for track, breakdown in ranked:
                aliases = exploration_artist_match(track, anchors)
                if not aliases:
                    others.append((track, breakdown))
                    continue
                key = aliases[0]
                if counts.get(key, 0) >= 3:
                    continue
                counts[key] = counts.get(key, 0) + 1
                preferred.append((track, breakdown))
            ranked = preferred + others if len(preferred) < total else preferred
        return [(track, breakdown.components, breakdown.reason, breakdown.score) for track, breakdown in ranked]

    def regenerate_taste_experiment_bucket(
        self,
        user_id: str,
        experiment_id: str,
        bucket: str,
        *,
        online_only: bool | None = None,
        taste_experiment_seeds_for_bucket: Any,
        collect_taste_candidates: Any,
        filter_taste_experiment_candidates: Any,
        taste_experiment_track_key: Any,
        candidate_key: Any,
        taste_familiarity: Any,
        slice_for_bucket: Any,
        taste_experiment_track: Any,
    ) -> TasteExperiment:
        if bucket not in {"safe", "stretch", "bold"}:
            raise ValueError(f"unknown bucket: {bucket}")
        with self.store.lock("taste_experiments", user_id):
            experiments = self.store.read_models("taste_experiments", user_id, TasteExperiment)
            exp = next((e for e in experiments if e.experiment_id == experiment_id), None)
            if exp is None:
                raise ValueError("Experiment not found")
            memory = self.memory.get_memory(user_id)
            segment = next((s for s in exp.segments if s.name == bucket), None)
            existing = sum(len(s.tracks) for s in exp.segments)
            per_bucket = len(segment.tracks) if segment and segment.tracks else max(1, (existing // 3) or 1)

            use_online = exp.online_only if online_only is None else online_only
            seeds = taste_experiment_seeds_for_bucket(memory, exp.prompt, bucket)
            if use_online:
                recent = [*exp.previous_tracks, *(item.track for seg in exp.segments for item in seg.tracks)]
                candidates = collect_taste_candidates(
                    user_id, seeds, per_bucket, online_only=True, prompt=exp.prompt, excluded_tracks=recent
                )
            else:
                candidates = collect_taste_candidates(user_id, seeds, per_bucket * 6)
            candidates = filter_taste_experiment_candidates(
                user_id, candidates, [*memory.exclusion_rules, *self.taste_prompt_exclusions(exp.prompt)]
            )
            existing_keys = {taste_experiment_track_key(item) for seg in exp.segments for item in seg.tracks}
            candidates = [c for c in candidates if candidate_key(c) not in existing_keys]
            ranked = sorted(candidates, key=taste_familiarity, reverse=True)
            if use_online:
                if bucket == "stretch" and not any(seg.tracks for seg in exp.segments if seg.name != bucket):
                    band = candidates[:per_bucket]
                else:
                    offset = {"safe": 0, "stretch": 1, "bold": 2}[bucket]
                    start = min(offset * per_bucket, max(0, len(ranked) - per_bucket))
                    band = ranked[start : start + per_bucket]
            else:
                band = slice_for_bucket(ranked, bucket, per_bucket)
            new_tracks = [
                taste_experiment_track(track, bucket, components, reason, score)
                for track, components, reason, score in band
            ]
            seen: set[str] = set()
            deduped: list[TasteExperimentTrack] = []
            for item in new_tracks:
                key = taste_experiment_track_key(item)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(item)
            if segment is not None and deduped:
                if use_online:
                    exp.previous_tracks = [*exp.previous_tracks, *(item.track for item in segment.tracks)][-90:]
                segment.tracks = deduped
                exp.report = None
                exp.status = (
                    "ready"
                    if sum(
                        bool(item.feedback and item.feedback.last_signal) for seg in exp.segments for item in seg.tracks
                    )
                    >= 6
                    else "collecting"
                )
                exp.result_summary = f"本档换了 {len(deduped)} 首新歌。"
            elif use_online:
                exp.result_summary = "这一档暂时没有更多符合条件的新歌，已保留当前曲目。"
            exp.online_only = use_online
            exp.updated_at = utc_now_iso()
            self.store.write_models("taste_experiments", user_id, experiments[-20:])
            return exp

    @staticmethod
    def taste_prompt_exclusions(prompt: str) -> list[str]:
        text = (prompt or "").lower()
        rules: list[str] = []
        if any(token in text for token in ("别太吵", "不要太吵", "不吵", "低能量")):
            rules.extend(["激昂", "金属", "hard rock", "heavy metal"])
        if any(cue in text for cue in ("不要 type beat", "别推 type beat", "no type beat", "不要beat", "不要 beat")):
            rules.append("type beat")
        for match in re.finditer(r"(?:不要|别推|不想听|\bno\s+|\bwithout\s+)\s*([^，。,.；;]{1,24})", text):
            value = match.group(1).strip()
            if value:
                rules.append(value)
        return list(dict.fromkeys(rules))

    @staticmethod
    def taste_experiment_seeds_for_bucket(memory: UserMemory, prompt: str, bucket: str) -> list[str]:
        taste = memory.taste_profile or TasteProfile()
        genres = [g for g, _ in taste.top_genres[:3] if g]
        artists = [a for a, _ in taste.top_artists[:6] if a]
        if bucket == "safe":
            seeds: list[str] = [f"{a} {genres[0]}" for a in artists[:6] if genres]
            if genres:
                seeds.append(" ".join(genres[:3]))
            return TasteExperimentService.dedupe_seeds(seeds) or ["热门 推荐"]
        if bucket == "stretch":
            seeds = []
            if genres:
                seeds.append(" ".join([genres[0], "相邻", "新风格"]))
            seeds += ["neo soul", "另类 R&B", "独立流行", "律动 R&B", "氛围 说唱"]
            return TasteExperimentService.dedupe_seeds(seeds)
        return TasteExperimentService.dedupe_seeds(
            [
                "探索 新风格",
                "小众 世界音乐",
                "实验 电子",
                "融合 爵士",
                "独立 民谣",
                "另类 摇滚",
            ]
        )

    @staticmethod
    def dedupe_seeds(seeds: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for seed in seeds:
            seed = seed.strip()
            if seed and seed.lower() not in seen:
                seen.add(seed.lower())
                out.append(seed)
        return out

    def list_taste_experiments(self, user_id: str) -> list[TasteExperiment]:
        return self.store.read_models("taste_experiments", user_id, TasteExperiment)

    def get_taste_experiment(self, user_id: str, experiment_id: str) -> TasteExperiment | None:
        return next((exp for exp in self.list_taste_experiments(user_id) if exp.experiment_id == experiment_id), None)

    def delete_taste_experiment(self, user_id: str, experiment_id: str) -> bool:
        with self.store.lock("taste_experiments", user_id):
            experiments = self.store.read_models("taste_experiments", user_id, TasteExperiment)
            remaining = [exp for exp in experiments if exp.experiment_id != experiment_id]
            if len(remaining) == len(experiments):
                return False
            self.store.write_models("taste_experiments", user_id, remaining)
            return True

    def record_taste_experiment_feedback(
        self,
        request: TasteExperimentFeedbackRequest,
        *,
        find_taste_experiment_track: Any,
        apply_taste_experiment_ts_feedback: Any,
        record_taste_experiment_listen: Any,
        taste_experiment_feedback_count: Any,
    ) -> TasteExperiment:
        with self.store.lock("taste_experiments", request.user_id):
            experiments = self.store.read_models("taste_experiments", request.user_id, TasteExperiment)
            for exp in experiments:
                if exp.experiment_id != request.experiment_id:
                    continue
                item = find_taste_experiment_track(exp, request.track_key)
                if item is None:
                    raise ValueError("Track not found in experiment")
                feedback = item.feedback
                current = getattr(feedback, request.signal)
                setattr(feedback, request.signal, current + 1)
                feedback.last_signal = request.signal
                if request.signal == "rated" and request.score is not None:
                    feedback.scores.append(float(request.score))
                apply_taste_experiment_ts_feedback(item, request.signal, request.score)
                record_taste_experiment_listen(request.user_id, item, request.signal, request.score)
                if taste_experiment_feedback_count(exp) >= 6 and exp.status == "collecting":
                    exp.status = "ready"
                exp.updated_at = utc_now_iso()
                self.store.write_models("taste_experiments", request.user_id, experiments[-20:])
                return exp
        raise ValueError("Experiment not found")

    def summarize_taste_experiment(
        self,
        user_id: str,
        experiment_id: str,
        *,
        taste_experiment_bucket_stats: Any,
        bucket_label: Any,
    ) -> TasteExperimentReport:
        with self.store.lock("taste_experiments", user_id):
            experiments = self.store.read_models("taste_experiments", user_id, TasteExperiment)
            for exp in experiments:
                if exp.experiment_id != experiment_id:
                    continue
                stats = taste_experiment_bucket_stats(exp)
                feedback_total = int(sum(bucket.get("feedback_count", 0) for bucket in stats.values()))
                if feedback_total < 6:
                    summary = f"目前只有 {feedback_total} 条实验反馈，再听/跳过几首后报告会更可靠。"
                    hypothesis_result = "继续收集"
                    next_strategy = "先保持三档结构，优先补足 stretch 和 bold 的反馈。"
                else:
                    best_bucket = max(
                        stats,
                        key=lambda name: (stats[name].get("liked_rate", 0), stats[name].get("completed_rate", 0)),
                    )
                    too_far = stats.get("bold", {}).get("too_far", 0)
                    summary = f"已收集 {feedback_total} 条反馈，{bucket_label(best_bucket)} 的正反馈最强。"
                    hypothesis_result = (
                        "大胆探索边界偏远，需要收窄。"
                        if too_far >= 2
                        else f"假设部分成立：{bucket_label(best_bucket)} 当前最能解释你的反应。"
                    )
                    next_strategy = (
                        "下一轮降低 bold 能量跨度，多做相邻风格实验。"
                        if too_far >= 2
                        else "下一轮保留 safe 锚点，把 stretch 的比例提高一点。"
                    )
                report = TasteExperimentReport(
                    summary=summary,
                    bucket_stats=stats,
                    hypothesis_result=hypothesis_result,
                    next_recommendation_strategy=next_strategy,
                )
                exp.report = report
                exp.result_summary = summary
                exp.status = "reported" if feedback_total >= 6 else exp.status
                exp.updated_at = utc_now_iso()
                self.store.write_models("taste_experiments", user_id, experiments[-20:])
                return report
        raise ValueError("Experiment not found")

    def save_taste_experiment(self, experiment: TasteExperiment) -> None:
        with self.store.lock("taste_experiments", experiment.user_id):
            experiments = self.store.read_models("taste_experiments", experiment.user_id, TasteExperiment)
            experiments = [exp for exp in experiments if exp.experiment_id != experiment.experiment_id]
            experiments.append(experiment)
            self.store.write_models("taste_experiments", experiment.user_id, experiments[-20:])

    @staticmethod
    def new_taste_experiment_id(user_id: str, prompt: str) -> str:
        raw = f"{user_id}|{prompt}|{datetime.now(UTC).isoformat()}".encode()
        return "taste_" + hashlib.sha1(raw).hexdigest()[:12]

    @staticmethod
    def taste_experiment_hypothesis(memory: UserMemory) -> str:
        taste = memory.taste_profile or TasteProfile()
        genres = "、".join(name for name, _ in taste.top_genres[:2]) or "你最近反复命中的风格"
        moods = "、".join(name for name, _ in taste.top_moods[:2]) or "稳定情绪锚点"
        return f"我猜你会稳定接受 {genres}/{moods}，但探索边界可能藏在相邻风格和不同能量密度里。"

    @staticmethod
    def taste_experiment_search_seeds(memory: UserMemory, prompt: str) -> list[str]:
        taste = memory.taste_profile or TasteProfile()
        genres = [name for name, _ in taste.top_genres[:3] if name]
        moods = [name for name, _ in taste.top_moods[:3] if name]
        artists = [name for name, _ in taste.top_artists[:5] if name]
        prompt = prompt or ""
        seeds: list[str] = []
        for artist in artists[:6]:
            primary_genre = genres[1] if len(genres) > 1 else (genres[0] if genres else "")
            if primary_genre:
                seeds.append(f"{artist} {primary_genre}")
            if moods:
                seeds.append(f"{artist} {moods[0]}")
        if genres:
            seeds.append(" ".join([*genres[:2], "新风格"]))
            seeds.append(" ".join([genres[0], "小众", "相邻风格"]))
        if moods:
            seeds.append(" ".join([moods[0], "氛围", "新歌"]))
        if any(token in prompt for token in ["不一样", "听腻", "新风格", "探索", "实验"]):
            seeds.extend(["探索 新风格", "小众 R&B", "另类 R&B", "neo soul", "氛围 说唱", "新灵魂"])
        seeds.extend(["独立流行", "律动 R&B", "另类流行", "chill R&B"])
        return TasteExperimentService.dedupe_seeds(seeds) or ["探索 新风格"]

    @staticmethod
    def taste_experiment_track(
        track: Any,
        bucket: str,
        components: dict[str, float],
        reason: str,
        score: float,
    ) -> TasteExperimentTrack:
        source = getattr(track, "source", "local") or "local"
        source_id = getattr(track, "external_id", "") or getattr(track, "asset_id", "") or ""
        ref = TrackRef(
            title=getattr(track, "title", "") or "",
            artist=getattr(track, "artist", "") or "",
            source=source,
            source_id=source_id,
            cover_url=getattr(track, "cover_url", "") or "",
            genre=getattr(track, "genre", []) or [],
            mood=getattr(track, "mood", []) or [],
            score=score,
            components=components or {},
        )
        expected = {
            "safe": "如果你听完或收藏，说明稳定画像可信。",
            "stretch": "如果你喜欢，说明相邻风格可以扩大。",
            "bold": "如果你没跳过，说明探索边界比画像更宽。",
        }[bucket]
        return TasteExperimentTrack(
            track=ref,
            bucket=bucket,  # type: ignore[arg-type]
            reason=reason or f"{bucket} bucket candidate",
            expected_signal=expected,
            components=components or {},
        )
