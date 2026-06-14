from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from typing import Any

from app.models import (
    AgentGoal,
    Asset,
    DialogueState,
    ListeningEvent,
    MemoryEntry,
    MemoryUpdateRequest,
    RatingEntry,
    Segment,
    TasteProfile,  # noqa: F401  —— 对外 re-export（测试/外部按 from app.memory import TasteProfile 使用）
    UserMemory,
    utc_now_iso,
)
from app.recommend.engine import compute_taste_profile
from app.storage import JsonStore

PREFERENCE_PATTERNS = [
    re.compile(r"(?:i\s+)?(?:like|love|prefer)\s+(.+)", re.IGNORECASE),
    re.compile(r"(?:喜欢|偏好|更想要|更喜欢|爱听)(.+)"),
    # 更宽泛的音乐偏好捕获：听/在听/循环/追/迷 + 歌手/曲风
    re.compile(r"(?:在听|正在听|最近听|听了|一直听|常听|循环|单曲循环|追|迷|粉)\s*(.+)"),
    re.compile(r"(?:听了|在听)\s*(.{2,15})的?歌"),
    re.compile(r"(?:i\s+)?(?:listen\s+to|into|vibing\s+(?:to|with))\s+(.+)", re.IGNORECASE),
]

# 负面偏好提取：匹配"不要/别推/讨厌/不喜欢"等 + 后续的风格/类型词
NEGATIVE_PREFERENCE_PATTERNS = [
    re.compile(r"(?:不要|别推|讨厌|不喜欢|排除|过滤|少推|别给我|别再|不想听)\s*(.+)"),
    re.compile(r"(?:no\s+|don'?t\s+(?:want|like|give)\s+)(.+)", re.IGNORECASE),
]

GOAL_KEYWORDS = ["目标", "任务", "帮我", "导入", "歌单", "先", "然后", "再", "跑步", "整理"]

ACTION_TO_GOAL_STEP = {
    "search_web_music": "联网搜索真实曲目",
    "fetch_track_metadata": "抓取真实元数据",
    "import_netease_playlist": "导入网易云歌单",
    "search": "搜索候选音乐",
    "recommend": "挑选推荐内容",
    "playlist": "生成歌单",
    "taste": "分析用户品味",
    "retrieve": "检索证据",
    "memory_update": "更新记忆",
    "web_music_search": "联网搜索真实曲目",
    "fetch_metadata": "抓取真实元数据",
}

TECHNICAL_ACTIONS = {
    "finalize",
    "max_steps_reached",
    "fallback",
    "plan",
}


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
        with self.store.lock("memory", request.user_id):
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
        with self.store.lock("memory", user_id):
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

    @staticmethod
    def _extract_negative_preference(query: str) -> str | None:
        """从用户输入中提取负面偏好（如"不要抖音热歌"→"抖音热歌"）。"""
        for pat in NEGATIVE_PREFERENCE_PATTERNS:
            m = pat.search(query)
            if m:
                text = m.group(1).strip().rstrip("的了着过")
                if text and 2 <= len(text) <= 20:
                    return text
        return None

    def add_exclusion(self, user_id: str, rule: str) -> bool:
        """添加一条排除规则。已存在则返回 False。"""
        rule = rule.strip()
        if not rule:
            return False
        with self.store.lock("memory", user_id):
            memory = self.get_memory(user_id)
            if rule in memory.exclusion_rules:
                return False
            memory.exclusion_rules.append(rule)
            memory.updated_at = utc_now_iso()
            self.store.write_model("memory", user_id, memory)
            return True

    def remove_exclusion(self, user_id: str, rule: str) -> bool:
        """删除一条排除规则。不存在则返回 False。"""
        with self.store.lock("memory", user_id):
            memory = self.get_memory(user_id)
            if rule not in memory.exclusion_rules:
                return False
            memory.exclusion_rules.remove(rule)
            memory.updated_at = utc_now_iso()
            self.store.write_model("memory", user_id, memory)
            return True

    def list_exclusions(self, user_id: str) -> list[str]:
        """返回用户的排除规则列表。"""
        return list(self.get_memory(user_id).exclusion_rules)

    def record_listen(self, user_id: str, asset_id: str, duration: int, completed: bool, context: str | None = None) -> UserMemory:
        with self.store.lock("memory", user_id):
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
        with self.store.lock("memory", user_id):
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
            with self.store.lock("memory", uid):
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
        with self.store.lock("memory", user_id):
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
        # 关键修复：把从上传/收听歌曲算出的品味档案也带进查询。
        # 否则用户上传一堆 Beatles（taste=摇滚），推荐查询却完全不含"摇滚"，
        # 在线搜索只能返回与泛化词匹配的热门垃圾。
        taste = memory.taste_profile
        if taste:
            for genre, _ in taste.top_genres[:3]:
                parts.append(genre)
            for mood, _ in taste.top_moods[:2]:
                parts.append(mood)
            # 艺术家偏好：高分歌手带进搜索查询，提升同风格歌曲召回
            for artist, _ in taste.top_artists[:3]:
                parts.append(artist)
        return " ".join(parts)

    def auto_learn_from_turn(self, user_id: str, query: str, results: list[dict[str, Any]]) -> bool:
        """Conservatively learn from an agent turn without requiring an explicit memory tool call."""
        with self.store.lock("memory", user_id):
            memory = self.get_memory(user_id)
            changed = False

            explicit = extract_preference(query)
            if explicit:
                if self._upsert_entry(memory, explicit, "auto_explicit"):
                    changed = True
                if explicit not in memory.preferences:
                    memory.preferences.append(explicit)
                    changed = True

            # 从检索结果中提取歌手/歌名作为兴趣信号（即使没有明确说"喜欢"）
            entities_from_results = _extract_entities_from_results(results)
            for entity in entities_from_results:
                if self._upsert_entry(memory, entity, "from_search_result"):
                    changed = True

            # 负面偏好提取：用户说"不要抖音热歌""别推孟菲斯说唱"等
            negative = self._extract_negative_preference(query)
            if negative and negative not in memory.exclusion_rules:
                memory.exclusion_rules.append(negative)
                changed = True

            inferred = infer_preferences_from_results(query, results)
            for item in inferred:
                if self._upsert_entry(memory, item, "inferred_from_result"):
                    changed = True

            if changed:
                memory.updated_at = utc_now_iso()
                self.store.write_model("memory", user_id, memory)
            return changed

    def get_active_goal(self, user_id: str) -> AgentGoal | None:
        goal = self.store.read_model("goals", user_id, AgentGoal)
        if goal is None or goal.status != "active":
            return None
        return goal

    # ── DialogueState：轻量多轮延续状态 ───────────────────────────────
    def get_dialogue_state(self, user_id: str) -> DialogueState:
        state = self.store.read_model("dialogue", user_id, DialogueState)
        return state or DialogueState(user_id=user_id)

    def save_dialogue_state(
        self,
        user_id: str,
        intent: str,
        query: str,
        entities: list[str],
        genre_tags: list[str] | None = None,
        mood_tags: list[str] | None = None,
        scenario_tags: list[str] | None = None,
        shown_tracks: list[dict[str, str]] | None = None,
    ) -> DialogueState:
        prev = self.get_dialogue_state(user_id)
        state = DialogueState(
            user_id=user_id,
            last_intent=intent,
            last_query=query,
            entities=list(entities),
            genre_tags=list(genre_tags or []),
            mood_tags=list(mood_tags or []),
            scenario_tags=list(scenario_tags or []),
            turn_count=prev.turn_count + 1,
            shown_tracks=list(shown_tracks) if shown_tracks else [],
            updated_at=utc_now_iso(),
        )
        self.store.write_model("dialogue", user_id, state)
        return state

    def clear_dialogue_state(self, user_id: str) -> None:
        self.store.delete_key("dialogue", user_id)

    def ensure_goal(self, user_id: str, query: str) -> AgentGoal | None:
        goal = self.get_active_goal(user_id)
        if goal is not None:
            # 相关性门控：新 query 与旧 goal 无关时，归档旧 goal 而非强行续接，
            # 避免"50首chill歌单"这种旧任务绑死后续每一轮对话。
            if _goal_still_relevant(goal, query):
                return goal
            goal.status = "completed"
            goal.updated_at = utc_now_iso()
            self.store.write_model("goals", user_id, goal)
        if not should_start_goal(query):
            return None
        goal = AgentGoal(goal=query, steps_pending=infer_goal_steps(query))
        self.store.write_model("goals", user_id, goal)
        return goal

    def update_goal_progress(self, user_id: str, goal: AgentGoal | None, actions: list[str]) -> AgentGoal | None:
        if goal is None:
            return None
        changed = False
        for action in actions:
            if action in TECHNICAL_ACTIONS:
                continue
            label = ACTION_TO_GOAL_STEP.get(action, action)
            if label not in goal.steps_done:
                goal.steps_done.append(label)
                changed = True
            if label in goal.steps_pending:
                goal.steps_pending.remove(label)
                changed = True
        if not goal.steps_pending and goal.steps_done:
            goal.status = "completed"
            changed = True
        if changed:
            goal.updated_at = utc_now_iso()
            self.store.write_model("goals", user_id, goal)
        return goal

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
    now = datetime.now(UTC)
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


def infer_preferences_from_results(query: str, results: list[dict[str, Any]]) -> list[str]:
    lowered = query.lower()
    if not any(token in lowered for token in ["推荐", "歌单", "适合", "喜欢", "chill", "playlist", "recommend"]):
        return []

    genre_counts: dict[str, int] = {}
    mood_counts: dict[str, int] = {}
    considered = 0

    for track in _iter_verified_tracks(results):
        considered += 1
        for genre in getattr(track, "genre", []) or []:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        for mood in getattr(track, "mood", []) or []:
            mood_counts[mood] = mood_counts.get(mood, 0) + 1

    if considered < 3:
        return []

    inferred: list[str] = []
    for label, counts in [("风格", genre_counts), ("情绪", mood_counts)]:
        if not counts:
            continue
        name, count = max(counts.items(), key=lambda item: item[1])
        if count >= 2:
            inferred.append(f"{label}偏好：{name}")
    return inferred[:2]


def _extract_entities_from_results(results: list[dict[str, Any]]) -> list[str]:
    """从检索结果中提取高频歌手名，作为兴趣信号写入记忆。

    如果某个歌手在结果中出现了 2 次以上，说明用户正在关注这个歌手。
    这比等用户说"我喜欢XXX"更主动。
    """
    artist_counts: dict[str, int] = {}
    for track in _iter_verified_tracks(results):
        artist = (getattr(track, "artist", "") or "").strip()
        if artist and 2 <= len(artist) <= 30:
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
    return [artist for artist, count in artist_counts.items() if count >= 2][:5]


def _iter_verified_tracks(results: list[dict[str, Any]]) -> list[Any]:
    tracks: list[Any] = []
    for result in results:
        result_type = result.get("type")
        if result_type == "web_music_search":
            for track in result["tracks"]:
                source = getattr(track, "source", "")
                if source in {"netease", "bilibili", "youtube"}:
                    tracks.append(track)
        elif result_type == "daily_recommend":
            tracks.extend(item.asset for item in result["recommendation"].tracks[:5])
        elif result_type == "playlist":
            for track in result["playlist"].tracks[:8]:
                source = getattr(track, "source", "local")
                if source != "llm" and "fallback" not in source:
                    tracks.append(track)
    return tracks


def compute_behavior_scores(
    listening_history: list[ListeningEvent],
    asset_durations: dict[str, int] | None = None,
) -> dict[str, float]:
    """从收听历史计算每个 asset 的行为分数（Spotify BaRT 的奖励信号思想）。

    听完 (completed) → +1；秒跳（听了不到时长 10% 或绝对 < 15 秒）→ -1；
    其余部分收听按收听比例线性给分。多次收听按时间指数衰减累加，
    近期行为权重更高。返回 {asset_id: score}，分数未归一化。
    """
    asset_durations = asset_durations or {}
    now = datetime.now(UTC)
    scores: dict[str, float] = {}
    for event in listening_history:
        if not event.asset_id:
            continue
        full = asset_durations.get(event.asset_id, 0)
        listened = event.duration_listened or 0
        if event.completed:
            reward = 1.0
        elif listened < 15 or (full > 0 and listened < full * 0.1):
            reward = -1.0
        elif full > 0:
            reward = max(-1.0, min(1.0, (listened / full) * 2 - 1))
        else:
            reward = 0.0
        try:
            ts = datetime.fromisoformat(event.timestamp)
            age_days = max(0, (now - ts).days)
        except (ValueError, TypeError):
            age_days = 0
        decay = math.exp(-0.05 * age_days)
        scores[event.asset_id] = scores.get(event.asset_id, 0.0) + reward * decay
    return scores


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
    # 去掉前后的语气词（上/了/过/的/啊/呢/吧/哦）和标点
    value = re.sub(r"\s+", " ", value.strip("。.!? ，,；;！"))
    value = re.sub(r"^[上进了的了啊呢吧哦着过]", "", value)
    value = re.sub(r"[上了的了啊呢吧哦着过]$", "", value)
    return value.strip()


def should_start_goal(query: str) -> bool:
    lowered = query.lower()
    return any(keyword in lowered for keyword in GOAL_KEYWORDS)


def _goal_still_relevant(goal: AgentGoal, query: str) -> bool:
    """判断新一轮 query 是否还属于旧 goal 的范畴。

    不相关时应归档旧 goal、避免"50首chill歌单"这类任务绑死后续每轮对话。
    判据：query 与 goal 文本有词汇重叠，或 query 本身不像一个新任务请求
    （没有 should_start_goal 关键词，视为对旧任务的追问/延续）。
    """
    if not should_start_goal(query):
        return True  # 不是新任务请求 → 当作旧 goal 的延续

    goal_tokens = set(re.findall(r"[\w一-鿿]+", goal.goal.lower()))
    query_tokens = set(re.findall(r"[\w一-鿿]+", query.lower()))
    # 去掉通用动词噪声，避免"帮我""生成"等词造成假相关
    noise = {"帮我", "生成", "一个", "的", "我", "想", "要", "请", "给我", "做", "个"}
    overlap = (goal_tokens & query_tokens) - noise
    return bool(overlap)


def infer_goal_steps(query: str) -> list[str]:
    lowered = query.lower()
    steps: list[str] = []
    if "导入" in lowered and "歌单" in lowered:
        steps.append("导入网易云歌单")
    if any(token in lowered for token in ["联网", "真实", "外部", "网易", "b站", "bilibili"]):
        steps.append("联网搜索真实曲目")
    if any(token in lowered for token in ["挑", "推荐", "跑步", "适合"]):
        steps.append("挑选推荐内容")
    if any(token in lowered for token in ["建歌单", "生成歌单", "做歌单", "整理", "歌单"]):
        steps.append("生成歌单")
    # 不再添加"理解并推进用户目标"这种永远完不成的兜底步骤——
    # 它会让 goal 永远停留在 active、绑死后续对话。无明确步骤时返回空列表，
    # goal 在首轮动作后即可正常完成。
    return steps
