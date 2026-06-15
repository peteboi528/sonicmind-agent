from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from app.answer import (
    collect_known_titles,
    guard_answer,
)
from app.answer import (
    collect_tracks as _collect_tracks,
)
from app.answer import (
    goal_progress as _goal_progress,
)
from app.answer import (
    infer_count as _infer_count,
)
from app.answer import (
    song_card as _song_card,
)
from app.config import settings
from app.context import ContextBudgetManager, ContextSource
from app.graph.tag_rules import extract_tags
from app.intents import get_intent, is_continuation, is_valid_intent, match_intent_by_keywords
from app.llm.structured import extract_json_dict
from app.models import AgentAnswer, AgentPlan, RetrievalPlan, StreamEvent
from app.prompts import QUERY_PLAN_SYSTEM
from app.prompts.reflect import CANDIDATE_REFLECTION_SYSTEM, CANDIDATE_REFLECTION_USER

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.agent import AudioVisualAgent
    from app.graph.state import AgentState


def load_context(agent: AudioVisualAgent, state: AgentState) -> AgentState:
    memory = agent.memory.get_memory(state["user_id"])
    goal = agent.memory.get_active_goal(state["user_id"])
    memory_query = agent.memory.weighted_query(memory)
    dialogue = agent.memory.get_dialogue_state(state["user_id"])

    # P1-G：跨会语义召回——把与本轮 query 相关的历史情景记忆 + 巩固画像并入记忆上下文，
    # 让"你三周前说过想要慵懒爵士"这类信号也能影响检索与作答。
    recall_lines: list[str] = []
    try:
        recalled = agent.memory.recall_episodes(state["user_id"], state["query"])
    except Exception:
        recalled = []
        logger.debug("load_context: 语义召回失败，跳过", exc_info=True)
    profile = (memory.consolidated_profile or "").strip()
    memory_parts = [p for p in [memory_query, profile, *recalled] if p]
    enriched_memory = " ".join(memory_parts)
    if profile:
        recall_lines.append(f"巩固画像：{profile}")
    if recalled:
        recall_lines.append(f"语义召回 {len(recalled)} 条相关历史偏好。")

    # GSSC：按优先级把用户输入/记忆/历史压进 token 预算，产出追踪报告。
    history = state.get("history") or []
    history_text = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in history)
    sources = [
        ContextSource(name="user_query", content=state["query"], priority=0, min_tokens=200),
        ContextSource(name="memory", content=enriched_memory, priority=1, min_tokens=80),
        ContextSource(name="history", content=history_text, priority=2, min_tokens=40),
    ]
    budgeted, report = ContextBudgetManager(total_budget=2000).allocate(sources)

    context = {
        "memory_query": budgeted.get("memory", enriched_memory),
        "history_text": budgeted.get("history", ""),
        "active_goal": goal.model_dump(mode="json") if goal else None,
        "resource_count": len(agent.list_resource_tracks(50)),
        "budget_report": report.as_lines(),
        "dialogue_state": dialogue.model_dump(mode="json"),
    }
    return {
        **state,
        "context": context,
        "results": [],
        "trace": ["[load_context] 载入记忆、目标和资源库摘要。", *recall_lines, *report.as_lines()],
        "events": [StreamEvent(type="plan", content="正在读取记忆和资源库状态。")],
    }


def plan_intent(agent: AudioVisualAgent, state: AgentState) -> AgentState:
    query = state["query"]
    context = state.get("context") or {}
    history_text = context.get("history_text", "")
    plan = plan_with_llm(agent, query, history_text) or build_agent_plan(query)
    plan, inherited = _apply_dialogue_continuation(plan, query, context.get("dialogue_state"))
    # 安全网：LLM 可能把介绍/百科类问题误判为 discuss，检查关键词自动升级
    plan = _upgrade_artist_info(plan, query)
    trace_line = f"[plan] {plan.reasoning_summary}"
    if inherited:
        trace_line += f"（延续上一轮：继承 {inherited}）"
    return {
        **state,
        "plan": plan,
        "trace": [*state.get("trace", []), trace_line],
        "events": [
            *state.get("events", []),
            StreamEvent(type="plan", content=plan.reasoning_summary, payload=plan.model_dump(mode="json")),
        ],
    }


def _apply_dialogue_continuation(
    plan: AgentPlan,
    query: str,
    dialogue_state: dict[str, Any] | None,
) -> tuple[AgentPlan, str]:
    """延续指令（再来几首/换一批/类似这个）程序化继承上一轮实体与标签。

    仅当本轮被判为延续、且本轮没抽到自己的实体时才继承；返回 (新计划, 继承说明)。
    话题切换（本轮自带新实体/明确意图）不继承——交由 finalize 时覆盖旧状态。
    """
    if not dialogue_state or not is_continuation(query):
        return plan, ""
    rp = plan.retrieval_plan
    # 本轮已自带实体，说明用户给了新对象，不继承。
    if rp.entities:
        return plan, ""
    prev_entities = dialogue_state.get("entities") or []
    prev_intent = dialogue_state.get("last_intent") or plan.intent
    if not prev_entities and prev_intent in {"chat"}:
        return plan, ""

    # 继承上一轮意图（除非本轮 LLM 给了更具体的非 chat 意图）。
    intent = plan.intent if plan.intent not in {"chat"} else prev_intent
    spec = get_intent(intent)
    use_web = plan.online_required or spec.online_default
    merged = RetrievalPlan(
        use_local=rp.use_local,
        use_vector=rp.use_vector or bool(dialogue_state.get("mood_tags") or dialogue_state.get("scenario_tags")),
        use_web=use_web,
        entities=prev_entities,
        genre_filter=rp.genre_filter or dialogue_state.get("genre_tags") or [],
        mood_filter=rp.mood_filter or dialogue_state.get("mood_tags") or [],
        scenario_filter=rp.scenario_filter or dialogue_state.get("scenario_tags") or [],
    )
    new_plan = plan.model_copy(update={
        "intent": intent,
        "strategy": spec.strategy_for(use_web),
        "tools_needed": spec.tools_for(use_web),
        "online_required": use_web,
        "retrieval_plan": merged,
    })
    # 把上一轮已展示的曲目挂到 new_plan 上，供 _run_tool 传给 recommend/search 层去重
    prev_shown = dialogue_state.get("shown_tracks") or []
    if prev_shown:
        new_plan._excluded_tracks = prev_shown  # type: ignore[attr-defined]
    return new_plan, "、".join(prev_entities) or prev_intent


# artist_info 安全网信号——命中时从 discuss 升级到 artist_info
_ARTIST_INFO_SIGNALS = (
    "介绍", "背景", "成员", "出道", "简介", "资料", "百科", "是谁", "什么团",
    "来头", "历史", "故事", "怎么火", "怎么出", "经历", "生平",
    "biography", "about", "history",
)


def _upgrade_artist_info(plan: AgentPlan, query: str) -> AgentPlan:
    """LLM 可能把介绍/百科类问题误判为 discuss；检查关键词自动升级到 artist_info。"""
    if plan.intent != "discuss":
        return plan
    lowered = query.lower()
    if any(sig in lowered or sig in query for sig in _ARTIST_INFO_SIGNALS):
        spec = get_intent("artist_info")
        return plan.model_copy(update={
            "intent": "artist_info",
            "strategy": spec.strategy_for(True),
            "tools_needed": spec.tools_for(True),
            "online_required": True,
            "reasoning_summary": f"检测到百科类信号，升级为 artist_info。{plan.reasoning_summary}",
        })
    return plan


def execute_tools(agent: AudioVisualAgent, state: AgentState) -> AgentState:
    plan = state["plan"]
    query = state["query"]
    user_id = state["user_id"]
    top_k = state.get("top_k", 5)
    results = list(state.get("results", []))
    trace = list(state.get("trace", []))
    events = list(state.get("events", []))

    executed: set[str] = set()
    for tool in plan.tools_needed:
        _run_tool(agent, tool, plan, query, user_id, top_k, results, trace, events)
        executed.add(tool)

    # 标记是否需要 web_fallback（由 route_after_execute 条件边消费）。
    need_fallback = _needs_web_fallback(plan, results, executed)
    return {**state, "results": results, "trace": trace, "events": events, "_need_web_fallback": need_fallback}


def web_fallback(agent: AudioVisualAgent, state: AgentState) -> AgentState:
    """web_fallback 节点（对齐 SoulTuner route_after_search / _need_web_fallback）：
    本地/检索类候选不足时联网兜底补搜。"""
    plan = state["plan"]
    query = state["query"]
    user_id = state["user_id"]
    top_k = state.get("top_k", 5)
    results = list(state.get("results", []))
    trace = list(state.get("trace", []))
    events = list(state.get("events", []))

    trace.append("[web_fallback] 本地候选不足，触发联网兜底补搜。")
    events.append(StreamEvent(type="eval", content="本地候选不足，联网兜底补搜。"))
    _run_tool(agent, "web_music_search", plan, query, user_id, top_k, results, trace, events)
    return {**state, "results": results, "trace": trace, "events": events, "_need_web_fallback": False}


def route_after_execute(state: AgentState) -> str:
    """条件路由：候选不足 → web_fallback，否则 → evaluate。"""
    return "web_fallback" if state.get("_need_web_fallback") else "evaluate"


def _needs_web_fallback(plan: AgentPlan, results: list[dict[str, Any]], executed: set[str]) -> bool:
    if "web_music_search" in executed or not plan.online_required:
        return False
    if plan.intent not in {"recommend", "search", "playlist"}:
        return False
    verified = [
        t for t in _collect_tracks(results)
        if getattr(t, "source", "local") in {"netease", "bilibili", "youtube"}
    ]
    need = plan.target_count or 3
    return len(verified) < need


def _query_with_entities(query: str, plan: AgentPlan) -> str:
    """把 plan 里的实体拼进 query。延续指令（"多来几首"）本身不含实体，
    实体是从上一轮 DialogueState 继承到 retrieval_plan 的；不拼进 query 的话
    recommend/search/playlist 工具拿不到实体，会走情绪/场景路由搜空。

    实体已出现在 query 中则不重复拼接（避免 "找 The Weeknd The Weeknd"）。
    """
    entities = plan.retrieval_plan.entities
    if not entities:
        return query
    lowered = query.lower()
    extra = [e for e in entities if e and e.lower() not in lowered]
    if not extra:
        return query
    return " ".join([query, *extra]).strip()


def _filter_excluded(tracks: list[Any], excluded: list[dict[str, str]]) -> list[Any]:
    """过滤掉上一轮已展示给用户的歌曲，避免延续指令时推荐重复曲目。

    匹配策略：(title, source_id) 组合键，source_id 为空时退化为 title 匹配。
    """
    if not excluded:
        return tracks
    seen_keys: set[tuple[str, str]] = set()
    seen_titles: set[str] = set()
    for ex in excluded:
        title = ex.get("title", "").lower().strip()
        sid = ex.get("source_id", "").strip()
        if title:
            seen_titles.add(title)
            if sid:
                seen_keys.add((title, sid))
    filtered = []
    for t in tracks:
        t_title = (getattr(t, "title", "") or "").lower().strip()
        t_sid = getattr(t, "external_id", "") or getattr(t, "asset_id", "") or ""
        # source_id 精确匹配
        if t_title and t_sid and (t_title, t_sid) in seen_keys:
            continue
        # title 退化匹配
        if t_title and t_title in seen_titles:
            continue
        filtered.append(t)
    return filtered


def _run_tool(
    agent: AudioVisualAgent,
    tool: str,
    plan: AgentPlan,
    query: str,
    user_id: str,
    top_k: int,
    results: list[dict[str, Any]],
    trace: list[str],
    events: list[StreamEvent],
) -> None:
    events.append(StreamEvent(type="tool_start", content=f"调用 {tool}", payload={"tool": tool}))
    # 从原始 query 中提取核心搜索词（去掉中文功能词），用于搜索 API 和相关性过滤。
    # 不做这步的话 "帮我生成一些drake的歌" 直接发到网易云 API，搜不到结果。
    from app.agent import _extract_search_query as _extract_core
    search_core = _extract_core(query) or query
    # 延续指令时从 plan 中提取上一轮已展示的曲目（用于去重）
    excluded_tracks = getattr(plan, "_excluded_tracks", None) or []
    if tool == "web_music_search":
        # 搜索用核心词 + plan 实体（LLM 抽取的歌手/歌名），相关性过滤只用核心词
        search_q = " ".join(filter(None, [search_core, *plan.retrieval_plan.entities])).strip()
        tracks = agent.search_web_music(search_q, top_k=max(plan.target_count or top_k, top_k), relevance_query=search_core)
        # 去除上一轮已展示的歌曲
        if excluded_tracks:
            tracks = _filter_excluded(tracks, excluded_tracks)
        for track in tracks:
            if hasattr(agent, "library"):
                agent.library.upsert_external(track)
        results.append({"type": "web_music_search", "tracks": tracks})
        trace.append(f"[web_music_search] 获取 {len(tracks)} 个线上/候选结果。")
        # 先吐候选卡片（对齐 SoulTuner __songs__ 先于解释文本的体验）。
        cards = [_song_card(t) for t in tracks]
        events.append(StreamEvent(type="candidates", content=f"候选 {len(tracks)} 个", payload={"count": len(tracks), "cards": cards}))
    elif tool == "recommend":
        # 延续指令（"多来几首"）本身不含实体，但 plan 从上一轮继承了实体；
        # 拼进 query 让 recommend_for_query 走精确搜索而非情绪/场景路由（否则搜空）。
        rec = agent.recommend_for_query(user_id, _query_with_entities(query, plan), top_k=top_k, excluded_tracks=excluded_tracks)
        results.append({"type": "daily_recommend", "recommendation": rec})
        trace.append(f"[recommend] 生成 {len(rec.tracks)} 首推荐。")
        cards = [_song_card(item.asset, reason=item.reason, score=item.score, components=item.components) for item in rec.tracks]
        events.append(StreamEvent(type="candidates", content=f"推荐 {len(cards)} 首", payload={"count": len(cards), "cards": cards}))
    elif tool == "playlist":
        seed_tracks = _collect_tracks(results)
        playlist = agent.generate_playlist(user_id, _query_with_entities(query, plan), seed_tracks=seed_tracks, target_count=plan.target_count)
        # 延续指令去重：过滤掉上一轮已展示的曲目
        if excluded_tracks and playlist.tracks:
            playlist.tracks = _filter_excluded(playlist.tracks, excluded_tracks)
        results.append({"type": "playlist", "playlist": playlist})
        trace.append(f"[playlist] 生成 {len(playlist.tracks)} 首歌单。")
        cards = [_song_card(t) for t in playlist.tracks]
        events.append(StreamEvent(type="candidates", content=f"歌单 {len(cards)} 首", payload={"count": len(cards), "cards": cards}))
    elif tool == "search":
        search = agent.search(user_id, _query_with_entities(query, plan), include_external=True, top_k=max(top_k, 12))
        # 延续指令去重：过滤掉上一轮已展示的曲目
        if excluded_tracks:
            search.external = _filter_excluded(search.external, excluded_tracks)
        results.append({"type": "search", "response": search})
        trace.append(f"[search] 本地 {len(search.local)} 首，外部 {len(search.external)} 首。")
    elif tool == "taste":
        summary = agent.summarize_taste(user_id)
        results.append({"type": "taste", "summary": summary})
        trace.append("[taste] 已总结用户品味。")
    elif tool == "import":
        imported = agent.import_netease_playlist(query, user_id=user_id, limit=plan.target_count or 100)
        results.append({"type": "import_netease_playlist", "result": imported})
        trace.append(f"[import] 导入歌单《{imported.get('name', '')}》：新增 {imported.get('imported', 0)} 首。")
    elif tool == "journey":
        journey = agent.generate_music_journey(user_id, query)
        results.append({"type": "journey", "journey": journey})
        trace.append(f"[journey] 生成 {len(journey.get('phases', []))} 个音乐旅程阶段。")
    elif tool == "video_search":
        # video 意图：直接搜 B站/YouTube，不走网易云
        search_q = " ".join(filter(None, [search_core, *plan.retrieval_plan.entities])).strip()
        tracks = agent.search_videos(search_q, top_k=plan.target_count or top_k)
        for track in tracks:
            if hasattr(agent, "library"):
                agent.library.upsert_external(track)
        results.append({"type": "video_search", "tracks": tracks})
        trace.append(f"[video_search] 获取 {len(tracks)} 个视频结果（B站+YouTube）。")
        cards = [_song_card(t) for t in tracks]
        events.append(StreamEvent(type="candidates", content=f"视频 {len(tracks)} 个", payload={"count": len(tracks), "cards": cards}))
    elif tool == "web_info_search":
        # artist_info 意图：用搜索引擎查百科
        search_q = " ".join(filter(None, [search_core, *plan.retrieval_plan.entities])).strip()
        search_results = agent.search_artist_info(search_q)
        results.append({"type": "web_info_search", "search_results": search_results})
        trace.append(f"[web_info_search] 获取 {len(search_results)} 条搜索结果。")
    events.append(StreamEvent(type="tool_result", content=trace[-1], payload={"tool": tool}))


def evaluate(agent: AudioVisualAgent, state: AgentState) -> AgentState:
    plan = state["plan"]
    results = state.get("results", [])
    candidate_count = len(_collect_tracks(results))
    message = "结果可用于回答。"
    if plan.target_count and candidate_count < plan.target_count:
        message = f"候选 {candidate_count}/{plan.target_count}，不足时必须诚实说明，不编造补齐。"
    return {
        **state,
        "trace": [*state.get("trace", []), f"[eval] {message}"],
        "events": [*state.get("events", []), StreamEvent(type="eval", content=message)],
    }


def reflect(agent: AudioVisualAgent, state: AgentState) -> AgentState:
    """P1-F：finalize 前的自省节点——LLM 语义核对候选是否违反用户约束，剔除违规者。

    把质量控制从「事后清理」（guard_answer 删幻觉歌名）升级到「交付前自查」。
    - 仅对会列曲目的意图（recommend/search/playlist/video）生效。
    - mock 模式（无真实 LLM）跳过：确定性 _filter_excluded 已在 execute_tools 跑过子串过滤。
    - LLM 调用失败/无违反时不改动 results（安全降级，不阻塞主流程）。
    """
    plan = state["plan"]
    if plan.intent not in {"recommend", "search", "playlist", "video"}:
        return state
    if settings.mock_mode:
        return state
    results = state.get("results", [])
    tracks = _collect_tracks(results)
    if not tracks:
        return state
    constraints = _gather_constraints(agent, state)
    if not constraints:
        return state
    drop_indices = _llm_reflect_tracks(agent, tracks, constraints)
    trace = list(state.get("trace", []))
    events = list(state.get("events", []))
    if drop_indices:
        drop_keys = {_track_key(tracks[i]) for i in drop_indices if i < len(tracks)}
        results = _drop_tracks_from_results(results, drop_keys)
        labels = "、".join(_track_label(tracks[i]) for i in drop_indices if i < len(tracks))[:120]
        note = f"[reflect] LLM 核对剔除 {len(drop_indices)} 首违反约束候选：{labels}"
    else:
        note = "[reflect] LLM 核对通过，无候选违反用户约束。"
    trace.append(note)
    events.append(StreamEvent(type="eval", content=note))
    return {**state, "results": results, "trace": trace, "events": events}


def _gather_constraints(agent: AudioVisualAgent, state: AgentState) -> list[str]:
    """收集用户约束：排除规则 + query 里的负面偏好 + plan 的正面偏好。"""
    constraints: list[str] = []
    user_id = state["user_id"]
    try:
        constraints.extend(agent.memory.list_exclusions(user_id))
    except Exception:
        logger.debug("reflect: 读取排除规则失败", exc_info=True)
    try:
        neg = agent.memory._extract_negative_preference(state["query"])
        if neg:
            constraints.append(neg)
    except Exception:
        pass
    plan = state["plan"]
    if getattr(plan, "genre_filter", None):
        constraints.append(f"曲风偏好：{'/'.join(plan.genre_filter)}")
    if getattr(plan, "mood_filter", None):
        constraints.append(f"情绪偏好：{'/'.join(plan.mood_filter)}")
    return [c for c in constraints if c]


def _track_key(track: Any) -> str:
    title = (getattr(track, "title", "") or "").strip().lower()
    src = (getattr(track, "source", "") or "")
    sid = (getattr(track, "external_id", "") or getattr(track, "asset_id", "") or "")
    return f"{title}|{src}|{sid}".lower()


def _track_label(track: Any) -> str:
    title = getattr(track, "title", "") or "?"
    artist = getattr(track, "artist", "") or ""
    return f"{title}" + (f"-{artist}" if artist else "")


def _llm_reflect_tracks(agent: AudioVisualAgent, tracks: list[Any], constraints: list[str]) -> list[int]:
    """让 LLM 核对候选 vs 约束，返回该剔除的下标。失败/无违反返回 []。"""
    catalog = "\n".join(
        f"[{i}] {_track_label(t)} | 曲风:{'/'.join(getattr(t, 'genre', []) or [])} "
        f"情绪:{'/'.join(getattr(t, 'mood', []) or [])}"
        for i, t in enumerate(tracks[:15])
    )
    cons = "；".join(f"({i + 1}) {c}" for i, c in enumerate(constraints[:8]))
    prompt = CANDIDATE_REFLECTION_USER.format(catalog=catalog, constraints=cons)
    try:
        raw = agent.llm.generate(prompt, system=CANDIDATE_REFLECTION_SYSTEM, temperature=0.0)
        data = extract_json_dict(raw)
        drop = data.get("drop") if isinstance(data, dict) else None
        if isinstance(drop, list):
            return [int(x) for x in drop if isinstance(x, (int, float)) and 0 <= int(x) < len(tracks)]
    except Exception:
        logger.debug("reflect LLM 核对失败，跳过", exc_info=True)
    return []


def _drop_tracks_from_results(results: list[dict[str, Any]], drop_keys: set[str]) -> list[dict[str, Any]]:
    """从 results 的各类型结果里移除命中 drop_keys 的曲目（reflect 拒绝的）。

    覆盖主要结果类型；未知类型不处理（graceful，finalize 仍能正常工作）。
    """
    for r in results:
        t = r.get("type")
        if t in {"recommend", "recommend_music", "daily_recommend"}:
            rec = r.get("recommendation")
            if rec is not None and hasattr(rec, "tracks"):
                rec.tracks = [tr for tr in rec.tracks if _track_key(tr) not in drop_keys]
        elif t == "playlist":
            pl = r.get("playlist")
            if pl is not None and hasattr(pl, "tracks"):
                pl.tracks = [tr for tr in pl.tracks if _track_key(tr) not in drop_keys]
        elif t in {"web_music_search", "search"}:
            r["tracks"] = [tr for tr in r.get("tracks", []) if _track_key(tr) not in drop_keys]
    return results


def finalize(agent: AudioVisualAgent, state: AgentState) -> AgentState:
    memory_query = (state.get("context") or {}).get("memory_query", "")
    history_text = (state.get("context") or {}).get("history_text", "")
    answer_text = compose_answer(
        state["query"], state.get("results", []), state["plan"],
        agent=agent, memory_query=memory_query, history_text=history_text,
        user_id=state["user_id"],
    )
    known = collect_known_titles(state.get("results", []))
    answer_text, removed = guard_answer(answer_text, known)
    memory_updated = agent.memory.auto_learn_from_turn(state["user_id"], state["query"], state.get("results", []))
    goal = None
    if state["plan"].intent != "chat":
        goal = agent.memory.ensure_goal(state["user_id"], state["query"])
        goal = agent.memory.update_goal_progress(state["user_id"], goal, _completed_actions(state.get("results", [])))
    trace = list(state.get("trace", []))
    if removed:
        trace.append(f"[guard] 移除 {len(removed)} 个未核实歌名。")
    trace.append("[final] 输出 grounded answer。")
    _persist_dialogue_state(agent, state)
    answer = AgentAnswer(
        answer=answer_text,
        evidences=[],
        memory_updated=memory_updated,
        agent_trace=trace,
        pending_goal=goal.goal if goal and goal.status == "active" else None,
        goal_progress=_goal_progress(goal),
    )
    # 权威卡片：与答案文本实际列出的曲目严格一一对应，挂到 final 事件。
    # 前端收到后替换流式预览卡片，保证「文本列几首 = 底部几张卡」。
    final_payload = answer.model_dump(mode="json")
    listed = _select_listed_tracks(state.get("results", []), state["plan"])
    if listed:
        final_payload["cards"] = _aligned_cards(listed, state.get("events", []))
    return {
        **state,
        "answer": answer,
        "trace": trace,
        "events": [*state.get("events", []), StreamEvent(type="final", content=answer_text, payload=final_payload)],
    }


def _select_listed_tracks(results: list[dict[str, Any]], plan: AgentPlan) -> list[Any]:
    """返回答案文本实际会列出的那批曲目（与 compose_answer 的截断逻辑严格一致）。

    仅对会渲染确定性曲目清单的意图返回非空（recommend/search/playlist）；
    chat/discuss/taste/journey 的文本不是一行行歌名，返回 [] 表示不接管卡片，
    让前端保留流式预览卡片。
    """
    if plan.intent == "playlist":
        pl = next((r["playlist"] for r in results if r.get("type") == "playlist"), None)
        tracks = list(pl.tracks) if pl and pl.tracks else _collect_tracks(results)
        return tracks[: plan.target_count or 30]
    if plan.intent in {"recommend", "search", "video"}:
        tracks = _collect_tracks(results)
        if plan.target_count:
            tracks = tracks[: plan.target_count]
        return tracks[: plan.target_count or 12]
    return []


def _aligned_cards(tracks: list[Any], events: list[StreamEvent]) -> list[dict[str, Any]]:
    """为 listed 曲目生成卡片，按 (title, source, id) 复用流式预览阶段已发出的
    卡片（保留 reason/score/components），找不到的再用 _song_card 兜底合成。"""
    existing: dict[tuple[str, str, str], dict[str, Any]] = {}
    for ev in events:
        if ev.type != "candidates":
            continue
        for card in (ev.payload or {}).get("cards", []):
            key = (
                str(card.get("title", "")).lower(),
                str(card.get("source", "")),
                str(card.get("source_id", "")),
            )
            existing.setdefault(key, card)

    cards: list[dict[str, Any]] = []
    for track in tracks:
        title = getattr(track, "title", "")
        source = getattr(track, "source", "local")
        sid = getattr(track, "external_id", "") or getattr(track, "asset_id", "")
        key = (str(title).lower(), str(source), str(sid))
        cards.append(existing.get(key) or _song_card(track))
    return cards


def _persist_dialogue_state(agent: AudioVisualAgent, state: AgentState) -> None:
    """把本轮意图/实体/标签写回 DialogueState，供下一轮延续指令继承。

    chat 意图视为话题中断，清空旧状态（下一轮"再来几首"无可继承对象时
    不会错误复用过期实体）。其余意图覆盖保存最新一轮上下文。
    同时记录本轮展示给用户的曲目（shown_tracks），供下一轮去重。
    """
    plan = state["plan"]
    user_id = state["user_id"]
    if plan.intent == "chat":
        agent.memory.clear_dialogue_state(user_id)
        return
    rp = plan.retrieval_plan
    # 收集本轮最终展示的曲目摘要，用于下一轮去重
    listed = _select_listed_tracks(state.get("results", []), plan)
    shown = [
        {
            "title": getattr(t, "title", ""),
            "artist": getattr(t, "artist", "") or "",
            "source": getattr(t, "source", "local"),
            "source_id": getattr(t, "external_id", "") or getattr(t, "asset_id", ""),
        }
        for t in listed
    ]
    agent.memory.save_dialogue_state(
        user_id,
        intent=plan.intent,
        query=state["query"],
        entities=rp.entities,
        genre_tags=rp.genre_filter,
        mood_tags=rp.mood_filter,
        scenario_tags=rp.scenario_filter,
        shown_tracks=shown,
    )


def plan_with_llm(agent: AudioVisualAgent, query: str, history_text: str = "") -> AgentPlan | None:
    """用 LLM 产出结构化 AgentPlan：LLM 判意图 + 抽实体，标签走确定性规则。

    history_text 非空时拼进 prompt，让意图规划理解多轮指代（如"再来几首"
    需沿用上一轮的歌手/场景实体）。失败（解析错误 / LLM 不可用）返回 None，
    调用方降级到关键词 build_agent_plan。
    """
    try:
        if history_text.strip():
            user_prompt = f"【最近对话】\n{history_text}\n\n【本轮输入】\n{query}"
        else:
            user_prompt = query
        raw = agent.llm.generate(user_prompt, system=QUERY_PLAN_SYSTEM, temperature=0.1)
    except Exception:
        return None
    data = extract_json_dict(raw)
    if not data or not is_valid_intent(data.get("intent", "")):
        return None

    intent = data["intent"]
    spec = get_intent(intent)
    entities = [e for e in (data.get("entities") or []) if isinstance(e, str) and e.strip()]
    use_local = bool(data.get("use_local", True))
    use_vector = bool(data.get("use_vector", False))
    use_web = bool(data.get("use_web", spec.online_default))
    target = data.get("target_count")
    target = int(target) if isinstance(target, (int, float)) and target else _infer_count(query)

    tags = extract_tags(query)
    retrieval = RetrievalPlan(
        use_local=use_local,
        use_vector=use_vector,
        use_web=use_web,
        entities=entities,
        genre_filter=tags["genre"],
        mood_filter=tags["mood"],
        scenario_filter=tags["scenario"],
    )
    return AgentPlan(
        intent=intent,
        strategy=spec.strategy_for(use_web),
        tools_needed=spec.tools_for(use_web),
        target_count=target,
        online_required=use_web,
        reasoning_summary=(data.get("reasoning") or "").strip() or spec.summary,
        retrieval_plan=retrieval,
    )


def _keyword_retrieval_plan(query: str, use_web: bool) -> RetrievalPlan:
    tags = extract_tags(query)
    return RetrievalPlan(
        use_local=True,
        use_vector=bool(tags["mood"] or tags["scenario"]),
        use_web=use_web,
        genre_filter=tags["genre"],
        mood_filter=tags["mood"],
        scenario_filter=tags["scenario"],
    )


def build_agent_plan(query: str) -> AgentPlan:
    """关键词 fallback 规划：LLM 不可用 / 解析失败时按 registry 的关键词信号判意图。"""
    target = _infer_count(query)
    if _is_smalltalk(query):
        spec = get_intent("chat")
        return AgentPlan(
            intent="chat",
            strategy=spec.strategy_for(False),
            tools_needed=[],
            online_required=False,
            reasoning_summary="这是普通寒暄，不需要联网搜索或音乐候选。",
        )

    intent = match_intent_by_keywords(query) or "chat"
    spec = get_intent(intent)
    use_web = spec.online_default
    tools = spec.tools_for(use_web)

    # import 特例：同时想"推荐/挑/适合"时，导入后接 recommend。
    if intent == "import" and any(t in query for t in ["推荐", "挑", "适合"]):
        tools = ["import", "recommend"]

    # 需要联网/向量检索的意图带上规则填充的 retrieval_plan；纯对话/品味不带。
    retrieval = (
        _keyword_retrieval_plan(query, use_web=use_web)
        if intent in {"import", "playlist", "discuss"}
        else RetrievalPlan()
    )

    return AgentPlan(
        intent=intent,
        strategy=spec.strategy_for(use_web),
        tools_needed=tools,
        target_count=target if intent != "chat" else None,
        online_required=use_web,
        reasoning_summary=spec.summary,
        retrieval_plan=retrieval,
    )


def _completed_actions(results: list[dict[str, Any]]) -> list[str]:
    mapping = {
        "web_music_search": "web_music_search",
        "daily_recommend": "recommend",
        "playlist": "playlist",
        "search": "search",
        "taste": "taste",
        "journey": "playlist",
        "import_netease_playlist": "import_netease_playlist",
        "video_search": "video_search",
        "web_info_search": "web_info_search",
    }
    return [mapping.get(result.get("type", ""), result.get("type", "")) for result in results]


def compose_answer(
    query: str,
    results: list[dict[str, Any]],
    plan: AgentPlan,
    agent: AudioVisualAgent | None = None,
    memory_query: str = "",
    history_text: str = "",
    user_id: str = "",
) -> str:
    if plan.intent == "chat":
        return _compose_chat_response(query, agent, history_text, user_id=user_id) or "你好，我在。有什么音乐上的事可以帮你?"
    if plan.intent == "discuss":
        tracks = _collect_tracks(results)
        return _compose_discussion(query, tracks, agent, history_text) or "抱歉，我暂时无法讨论这个话题。"
    if plan.intent == "video":
        tracks = _collect_tracks(results)
        return _compose_video_answer(query, tracks, agent, history_text) or "这轮没有搜到视频结果。"
    if plan.intent == "artist_info":
        return _compose_artist_info_answer(query, results, agent, history_text) or "抱歉，暂时没找到相关信息。"
    if plan.intent == "taste":
        return next((r["summary"] for r in results if r.get("type") == "taste"), "还没有足够品味数据。")
    if plan.intent == "journey":
        journey = next((r["journey"] for r in results if r.get("type") == "journey"), None)
        if not journey:
            return "这轮没有生成可追溯的音乐旅程。"
        lines = [f"音乐旅程：{journey['instruction']}"]
        for phase in journey["phases"]:
            titles = "、".join(f"《{t['title']}》" for t in phase["tracks"])
            lines.append(f"- {phase['name']}：{phase['goal']}。{titles or '暂无候选'}")
        return "\n".join(lines)
    # 歌单意图：用 LLM 精选的 playlist 结果，保留歌单名和描述
    if plan.intent == "playlist":
        return _compose_playlist_answer(query, results, plan, agent, memory_query, history_text)
    tracks = _collect_tracks(results)
    if not tracks:
        return "这轮没有拿到可追溯的音乐候选；我不会用未核实歌名硬凑结果。"
    if plan.target_count:
        tracks = tracks[: plan.target_count]
    shortfall = bool(plan.target_count and len(tracks) < plan.target_count)
    # 引言：优先让 LLM 结合记忆+历史生成有温度的开场；失败回退确定性模板。
    # 歌曲清单始终由真实候选确定性拼接——LLM 绝不参与歌名生成（防幻觉）。
    intro = _compose_intro(query, tracks, plan, agent, memory_query, shortfall, history_text)
    lines = [
        f"{idx}. 《{getattr(track, 'title', '')}》 - {getattr(track, 'artist', '') or '未知'}（{getattr(track, 'source', 'local')}）"
        for idx, track in enumerate(tracks[: plan.target_count or 12], start=1)
    ]
    return intro + "\n" + "\n".join(lines)


def _compose_intro(
    query: str,
    tracks: list[Any],
    plan: AgentPlan,
    agent: AudioVisualAgent | None,
    memory_query: str,
    shortfall: bool,
    history_text: str = "",
) -> str:
    """生成推荐引言。LLM 可用时结合记忆+对话历史个性化，否则用确定性模板。"""
    fallback = f"我按在线优先策略整理了 {len(tracks)} 个可追溯候选："
    if shortfall:
        fallback += f"\n说明：你要求 {plan.target_count} 首，但当前候选只有 {len(tracks)} 首。"

    if agent is None or getattr(agent, "llm", None) is None:
        return fallback

    titles_preview = "、".join(getattr(t, "title", "") for t in tracks[:5])
    mem_hint = f"用户偏好：{memory_query[:150]}" if memory_query else "暂无明确偏好记录"
    history_hint = ""
    if history_text:
        # 只取最近几行避免过长
        recent_lines = history_text.strip().split("\n")[-6:]
        history_hint = "最近对话：\n" + "\n".join(recent_lines) + "\n"
    prompt = (
        f"{history_hint}"
        f"用户请求：{query}\n"
        f"我已找到 {len(tracks)} 首真实候选，前几首：{titles_preview}\n"
        f"{mem_hint}\n\n"
        "请写一句自然、有温度的推荐开场白（80字内），体现你理解了用户的需求、偏好和对话上下文。"
        "如果用户之前聊过相关话题，体现连贯性。"
        "不要列歌名，不要编造任何歌曲，不要用书名号。只输出这一句话。"
    )
    try:
        text = agent.llm.generate(prompt, temperature=settings.dialog_temperature).strip()
    except Exception:
        return fallback
    # LLM 兜底：异常输出（空/过长/含书名号疑似编造歌名）一律回退模板
    if not text or len(text) > 200 or "《" in text:
        return fallback
    if shortfall:
        text += f"\n说明：你要求 {plan.target_count} 首，但当前候选只有 {len(tracks)} 首。"
    return text


def _compose_chat_response(query: str, agent: AudioVisualAgent | None, history_text: str = "", user_id: str = "") -> str | None:
    """让 LLM 自然回复寒暄/闲聊，注入用户画像作为事实锚。"""
    if agent is None or getattr(agent, "llm", None) is None:
        return None
    history_hint = ""
    if history_text:
        recent_lines = history_text.strip().split("\n")[-6:]
        history_hint = "最近对话：\n" + "\n".join(recent_lines) + "\n"
    # 注入用户品味画像，让 chat 有事实基础（对齐 SoulTuner MUSIC_CHAT_RESPONSE_PROMPT 的 graphzep_facts 注入）
    taste_hint = ""
    if user_id and hasattr(agent, "summarize_taste"):
        try:
            taste = agent.summarize_taste(user_id)
            if taste and len(taste) > 10:
                taste_hint = f"用户音乐偏好：{taste[:150]}\n"
        except Exception:
            pass
    prompt = (
        f"{history_hint}"
        f"{taste_hint}"
        f"用户说：{query}\n\n"
        "你是用户的私人音乐搭子，用中文自然、友好地回复。"
        "如果用户只是在打招呼，友好回应并提示你可以帮他做什么音乐相关的事。"
        "如果用户提到了某个歌手/歌曲/风格，可以结合用户偏好简短聊聊你的看法。"
        "不要每次用同一句话，语气要自然口语化。50-100字。\n"
        "不要编造排名、发行时间、销量等你不确定的具体数据。"
    )
    try:
        text = agent.llm.generate(prompt, temperature=settings.dialog_temperature).strip()
    except Exception:
        return None
    if not text or len(text) > 300:
        return None
    return text


def _compose_discussion(
    query: str,
    tracks: list[Any],
    agent: AudioVisualAgent | None,
    history_text: str = "",
) -> str | None:
    """让 LLM 基于搜到的真实曲目讨论音乐话题。

    关键反幻觉策略（对齐 SoulTuner + 比它更严格）：
    - 没有搜到真实数据 → 直接拒绝回答，不编造
    - 搜到了 → LLM 只能基于真实曲目回答，不确定的说不知道
    - 限制 100 字（SoulTuner 的 general_chat 也是 100 字）
    """
    if agent is None or getattr(agent, "llm", None) is None:
        return None
    track_hint = ""
    if tracks:
        items = []
        for t in tracks[:10]:
            title = getattr(t, "title", "")
            artist = getattr(t, "artist", "") or ""
            items.append(f"《{title}》{artist}")
        track_hint = f"已搜到的真实曲目（网易云验证过）：{'、'.join(items)}\n"
    else:
        # 没有搜到真实数据 → 拒绝编造
        return "我暂时没找到关于这个话题的可靠音乐数据，不想编，怕误导你。你可以试试换个关键词再问我。"
    history_hint = ""
    if history_text:
        recent_lines = history_text.strip().split("\n")[-6:]
        history_hint = "最近对话：\n" + "\n".join(recent_lines) + "\n"
    prompt = (
        f"{history_hint}"
        f"{track_hint}"
        f"用户问：{query}\n\n"
        "请用中文自然地回答，像一个懂音乐的朋友在聊天。\n"
        "严格规则：\n"
        "1. 只讨论上面列出的真实曲目和你能确认的事实\n"
        "2. 不要编造专辑评价、歌曲细节、发行时间、排名、销量等你不确定的信息\n"
        "3. 不确定的就说\"我不太确定\"，不要猜测\n"
        "4. 不要推荐未在上面列出的歌曲——只讨论已验证的真实曲目\n"
        "5. 如果之前聊过相关话题，体现连贯性\n"
        "6. 100字以内\n"
        "7. 可以挑几首真实曲目推荐并说明理由"
    )
    try:
        text = agent.llm.generate(prompt, temperature=settings.dialog_temperature).strip()
    except Exception:
        return None
    if not text or len(text) > 300:
        return None
    return text


def _compose_video_answer(
    query: str,
    tracks: list[Any],
    agent: AudioVisualAgent | None,
    history_text: str = "",
) -> str | None:
    """video 意图专用回答：视频推荐开场白 + 视频列表。"""
    if not tracks:
        return None
    # 构建 LLM 开场白
    intro = "我帮你搜到了这些视频："
    if agent is not None and getattr(agent, "llm", None) is not None:
        titles_preview = "、".join(getattr(t, "title", "") for t in tracks[:5])
        history_hint = ""
        if history_text:
            recent_lines = history_text.strip().split("\n")[-4:]
            history_hint = "最近对话：\n" + "\n".join(recent_lines) + "\n"
        prompt = (
            f"{history_hint}"
            f"用户请求：{query}\n"
            f"我已找到 {len(tracks)} 个真实视频，前几个：{titles_preview}\n\n"
            "请写一句自然、有温度的视频推荐开场白（60字内），体现你理解了用户要看MV/现场/演唱会的需求。"
            "不要列视频名，不要编造信息，不要用书名号。只输出这一句话。"
        )
        try:
            text = agent.llm.generate(prompt, temperature=settings.dialog_temperature).strip()
            if text and len(text) <= 150 and "《" not in text:
                intro = text
        except Exception:
            pass
    # 视频列表（确定性拼接）
    lines = [
        f"{idx}. 《{getattr(track, 'title', '')}》 - {getattr(track, 'artist', '') or '未知'}"
        f"（{getattr(track, 'source', 'local')}）"
        for idx, track in enumerate(tracks[:10], start=1)
    ]
    return f"{intro}\n" + "\n".join(lines)


def _compose_artist_info_answer(
    query: str,
    results: list[dict[str, Any]],
    agent: AudioVisualAgent | None,
    history_text: str = "",
) -> str | None:
    """artist_info 意图专用回答：基于搜索引擎结果生成百科式介绍。"""
    search_results = next(
        (r.get("search_results", []) for r in results if r.get("type") == "web_info_search"),
        [],
    )
    if not search_results:
        # 搜索无结果，降级到 discuss 模式
        return None

    # 把搜索摘要拼接成上下文
    context_parts: list[str] = []
    for i, item in enumerate(search_results[:5], 1):
        title = item.get("title", "")
        content = item.get("content", "")
        if content:
            context_parts.append(f"[{i}] {title}\n{content}")
    search_context = "\n\n".join(context_parts)
    source_urls = [item["url"] for item in search_results if item.get("url")]

    if agent is None or getattr(agent, "llm", None) is None:
        # 无 LLM，直接输出搜索摘要
        return search_context

    history_hint = ""
    if history_text:
        recent_lines = history_text.strip().split("\n")[-4:]
        history_hint = "最近对话：\n" + "\n".join(recent_lines) + "\n"

    prompt = (
        f"{history_hint}"
        f"用户问：{query}\n\n"
        f"以下是搜索引擎返回的真实资料：\n{search_context}\n\n"
        "请用中文基于以上真实资料写一段介绍，像一个懂音乐的朋友在讲解（200-400字）。\n"
        "严格规则：\n"
        "1. 只使用上面列出的真实信息，不确定的说\"我不太确定\"\n"
        "2. 不要编造排名、销量、具体日期等未提及的数据\n"
        "3. 自然流畅，不要像百科词条那样枯燥\n"
        "4. 如果资料足够，可以涵盖：简介、成员、风格特点、代表作、影响力等\n"
        "5. 末尾附上参考来源链接"
    )
    try:
        text = agent.llm.generate(prompt, temperature=settings.dialog_temperature).strip()
    except Exception:
        return search_context
    if not text or len(text) > 800:
        return search_context
    # 追加来源链接
    if source_urls:
        text += "\n\n📎 参考来源：\n" + "\n".join(f"- {url}" for url in source_urls[:3])
    return text


def _compose_playlist_answer(
    query: str,
    results: list[dict[str, Any]],
    plan: AgentPlan,
    agent: AudioVisualAgent | None,
    memory_query: str,
    history_text: str,
) -> str:
    """歌单意图专用回答：保留歌单名/描述，只用 LLM 精选的 playlist 曲目，
    不混入原始搜索结果。

    compose_answer 的通用路径会 _collect_tracks(results) 把 web_music_search
    的原始结果和 playlist 的精选结果混在一起，且丢弃歌单名和描述。
    歌单场景需要区分：用户要的是「LLM 精心挑选的歌单」，不是「一堆搜索结果」。
    """
    playlist_result = next((r for r in results if r.get("type") == "playlist"), None)
    if not playlist_result:
        # 没有 playlist 结果，降级到通用路径
        tracks = _collect_tracks(results)
        if not tracks:
            return "这轮没有生成可追溯的歌单。"
        return "歌单生成未成功，以下是搜索到的候选：\n" + "\n".join(
            f"{i}. 《{getattr(t, 'title', '')}》 - {getattr(t, 'artist', '') or '未知'}"
            for i, t in enumerate(tracks[:12], 1)
        )

    pl = playlist_result["playlist"]
    tracks = pl.tracks
    if not tracks:
        return f"歌单《{pl.name}》生成了但暂无可追溯曲目。"

    # 引言：用 LLM 生成有温度的歌单介绍
    intro = _compose_intro(query, tracks, plan, agent, memory_query,
                           shortfall=bool(plan.target_count and len(tracks) < plan.target_count),
                           history_text=history_text)
    # 歌单元信息
    header = f"歌单《{pl.name}》"
    if pl.description:
        header += f"：{pl.description}"

    lines = [
        f"{idx}. 《{getattr(track, 'title', '')}》 - {getattr(track, 'artist', '') or '未知'}（{getattr(track, 'source', 'local')}）"
        for idx, track in enumerate(tracks[: plan.target_count or 30], start=1)
    ]
    return f"{intro}\n{header}\n" + "\n".join(lines)


def _is_smalltalk(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip().lower()).strip("。.!?！？，,")
    return normalized in {
        "hi",
        "hello",
        "hey",
        "你好",
        "嗨",
        "哈喽",
        "在吗",
        "早",
        "早上好",
        "晚上好",
    }
