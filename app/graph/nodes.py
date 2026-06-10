from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from app.context import ContextBudgetManager, ContextSource
from app.config import settings
from app.graph.tag_rules import extract_tags
from app.llm.structured import extract_json_dict
from app.models import AgentAnswer, AgentPlan, RetrievalPlan, StreamEvent
from app.prompts import QUERY_PLAN_SYSTEM
from app.react_loop import _goal_progress, guard_answer

if TYPE_CHECKING:
    from app.agent import AudioVisualAgent
    from app.graph.state import AgentState


def load_context(agent: AudioVisualAgent, state: AgentState) -> AgentState:
    memory = agent.memory.get_memory(state["user_id"])
    goal = agent.memory.get_active_goal(state["user_id"])
    memory_query = agent.memory.weighted_query(memory)

    # GSSC：按优先级把用户输入/记忆/历史压进 token 预算，产出追踪报告。
    history = state.get("history") or []
    history_text = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in history)
    sources = [
        ContextSource(name="user_query", content=state["query"], priority=0, min_tokens=200),
        ContextSource(name="memory", content=memory_query, priority=1, min_tokens=80),
        ContextSource(name="history", content=history_text, priority=2, min_tokens=40),
    ]
    budgeted, report = ContextBudgetManager(total_budget=2000).allocate(sources)

    context = {
        "memory_query": budgeted.get("memory", memory_query),
        "history_text": budgeted.get("history", ""),
        "active_goal": goal.model_dump(mode="json") if goal else None,
        "resource_count": len(agent.list_resource_tracks(50)),
        "budget_report": report.as_lines(),
    }
    return {
        **state,
        "context": context,
        "results": [],
        "trace": ["[load_context] 载入记忆、目标和资源库摘要。", *report.as_lines()],
        "events": [StreamEvent(type="plan", content="正在读取记忆和资源库状态。")],
    }


def plan_intent(agent: AudioVisualAgent, state: AgentState) -> AgentState:
    query = state["query"]
    history_text = (state.get("context") or {}).get("history_text", "")
    plan = plan_with_llm(agent, query, history_text) or build_agent_plan(query)
    return {
        **state,
        "plan": plan,
        "trace": [*state.get("trace", []), f"[plan] {plan.reasoning_summary}"],
        "events": [
            *state.get("events", []),
            StreamEvent(type="plan", content=plan.reasoning_summary, payload=plan.model_dump(mode="json")),
        ],
    }


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
    if tool == "web_music_search":
        tracks = agent.search_web_music(query, top_k=max(plan.target_count or top_k, top_k))
        for track in tracks:
            if hasattr(agent, "library"):
                agent.library.upsert_external(track)
        results.append({"type": "web_music_search", "tracks": tracks})
        trace.append(f"[web_music_search] 获取 {len(tracks)} 个线上/候选结果。")
        # 先吐候选卡片（对齐 SoulTuner __songs__ 先于解释文本的体验）。
        cards = [_song_card(t) for t in tracks]
        events.append(StreamEvent(type="candidates", content=f"候选 {len(tracks)} 个", payload={"count": len(tracks), "cards": cards}))
    elif tool == "recommend":
        rec = agent.recommend_for_query(user_id, query, top_k=top_k)
        results.append({"type": "daily_recommend", "recommendation": rec})
        trace.append(f"[recommend] 生成 {len(rec.tracks)} 首推荐。")
        cards = [_song_card(item.asset, reason=item.reason, score=item.score, components=item.components) for item in rec.tracks]
        events.append(StreamEvent(type="candidates", content=f"推荐 {len(cards)} 首", payload={"count": len(cards), "cards": cards}))
    elif tool == "playlist":
        seed_tracks = _collect_tracks(results)
        playlist = agent.generate_playlist(user_id, query, seed_tracks=seed_tracks, target_count=plan.target_count)
        results.append({"type": "playlist", "playlist": playlist})
        trace.append(f"[playlist] 生成 {len(playlist.tracks)} 首歌单。")
        cards = [_song_card(t) for t in playlist.tracks]
        events.append(StreamEvent(type="candidates", content=f"歌单 {len(cards)} 首", payload={"count": len(cards), "cards": cards}))
    elif tool == "search":
        search = agent.search(user_id, query, include_external=True, top_k=max(top_k, 12))
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


def finalize(agent: AudioVisualAgent, state: AgentState) -> AgentState:
    memory_query = (state.get("context") or {}).get("memory_query", "")
    answer_text = compose_answer(
        state["query"], state.get("results", []), state["plan"],
        agent=agent, memory_query=memory_query,
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
    answer = AgentAnswer(
        answer=answer_text,
        evidences=[],
        memory_updated=memory_updated,
        agent_trace=trace,
        pending_goal=goal.goal if goal and goal.status == "active" else None,
        goal_progress=_goal_progress(goal),
    )
    return {
        **state,
        "answer": answer,
        "trace": trace,
        "events": [*state.get("events", []), StreamEvent(type="final", content=answer_text, payload=answer.model_dump(mode="json"))],
    }


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
    if not data or data.get("intent") not in _VALID_INTENTS:
        return None

    intent = data["intent"]
    entities = [e for e in (data.get("entities") or []) if isinstance(e, str) and e.strip()]
    use_local = bool(data.get("use_local", True))
    use_vector = bool(data.get("use_vector", False))
    use_web = bool(data.get("use_web", intent not in {"taste", "chat"}))
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
        strategy=_strategy_for(intent, use_web),
        tools_needed=_tools_for_intent(intent, use_web),
        target_count=target,
        online_required=use_web,
        reasoning_summary=(data.get("reasoning") or "").strip() or _default_summary(intent),
        retrieval_plan=retrieval,
    )


_VALID_INTENTS = {"recommend", "search", "playlist", "taste", "import", "journey", "discuss", "chat"}


def _tools_for_intent(intent: str, use_web: bool) -> list[str]:
    if intent == "journey":
        return ["journey"]
    if intent == "taste":
        return ["taste"]
    if intent == "chat":
        return []
    if intent == "discuss":
        return ["web_music_search"] if use_web else []
    if intent == "import":
        return ["import"]
    if intent == "playlist":
        return (["web_music_search"] if use_web else []) + ["playlist"]
    if intent == "search":
        return (["web_music_search"] if use_web else []) + ["search"]
    # recommend
    return (["web_music_search"] if use_web else []) + ["recommend"]


def _strategy_for(intent: str, use_web: bool) -> str:
    if intent in {"taste"}:
        return "memory_only"
    if intent == "chat":
        return "no_search"
    if intent == "discuss":
        return "online_first" if use_web else "no_search"
    return "online_first" if use_web else "library_first"


def _default_summary(intent: str) -> str:
    return {
        "recommend": "用户要推荐音乐，优先获取真实线上候选，再结合记忆排序。",
        "search": "用户要找歌，优先搜索真实平台候选，再补充本地库命中。",
        "playlist": "用户要生成歌单，先联网扩展真实候选，再生成可追溯歌单。",
        "taste": "用户要分析品味，只读取记忆和行为画像。",
        "journey": "用户需要多阶段音乐编排，使用音乐旅程节点分段检索和解释。",
        "import": "用户要导入网易云歌单作为推荐输入。",
        "chat": "普通对话，不需要检索音乐候选。",
    }.get(intent, "推进用户的音乐目标。")


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
    lowered = query.lower()
    target = _infer_count(query)
    if _is_smalltalk(query):
        return AgentPlan(
            intent="chat",
            strategy="no_search",
            tools_needed=[],
            online_required=False,
            reasoning_summary="这是普通寒暄，不需要联网搜索或音乐候选。",
        )
    if any(token in lowered or token in query for token in ["旅程", "热身", "冲刺", "journey"]):
        return AgentPlan(
            intent="journey",
            tools_needed=["journey"],
            target_count=target,
            reasoning_summary="用户需要多阶段音乐编排，使用音乐旅程节点分段检索和解释。",
        )
    if "导入" in query and ("歌单" in query or "playlist" in lowered):
        return AgentPlan(
            intent="import",
            tools_needed=["import", "recommend"] if any(t in query for t in ["推荐", "挑", "适合"]) else ["import"],
            target_count=target,
            reasoning_summary="用户要导入网易云歌单作为推荐输入。",
            retrieval_plan=_keyword_retrieval_plan(query, use_web=True),
        )
    if any(token in lowered or token in query for token in ["歌单", "playlist", "合集"]):
        return AgentPlan(
            intent="playlist",
            tools_needed=["web_music_search", "playlist"],
            target_count=target,
            reasoning_summary="用户要生成歌单，先联网扩展真实候选，再生成可追溯歌单。",
            retrieval_plan=_keyword_retrieval_plan(query, use_web=True),
        )
    if any(token in lowered or token in query for token in ["搜索", "找歌", "search", "联网", "真实", "最新"]):
        return AgentPlan(
            intent="search",
            tools_needed=["web_music_search", "search"],
            target_count=target,
            reasoning_summary="用户要找歌，优先搜索真实平台候选，再补充本地库命中。",
        )
    if any(token in lowered or token in query for token in ["品味", "分析我", "taste"]):
        return AgentPlan(
            intent="taste",
            strategy="memory_only",
            tools_needed=["taste"],
            online_required=False,
            reasoning_summary="用户要分析品味，只需要读取记忆和行为画像。",
        )
    if any(token in lowered or token in query for token in ["推荐", "适合", "recommend", "chill"]):
        return AgentPlan(
            intent="recommend",
            tools_needed=["web_music_search", "recommend"],
            target_count=target,
            reasoning_summary="用户要推荐音乐，优先获取真实线上候选，再结合记忆排序。",
        )
    # 音乐讨论/知识类问题：搜歌手真实曲目作为论据，再让 LLM 讨论
    _DISCUSS_KEYWORDS = [
        "牛逼", "怎么样", "评价", "介绍", "背景", "风格是", "什么水平", "好听吗",
        "厉害", "经典", "代表", "值得听", "有什么歌", "有哪些歌", "成名曲",
        "特色", "曲风", "地位", "影响", "怎么样", "如何看", "聊聊",
        "和 ", " vs ", "对比", "谁的", "专辑", "出道", "代表作",
    ]
    if any(kw in query for kw in _DISCUSS_KEYWORDS):
        return AgentPlan(
            intent="discuss",
            tools_needed=["web_music_search"],
            online_required=True,
            reasoning_summary="音乐讨论，联网搜歌手真实曲目作为讨论论据。",
            retrieval_plan=_keyword_retrieval_plan(query, use_web=True),
        )
    return AgentPlan(
        intent="chat",
        strategy="no_search",
        tools_needed=[],
        online_required=False,
        reasoning_summary="这是普通对话，不需要检索音乐候选。",
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
    }
    return [mapping.get(result.get("type", ""), result.get("type", "")) for result in results]


def compose_answer(
    query: str,
    results: list[dict[str, Any]],
    plan: AgentPlan,
    agent: AudioVisualAgent | None = None,
    memory_query: str = "",
) -> str:
    if plan.intent == "chat":
        return _compose_chat_response(query, agent) or "你好，我在。有什么音乐上的事可以帮你?"
    if plan.intent == "discuss":
        tracks = _collect_tracks(results)
        return _compose_discussion(query, tracks, agent) or "抱歉，我暂时无法讨论这个话题。"
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
    tracks = _collect_tracks(results)
    if not tracks:
        return "这轮没有拿到可追溯的音乐候选；我不会用未核实歌名硬凑结果。"
    if plan.target_count:
        tracks = tracks[: plan.target_count]
    shortfall = bool(plan.target_count and len(tracks) < plan.target_count)
    # 引言：优先让 LLM 结合记忆生成有温度的开场；失败回退确定性模板。
    # 歌曲清单始终由真实候选确定性拼接——LLM 绝不参与歌名生成（防幻觉）。
    intro = _compose_intro(query, tracks, plan, agent, memory_query, shortfall)
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
) -> str:
    """生成推荐引言。LLM 可用时结合记忆个性化，否则用确定性模板。"""
    fallback = f"我按在线优先策略整理了 {len(tracks)} 个可追溯候选："
    if shortfall:
        fallback += f"\n说明：你要求 {plan.target_count} 首，但当前候选只有 {len(tracks)} 首。"

    if agent is None or getattr(agent, "llm", None) is None:
        return fallback

    titles_preview = "、".join(getattr(t, "title", "") for t in tracks[:5])
    mem_hint = f"用户偏好：{memory_query[:150]}" if memory_query else "暂无明确偏好记录"
    prompt = (
        f"用户请求：{query}\n"
        f"我已找到 {len(tracks)} 首真实候选，前几首：{titles_preview}\n"
        f"{mem_hint}\n\n"
        "请写一句自然、简短（40字内）的推荐开场白，体现你理解了用户的需求和偏好。"
        "不要列歌名，不要编造任何歌曲，不要用书名号。只输出这一句话。"
    )
    try:
        text = agent.llm.generate(prompt, temperature=settings.dialog_temperature).strip()
    except Exception:
        return fallback
    # LLM 兜底：异常输出（空/过长/含书名号疑似编造歌名）一律回退模板
    if not text or len(text) > 120 or "《" in text:
        return fallback
    if shortfall:
        text += f"\n说明：你要求 {plan.target_count} 首，但当前候选只有 {len(tracks)} 首。"
    return text


def _compose_chat_response(query: str, agent: AudioVisualAgent | None) -> str | None:
    """让 LLM 自然回复寒暄/闲聊，而非返回硬编码模板。"""
    if agent is None or getattr(agent, "llm", None) is None:
        return None
    prompt = (
        f"用户说：{query}\n\n"
        "你是用户的私人音乐搭子，用中文自然简短地回复。"
        "如果用户只是在打招呼，友好回应并提示你可以帮他做什么音乐相关的事。"
        "不要每次用同一句话。20字以内。"
    )
    try:
        text = agent.llm.generate(prompt, temperature=settings.dialog_temperature).strip()
    except Exception:
        return None
    if not text or len(text) > 200:
        return None
    return text


def _compose_discussion(
    query: str,
    tracks: list[Any],
    agent: AudioVisualAgent | None,
) -> str | None:
    """让 LLM 基于搜到的真实曲目 + 自身知识自然讨论音乐话题。"""
    if agent is None or getattr(agent, "llm", None) is None:
        return None
    track_hint = ""
    if tracks:
        items = []
        for t in tracks[:8]:
            title = getattr(t, "title", "")
            artist = getattr(t, "artist", "") or ""
            items.append(f"《{title}》{artist}")
        track_hint = f"已搜到该歌手/专辑的真实曲目：{'、'.join(items)}\n"
    prompt = (
        f"{track_hint}"
        f"用户问：{query}\n\n"
        "请用中文自然地回答，像一个懂音乐的朋友在聊天。"
        "可以讲风格特点、代表作品、在音乐圈的地位、适合听什么场景等。"
        "如果搜到了真实曲目，可以挑几首重点推荐并说明理由。"
        "不要编造不存在的歌名。40-200字。"
    )
    try:
        text = agent.llm.generate(prompt, temperature=settings.dialog_temperature).strip()
    except Exception:
        return None
    if not text or len(text) > 500:
        return None
    return text


def collect_known_titles(results: list[dict[str, Any]]) -> set[str]:
    return {getattr(track, "title", "") for track in _collect_tracks(results) if getattr(track, "title", "")}


def _song_card(track: Any, reason: str = "", score: float | None = None, components: dict | None = None) -> dict[str, Any]:
    """把一个 track 压成前端歌曲卡片所需的精简字段。"""
    return {
        "title": getattr(track, "title", ""),
        "artist": getattr(track, "artist", "") or "未知",
        "source": getattr(track, "source", "local"),
        "playback_url": getattr(track, "playback_url", None) or getattr(track, "source_url", None),
        "genre": getattr(track, "genre", []) or [],
        "mood": getattr(track, "mood", []) or [],
        "reason": reason,
        "score": score,
        "components": components or {},
    }


def _collect_tracks(results: list[dict[str, Any]]) -> list[Any]:
    tracks: list[Any] = []
    for result in results:
        t = result.get("type")
        if t == "web_music_search":
            tracks.extend(result["tracks"])
        elif t == "daily_recommend":
            tracks.extend(item.asset for item in result["recommendation"].tracks)
        elif t == "playlist":
            tracks.extend(result["playlist"].tracks)
        elif t == "search":
            tracks.extend(result["response"].external)
            tracks.extend(result["response"].local)
        elif t == "import_netease_playlist":
            tracks.extend(result["result"].get("tracks", []))
    seen: set[str] = set()
    unique: list[Any] = []
    for track in tracks:
        key = f"{getattr(track, 'source', 'local')}|{getattr(track, 'title', '').lower()}|{getattr(track, 'artist', '') or ''}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(track)
    return unique


def _infer_count(text: str) -> int | None:
    match = re.search(r"(\d{1,3})\s*(?:首|个|tracks?|songs?)?", text, re.IGNORECASE)
    if not match:
        return None
    return max(1, min(int(match.group(1)), 100))


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
