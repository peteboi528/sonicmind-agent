from __future__ import annotations

import logging
import re
import time
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
from app.answer import (
    track_ref_from_card as _track_ref_from_card,
)
from app.config import settings
from app.context import ContextBudgetManager, ContextSource
from app.graph.tag_rules import extract_tags
from app.intents import (
    expand_content_negation,
    extract_content_negations,
    get_intent,
    is_continuation,
    match_intent_by_keywords,
)
from app.llm.observability import (
    capture_llm_stats,
    empty_runtime_metrics,
    format_runtime_metrics,
    merge_runtime_metrics,
)
from app.llm.routing import select_llm
from app.llm.structured import extract_json_dict
from app.models import (
    AgentAnswer,
    AgentPlan,
    ExternalTrack,
    QueryPlanPayload,
    RecoveryDecision,
    RetrievalPlan,
    StreamEvent,
    ToolOutcome,
    ToolStage,
)
from app.prompts import QUERY_PLAN_SYSTEM, QUERY_PLAN_VERSION
from app.prompts.reflect import (
    CANDIDATE_REFLECTION_SYSTEM,
    CANDIDATE_REFLECTION_USER,
    CANDIDATE_REFLECTION_VERSION,
)
from app.tools.contracts import ToolCall, ToolError, ToolResult, ToolRisk, ToolStatus
from app.tools.handlers import _apply_language_filter  # noqa: F401  # re-export: tests/test_query_rewrite 从本模块导入
from app.tools.registry import get_handler, get_tool_spec

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.agent import AudioVisualAgent
    from app.graph.state import AgentState


def load_context(agent: AudioVisualAgent, state: AgentState) -> AgentState:
    memory = agent.memory.get_memory(state["user_id"])
    goal = agent.memory.get_active_goal(state["user_id"])
    memory_query = agent.memory.weighted_query(memory, include_artists=False)
    dialogue = agent.memory.get_dialogue_state(state["user_id"])

    recall_lines: list[str] = []
    profile = (memory.consolidated_profile or "").strip()
    memory_parts = [p for p in [memory_query, profile] if p]
    enriched_memory = " ".join(memory_parts)
    if profile:
        recall_lines.append(f"巩固画像：{profile}")

    # GSSC：按优先级把用户输入/记忆/历史压进 token 预算，产出追踪报告。
    history = state.get("history") or []
    history_text = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in history)
    sources = [
        ContextSource(name="user_query", content=state["query"], priority=0, min_tokens=200),
        ContextSource(name="memory", content=enriched_memory, priority=1, min_tokens=80),
        ContextSource(
            name="history",
            content=history_text,
            priority=2,
            min_tokens=40,
            preserve_tail=True,
        ),
    ]
    budgeted, report = ContextBudgetManager(total_budget=2000).allocate(sources)

    context = {
        "memory_query": budgeted.get("memory", enriched_memory),
        "history_text": budgeted.get("history", ""),
        "active_goal": goal.model_dump(mode="json") if goal else None,
        "resource_count": len(agent.list_resource_tracks(50)),
        "budget_report": report.as_lines(),
        "dialogue_state": dialogue.model_dump(mode="json"),
        "prompt_versions": {},
        "runtime_metrics": empty_runtime_metrics(),
        "semantic_recall_pending": True,
        "started_at_monotonic": time.monotonic(),
    }
    return {
        **state,
        "context": context,
        "results": [],
        "tool_outcomes": [],
        "trace": ["[load_context] 载入记忆、目标和资源库摘要。", *recall_lines, *report.as_lines()],
        "events": [StreamEvent(type="plan", content="正在读取记忆和资源库状态。")],
    }


async def plan_intent_async(agent: AudioVisualAgent, state: AgentState) -> AgentState:
    context = state.get("context") or {}
    plan, prompt_versions, runtime_metrics = await plan_with_llm_with_meta_async(
        agent,
        state["query"],
        context.get("history_text", ""),
        context.get("memory_query", ""),
    )
    return _finish_plan_intent(agent, state, plan, prompt_versions, runtime_metrics)


def _finish_plan_intent(
    agent: AudioVisualAgent,
    state: AgentState,
    plan: AgentPlan | None,
    prompt_versions: dict[str, str],
    runtime_metrics: dict[str, float | int],
) -> AgentState:
    query = state["query"]
    context = state.get("context") or {}
    plan = plan or build_agent_plan(query)
    plan = _sanitize_retrieval_entities(plan)
    # 明确的旅程信号由确定性意图兜底，不能被 LLM 偶发误判成普通学习推荐。
    if match_intent_by_keywords(query) == "journey" and plan.intent != "journey":
        plan = build_agent_plan(query)
    if match_intent_by_keywords(query) == "similar_artists" and plan.intent != "similar_artists":
        spec = get_intent("similar_artists")
        plan = plan.model_copy(update={
            "intent": "similar_artists",
            "strategy": spec.strategy_for(False),
            "tools_needed": spec.tools_for(False),
            "online_required": False,
            "reasoning_summary": spec.summary,
        })
    keyword_intent = match_intent_by_keywords(query)
    if keyword_intent and _is_knowledge_intent(keyword_intent) and plan.intent != keyword_intent:
        spec = get_intent(keyword_intent)
        plan = plan.model_copy(update={
            "intent": keyword_intent,
            "strategy": spec.strategy_for(True),
            "tools_needed": spec.tools_for(True),
            "online_required": True,
            "reasoning_summary": spec.summary,
        })
    plan, inherited = _apply_dialogue_continuation(plan, query, context.get("dialogue_state"))
    # 安全网：LLM 可能把介绍/百科类问题误判为 discuss，检查关键词自动升级
    plan = _upgrade_artist_info(plan, query)
    state = _attach_semantic_recall_if_needed(agent, state, plan)
    context = state.get("context") or {}
    plan, memory_seeds = _inject_preference_seeds(plan, query, context)
    plan = _materialize_tool_stages(plan, query, state.get("top_k", 5))
    context = dict(state.get("context") or {})
    if _is_knowledge_intent(plan.intent):
        from app.knowledge import knowledge_deadline

        context["deadline_at"] = knowledge_deadline()
        context["latency_budget"] = {
            "kind": "knowledge",
            "budget_seconds": settings.knowledge_turn_budget_seconds,
            "timed_out_tools": [],
            "skipped_due_to_deadline": [],
            "partial": False,
        }
    trace_line = f"[plan] {plan.reasoning_summary}"
    if inherited:
        trace_line += f"（延续上一轮：继承 {inherited}）"
    if memory_seeds:
        trace_line += f"（记忆检索种子：{'、'.join(memory_seeds)}）"
    if prompt_versions:
        trace_line += f" [prompt] {_format_prompt_versions(prompt_versions)}"
    return {
        **state,
        "plan": plan,
        "context": {
            **context,
            "prompt_versions": _merge_prompt_versions(context.get("prompt_versions"), prompt_versions),
            "runtime_metrics": runtime_metrics,
        },
        "trace": [*state.get("trace", []), trace_line],
        "events": [
            *state.get("events", []),
            StreamEvent(type="plan", content=plan.reasoning_summary, payload=plan.model_dump(mode="json")),
        ],
    }


_GENERIC_ENTITY_WORDS = {
    "随便", "好听", "歌曲", "音乐", "推荐", "来点", "来几首", "一些",
    "深夜", "放松", "专注", "学习", "工作", "跑步", "睡前", "chill",
    # 延续/反重复指令词不是音乐实体（mock 或 LLM 偶把"再来"抽成 entity）。
    "再来", "再来几首", "再来点", "再来些", "换一批", "换几首", "多来", "多来几首",
    "继续", "更多", "还要", "还想", "再推", "再给",
}
_ENTITY_REQUEST_SIGNALS = ("来点", "来些", "推荐", "适合", "搜索", "搜一下", "找点", "歌曲", "音乐", "再来", "换一批", "多来")


def _sanitize_retrieval_entities(plan: AgentPlan) -> AgentPlan:
    entities: list[str] = []
    for item in plan.retrieval_plan.entities:
        value = str(item or "").strip()
        key = _constraint_key(value)
        tags = extract_tags(value)
        is_request_phrase = any(signal in value.lower() for signal in _ENTITY_REQUEST_SIGNALS)
        is_pure_tag = bool(tags["genre"] or tags["mood"] or tags["scenario"]) and len(value) <= 12
        if not value or key in {_constraint_key(word) for word in _GENERIC_ENTITY_WORDS}:
            continue
        if is_request_phrase or is_pure_tag:
            continue
        entities.append(value)
    if entities == plan.retrieval_plan.entities:
        return plan
    return plan.model_copy(update={
        "retrieval_plan": plan.retrieval_plan.model_copy(update={"entities": entities}),
    })


def _inject_preference_seeds(
    plan: AgentPlan,
    query: str,
    context: dict[str, Any],
) -> tuple[AgentPlan, list[str]]:
    """Add low-specificity taste tags to broad retrieval without creating hard artist anchors."""
    if plan.intent not in {"recommend", "playlist"}:
        return plan, []
    retrieval = plan.retrieval_plan
    query_tags = extract_tags(query)
    constraints = extract_content_negations(query)
    explicit_genres = _without_negative_constraints(query_tags["genre"], constraints)
    # An explicit artist/song or genre is authoritative; memory still participates in ranking,
    # but must not redirect the retrieval query.
    if retrieval.entities or explicit_genres:
        return plan, []

    memory_text = str(context.get("memory_query") or "").strip()
    if not memory_text:
        return plan, []
    memory_tags = extract_tags(memory_text)
    blocked = {_constraint_key(item) for item in constraints}
    seeds: list[str] = []

    def add(values: list[str], *, allowed: bool) -> None:
        if not allowed:
            return
        explicit = [value for value in values if value.lower() in memory_text.lower()]
        for value in [*explicit, *values]:
            key = _constraint_key(value)
            if key and key not in blocked and value not in seeds:
                seeds.append(value)
                break

    add(memory_tags["genre"], allowed=True)
    add(memory_tags["mood"], allowed=not query_tags["mood"])
    seeds = seeds[:2]
    if not seeds:
        return plan, []

    base = _strip_negative_query(retrieval.search_query or query, constraints)
    parts = [base, *seeds]
    merged: list[str] = []
    seen: set[str] = set()
    for value in parts:
        value = str(value or "").strip()
        key = _constraint_key(value)
        if value and key and key not in seen:
            merged.append(value)
            seen.add(key)
    revised_query = " ".join(merged).strip()
    revised = retrieval.model_copy(update={
        "use_vector": retrieval.use_vector or bool(seeds),
        "search_query": revised_query,
        "search_variants": _cap_search_variants([*retrieval.search_variants, " ".join(seeds)], revised_query),
    })
    return plan.model_copy(update={"retrieval_plan": revised}), seeds


def _materialize_tool_stages(plan: AgentPlan, query: str, top_k: int) -> AgentPlan:
    """Turn compatibility tool names into explicit calls and dependency stages."""
    if plan.intent == "sample_lookup":
        resolve = ToolCall(name="resolve_music_entity", arguments=_planned_arguments("resolve_music_entity", query, plan, top_k))
        search = ToolCall(name="sample_relation_search", arguments=_planned_arguments("sample_relation_search", query, plan, top_k))
        locate = ToolCall(name="locate_sample_sources", arguments=_planned_arguments("locate_sample_sources", query, plan, top_k))
        build = ToolCall(name="build_sample_dossier", arguments=_planned_arguments("build_sample_dossier", query, plan, top_k))
        return plan.model_copy(update={
            "tools_needed": ["resolve_music_entity", "sample_relation_search", "locate_sample_sources", "build_sample_dossier"],
            "stages": [
                ToolStage(calls=[resolve], parallel=False),
                ToolStage(calls=[search], parallel=False),
                ToolStage(calls=[locate], parallel=False),
                ToolStage(calls=[build], parallel=False),
            ],
        })
    if _is_knowledge_intent(plan.intent):
        resolve = ToolCall(name="resolve_music_entity", arguments=_planned_arguments("resolve_music_entity", query, plan, top_k))
        metadata = ToolCall(name="music_metadata_lookup", arguments=_planned_arguments("music_metadata_lookup", query, plan, top_k))
        reviews = ToolCall(name="review_search", arguments=_planned_arguments("review_search", query, plan, top_k))
        build = ToolCall(name="build_music_dossier", arguments=_planned_arguments("build_music_dossier", query, plan, top_k))
        return plan.model_copy(update={
            "tools_needed": ["resolve_music_entity", "music_metadata_lookup", "review_search", "build_music_dossier"],
            "stages": [
                ToolStage(calls=[resolve], parallel=False),
                ToolStage(calls=[metadata, reviews], parallel=True),
                ToolStage(calls=[build], parallel=False),
            ],
        })
    stages: list[ToolStage] = []
    parallel_calls: list[ToolCall] = []
    previous: list[str] = []

    def flush() -> None:
        nonlocal parallel_calls
        if parallel_calls:
            stages.append(ToolStage(calls=parallel_calls, parallel=len(parallel_calls) > 1))
            parallel_calls = []

    for name in plan.tools_needed:
        call = ToolCall(name=name, arguments=_planned_arguments(name, query, plan, top_k))
        if _tool_depends_on_prior_results(name, previous):
            flush()
            stages.append(ToolStage(calls=[call], parallel=False))
        else:
            parallel_calls.append(call)
        previous.append(name)
    flush()
    return plan.model_copy(update={"stages": stages})


def _planned_arguments(name: str, query: str, plan: AgentPlan, top_k: int) -> dict[str, Any]:
    handler = get_handler(name) or name
    count = plan.target_count or top_k
    entity_query = _query_with_entities(query, plan)
    if handler == "recommend":
        return {"query": entity_query, "top_k": count}
    if handler == "search":
        return {"query": entity_query, "include_external": True}
    if handler == "web_music_search":
        return {"query": entity_query, "top_k": count}
    if handler == "playlist":
        return {"instruction": entity_query, "target_count": plan.target_count}
    if handler == "taste_experiment":
        return {"prompt": query, "total": plan.target_count or 12}
    if handler in {
        "resolve_music_entity", "music_metadata_lookup", "review_search", "build_music_dossier",
        "sample_relation_search", "locate_sample_sources", "build_sample_dossier",
    }:
        # 知识/对比工具必须保留用户原句；LLM 改写后的 search_query 可能把
        # “A 和 B 的区别”压成一个搜索串，导致实体切分失败。
        return {"query": query, "intent": plan.intent}
    if handler in {"artist_albums", "video_search", "web_info_search"}:
        return {"query": plan.retrieval_plan.entities[0] if handler == "artist_albums" and plan.retrieval_plan.entities else entity_query}
    if handler == "similar_artists":
        artist = plan.retrieval_plan.entities[0] if plan.retrieval_plan.entities else ""
        return {"artist": artist, "top_k": plan.target_count or 6}
    if handler == "import_netease_playlist":
        return {"playlist_ref": query, "limit": plan.target_count or 100}
    if handler in {
        "feedback", "listening_history", "list_my_playlists", "find_on_platform",
        "lyrics", "audio_features", "save_to_playlist", "favorite_track", "concert_events",
    }:
        return _infer_aux_arguments(handler, query, plan)
    return {}


def _is_knowledge_intent(intent: str) -> bool:
    try:
        from app.knowledge import is_knowledge_intent

        return is_knowledge_intent(intent)
    except Exception:
        return False


def _attach_semantic_recall_if_needed(agent: AudioVisualAgent, state: AgentState, plan: AgentPlan) -> AgentState:
    context = dict(state.get("context") or {})
    if not context.get("semantic_recall_pending", False):
        return state
    context["semantic_recall_pending"] = False
    if plan.intent == "chat":
        return {
            **state,
            "context": context,
            "trace": [*state.get("trace", []), "[load_context] chat 意图跳过跨会语义召回。"],
        }
    try:
        recalled = agent.memory.recall_episodes(state["user_id"], state["query"])
    except Exception:
        recalled = []
        logger.debug("plan_intent: 语义召回失败，跳过", exc_info=True)
    if not recalled:
        return {**state, "context": context}
    context["memory_query"] = " ".join(p for p in [context.get("memory_query", ""), *recalled] if p)
    return {
        **state,
        "context": context,
        "trace": [*state.get("trace", []), f"语义召回 {len(recalled)} 条相关历史偏好。"],
    }


def _apply_dialogue_continuation(
    plan: AgentPlan,
    query: str,
    dialogue_state: dict[str, Any] | None,
) -> tuple[AgentPlan, str]:
    """延续指令（再来几首/换一批/不要重复）程序化继承上一轮实体与标签，并挂跨轮去重排除集。

    仅当本轮被判为延续、且本轮没抽到自己的实体时才继承实体；返回 (新计划, 继承说明)。
    话题切换（本轮自带新实体/明确意图）不继承实体——交由 finalize 时覆盖旧状态。

    关键：跨轮去重的排除集（_excluded_tracks）与"是否自带实体"解耦——只要本轮是
    延续指令就挂上。否则用户重提歌手名（"The Weeknd 再来几首"或再次搜同名）时，
    rp.entities 非空会让这里整体提前 return，排除集永不挂，去重失效，netease 永远
    返回同一批 top-N。
    """
    if not dialogue_state or not is_continuation(query):
        return plan, ""

    def _mount_excluded(p: AgentPlan) -> AgentPlan:
        prev_shown = dialogue_state.get("shown_tracks") or []
        if prev_shown:
            p._excluded_tracks = prev_shown  # type: ignore[attr-defined]
        return p

    rp = plan.retrieval_plan
    negative_constraints = extract_content_negations(query)
    current_entities = _without_negative_constraints(rp.entities, negative_constraints)
    if plan.intent == "similar_artists":
        generic_refs = {"同类型", "同风格", "类似", "相似", "同类", "相似歌手", "类似歌手"}
        current_entities = [item for item in current_entities if _constraint_key(item) not in generic_refs]
    # 本轮正向实体代表话题切换；但“不要越南”里被 LLM 抽出的“越南”不是正向实体。
    if current_entities:
        if current_entities != rp.entities:
            plan = plan.model_copy(update={
                "retrieval_plan": rp.model_copy(update={"entities": current_entities}),
            })
        return _mount_excluded(plan), ""

    prev_entities = _without_negative_constraints(dialogue_state.get("entities") or [], negative_constraints)
    prev_intent = dialogue_state.get("last_intent") or plan.intent
    if not prev_entities and prev_intent in {"chat"}:
        return _mount_excluded(plan), ""

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
        # LLM 改写为空、仍含否定或丢失前文种子时，程序化合并上一轮正向上下文。
        search_query=_continuation_search_query(rp, dialogue_state, negative_constraints, prev_entities),
        search_variants=_continuation_search_variants(rp.search_variants, negative_constraints),
        language_filter=_continuation_language_filter(rp.language_filter, negative_constraints),
        excluded_terms=list(dict.fromkeys([*rp.excluded_terms, *negative_constraints])),
    )
    new_plan = plan.model_copy(update={
        "intent": intent,
        "strategy": spec.strategy_for(use_web),
        "tools_needed": spec.tools_for(use_web),
        "online_required": use_web,
        "retrieval_plan": merged,
    })
    return _mount_excluded(new_plan), "、".join(prev_entities) or prev_intent


def _constraint_key(value: str) -> str:
    return re.sub(r"[^a-z0-9一-龥]+", "", value.lower())


def _without_negative_constraints(values: list[str], constraints: list[str]) -> list[str]:
    blocked = [
        _constraint_key(alias)
        for item in constraints
        for alias in expand_content_negation(item)
        if _constraint_key(alias)
    ]
    if not blocked:
        return list(values)
    return [
        value for value in values
        if not any(block in _constraint_key(value) or _constraint_key(value) in block for block in blocked)
    ]


def _strip_negative_query(text: str, constraints: list[str]) -> str:
    cleaned = re.sub(
        r"(?:不要|别放|别推|别推荐|不想听|排除|避开|去掉|no\s+|without\s+|exclude\s+|avoid\s+)"
        r"[^，。,.!?！？]*",
        " ", text, flags=re.IGNORECASE,
    )
    for constraint in constraints:
        for alias in sorted(expand_content_negation(constraint), key=len, reverse=True):
            cleaned = re.sub(re.escape(alias), " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?:帮我|给我|推荐|来点|来些|找点|想听|几首|一些)", " ", cleaned)
    cleaned = re.sub(r"(?:歌曲|音乐)", " ", cleaned)
    return " ".join(cleaned.split()).strip()


def _continuation_search_query(
    retrieval: RetrievalPlan,
    dialogue_state: dict[str, Any],
    constraints: list[str],
    inherited_entities: list[str],
) -> str:
    llm_query = _strip_negative_query(retrieval.search_query, constraints)
    inherited = [
        *inherited_entities,
        *(dialogue_state.get("genre_tags") or []),
        *(dialogue_state.get("mood_tags") or []),
        *(dialogue_state.get("scenario_tags") or []),
    ]
    prior_query = _strip_negative_query(str(dialogue_state.get("last_query") or ""), constraints)
    parts = [llm_query, *inherited]
    if prior_query and not llm_query and not any(str(item).strip() for item in inherited):
        parts.append(prior_query)
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        part = part.strip()
        key = _constraint_key(part)
        if not part or not key or key in seen:
            continue
        seen.add(key)
        deduped.append(part)
    return " ".join(deduped).strip() or "音乐"


def _continuation_search_variants(variants: list[str], constraints: list[str]) -> list[str]:
    cleaned = [_strip_negative_query(item, constraints) for item in variants]
    return list(dict.fromkeys(item for item in cleaned if item))


def _continuation_language_filter(current: str, constraints: list[str]) -> str:
    if current:
        return current
    normalized = " ".join(constraints).lower()
    if any(token in normalized for token in ("中文", "华语", "国语", "chinese")):
        return "en"
    if any(token in normalized for token in ("英文", "英语", "欧美", "english")):
        return "zh"
    return ""


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


def _tool_depends_on_prior_results(tool: str, prior_tools: list[str] | None = None) -> bool:
    handler = get_handler(tool)
    if handler == "playlist":
        return True
    # 推荐消费前置 web 候选并做来源平衡；若并行执行，12 秒批次超时可能静默丢掉
    # 较慢的 recommend，只剩一个空的 web_music_search 结果。
    return handler == "recommend" and any(
        get_handler(previous) in {"web_music_search", "import_netease_playlist"}
        for previous in (prior_tools or [])
    )


async def web_fallback_async(agent: AudioVisualAgent, state: AgentState) -> AgentState:
    plan = state["plan"]
    query = state["query"]
    user_id = state["user_id"]
    top_k = state.get("top_k", 5)
    results = list(state.get("results", []))
    trace = [*state.get("trace", []), "[web_fallback] 本地候选不足，触发联网兜底补搜。"]
    events = [*state.get("events", []), StreamEvent(type="eval", content="本地候选不足，联网兜底补搜。")]
    outcomes = list(state.get("tool_outcomes", []))
    state_context = dict(state.get("context") or {})
    call = ToolCall(name="web_music_search", arguments=_planned_arguments("web_music_search", query, plan, top_k))
    local_results: list[dict[str, Any]] = []
    runtime_result = await _run_tool_async_safely(
        agent, call, plan, query, user_id, top_k, results,
        local_results, trace, events,
        state.get("thread_id") or f"{user_id}:default", state.get("run_id") or "",
        bool(state.get("_interrupt_enabled")),
    )
    results.extend(local_results)
    if runtime_result is not None:
        outcomes.append(_tool_outcome(call, runtime_result, state.get("_refine_count", 0)))
    return {
        **state, "results": results, "trace": trace, "events": events,
        "tool_outcomes": outcomes, "_need_web_fallback": False,
    }


def route_after_execute(state: AgentState) -> str:
    """条件路由：候选不足 → web_fallback，否则 → reflect。"""
    return "web_fallback" if state.get("_need_web_fallback") else "reflect"


def route_after_reflect(state: AgentState) -> str:
    """reflect 后条件路由：候选不足且仍可补量时回 execute_tools。"""
    return "refine" if state.get("_need_refine") else "finalize"


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

    P1 查询改写：plan 带 LLM 合成的 search_query 时，以它为基底（已融合多轮
    上下文 + 否定转正向），而非原始 query——否则"不要中文歌曲"会原样发给搜索层。
    实体已出现在基底中则不重复拼接（避免 "找 The Weeknd The Weeknd"）。
    """
    base = (getattr(plan.retrieval_plan, "search_query", "") or "").strip() or query
    entities = plan.retrieval_plan.entities
    if not entities:
        return base
    lowered = base.lower()
    extra = [e for e in entities if e and e.lower() not in lowered]
    if not extra:
        return base
    return " ".join([base, *extra]).strip()


def _record_runtime_result(
    handler: str,
    call: ToolCall,
    arguments: dict[str, Any],
    runtime_result: ToolResult,
    results: list[dict[str, Any]],
    trace: list[str],
    events: list[StreamEvent],
    *,
    checkpoint_store: Any,
    thread_id: str,
    user_id: str,
    query: str,
) -> ToolResult:
    summary = runtime_result.summary or handler
    if runtime_result.data:
        results.append(runtime_result.data)
    if runtime_result.status == ToolStatus.ERROR:
        error_message = runtime_result.error.message if runtime_result.error else f"{handler} failed"
        trace.append(f"[tool_error] {handler} 失败，已跳过：{error_message}")
        events.append(StreamEvent(
            type="error",
            content=f"{handler} 暂时不可用，已跳过该工具。",
            payload={"tool": handler, "error": error_message},
        ))
    else:
        trace.append(f"[{handler}] {summary}")
    trace.append(
        f"[tool_status] tool={handler} status={runtime_result.status.value} "
        f"candidates={len(runtime_result.cards)}"
    )
    if runtime_result.status == ToolStatus.CONFIRMATION_REQUIRED:
        checkpoint_store.put(
            call.call_id, thread_id, user_id, handler, arguments, query,
        )
        events.append(StreamEvent(
            type="confirmation_required",
            content=summary,
            payload={"action_id": call.call_id, "tool": handler, "arguments": arguments},
        ))
    if handler == "artist_albums":
        for album in runtime_result.data.get("albums", []):
            events.append(StreamEvent(type="album_card", content=album.get("name", ""), payload={"album": album}))
    if handler == "similar_artists":
        for artist in runtime_result.data.get("artists", []):
            events.append(StreamEvent(type="artist_card", content=artist.get("name", ""), payload=artist))
    if handler == "build_music_dossier" and runtime_result.data.get("dossier"):
        events.append(StreamEvent(
            type="dossier",
            content=runtime_result.data.get("answer", "已生成音乐档案。"),
            payload={"dossier": runtime_result.data.get("dossier")},
        ))
    if handler == "build_sample_dossier" and runtime_result.data.get("sample_dossier"):
        events.append(StreamEvent(
            type="sample_relations",
            content=runtime_result.data.get("answer", "已生成采样溯源结果。"),
            payload={
                "sample_dossier": runtime_result.data.get("sample_dossier"),
                "relations": runtime_result.data.get("sample_relations") or [],
                "source_cards": runtime_result.data.get("source_cards") or [],
            },
        ))
    if runtime_result.cards:
        payload: dict[str, Any] = {"count": len(runtime_result.cards), "cards": runtime_result.cards}
        if handler == "taste_experiment":
            experiment = runtime_result.data.get("experiment")
            payload["taste_experiment"] = experiment.model_dump(mode="json") if hasattr(experiment, "model_dump") else experiment
        if handler == "journey":
            payload["journey"] = runtime_result.data.get("journey")
        events.append(StreamEvent(type="candidates", content=f"{handler} {len(runtime_result.cards)} 个结果", payload=payload))
    events.append(StreamEvent(
        type="tool_result", content=summary,
        payload={"tool": handler, "status": runtime_result.status.value},
    ))
    return runtime_result


async def execute_tools_async(agent: AudioVisualAgent, state: AgentState) -> AgentState:
    """Native async stage executor used by the streaming LangGraph."""
    import asyncio

    plan = state["plan"]
    query = state["query"]
    user_id = state["user_id"]
    top_k = state.get("top_k", 5)
    results = list(state.get("results", []))
    trace = list(state.get("trace", []))
    events = list(state.get("events", []))
    outcomes = list(state.get("tool_outcomes", []))
    thread_id = state.get("thread_id") or f"{user_id}:default"
    run_id = state.get("run_id") or ""
    interrupt_enabled = bool(state.get("_interrupt_enabled"))
    refine_count = state.get("_refine_count", 0) + (1 if state.get("_need_refine") else 0)
    executed: set[str] = set()
    state_context = dict(state.get("context") or {})

    async def run_call(call: ToolCall, shared_results: list[dict[str, Any]]):
        local_results: list[dict[str, Any]] = []
        local_trace: list[str] = []
        local_events: list[StreamEvent] = []
        result = await _run_tool_async_safely(
            agent, call, plan, query, user_id, top_k, shared_results,
            local_results, local_trace, local_events,
            thread_id, run_id, interrupt_enabled, state_context,
        )
        return call, result, local_results, local_trace, local_events

    for stage in plan.stages or [ToolStage(calls=[], parallel=True)]:
        if stage.parallel and len(stage.calls) > 1:
            completed = await asyncio.gather(*(run_call(call, list(results)) for call in stage.calls))
        else:
            completed = []
            for call in stage.calls:
                completed.append(await run_call(call, results))
                _, runtime_result, local_results, local_trace, local_events = completed[-1]
                results.extend(local_results)
                trace.extend(local_trace)
                events.extend(local_events)
                if runtime_result is not None:
                    outcomes.append(_tool_outcome(call, runtime_result, refine_count))
                    executed.add(call.name)
            continue
        for call, runtime_result, local_results, local_trace, local_events in completed:
            results.extend(local_results)
            trace.extend(local_trace)
            events.extend(local_events)
            if runtime_result is not None:
                outcomes.append(_tool_outcome(call, runtime_result, refine_count))
                executed.add(call.name)
    return {
        **state,
        "results": results,
        "trace": trace,
        "events": events,
        "tool_outcomes": outcomes,
        "_need_web_fallback": _needs_web_fallback(plan, results, executed),
        "_need_refine": False,
        "_refine_count": refine_count,
        "_evaluated": False,
    }


async def _run_tool_async_safely(
    agent: AudioVisualAgent,
    call: ToolCall,
    plan: AgentPlan,
    query: str,
    user_id: str,
    top_k: int,
    prior_results: list[dict[str, Any]],
    results: list[dict[str, Any]],
    trace: list[str],
    events: list[StreamEvent],
    thread_id: str,
    run_id: str,
    interrupt_enabled: bool,
    state_context: dict[str, Any] | None = None,
) -> ToolResult | None:
    import asyncio

    try:
        return await _run_tool_async(
            agent, call, plan, query, user_id, top_k, prior_results,
            results, trace, events, thread_id, run_id, interrupt_enabled, state_context,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        if any(base.__name__ == "GraphBubbleUp" for base in type(exc).__mro__):
            raise
        handler = get_handler(call.name) or call.name
        result = ToolResult(
            tool=handler, status=ToolStatus.ERROR,
            error=ToolError(kind=type(exc).__name__, message=str(exc)),
        )
        trace.extend([
            f"[tool_error] {handler} 失败，已跳过：{exc}",
            f"[tool_status] tool={handler} status=error candidates=0",
        ])
        events.append(StreamEvent(
            type="error", content=f"{handler} 暂时不可用，已跳过该工具。",
            payload={"tool": handler, "error": str(exc)},
        ))
        return result


async def _run_tool_async(
    agent: AudioVisualAgent,
    call: ToolCall,
    plan: AgentPlan,
    query: str,
    user_id: str,
    top_k: int,
    prior_results: list[dict[str, Any]],
    results: list[dict[str, Any]],
    trace: list[str],
    events: list[StreamEvent],
    thread_id: str,
    run_id: str,
    interrupt_enabled: bool,
    state_context: dict[str, Any] | None = None,
) -> ToolResult:
    handler = get_handler(call.name)
    if handler is None:
        raise ValueError(f"Unknown tool: {call.name}")
    spec = get_tool_spec(handler)
    from app.tools.contracts import ToolContext
    from app.tools.service import checkpoint_store, tool_runtime

    arguments = call.arguments or _planned_arguments(handler, query, plan, top_k)
    call = call.model_copy(update={"name": handler, "arguments": arguments})
    plan_payload = plan.model_dump(mode="json")
    plan_payload["_excluded_tracks"] = getattr(plan, "_excluded_tracks", None) or []
    events.append(StreamEvent(type="tool_start", content=f"调用 {handler}", payload={"tool": handler}))
    confirmation: dict[str, Any] | None = None
    runtime_result: ToolResult | None = None
    if interrupt_enabled and spec is not None and spec.risk == ToolRisk.EXTERNAL_WRITE:
        try:
            arguments = spec.args_model.model_validate(arguments).model_dump(exclude_none=True)
            call = call.model_copy(update={"arguments": arguments})
        except Exception:
            pass
        else:
            from langgraph.config import get_stream_writer
            from langgraph.types import interrupt

            action_payload = {
                "action_id": call.call_id, "tool": handler,
                "arguments": arguments, "query": query[:200],
            }
            created = checkpoint_store.put(
                call.call_id, thread_id, user_id, handler, arguments, query,
            )
            if created:
                get_stream_writer()(StreamEvent(
                    type="confirmation_required",
                    content="这是外部账号写操作，需要明确确认后执行。",
                    payload=action_payload,
                ).model_dump(mode="json"))
            decision = interrupt(action_payload)
            approved = (
                isinstance(decision, dict)
                and decision.get("action_id") == call.call_id
                and decision.get("approved") is True
            )
            if not approved:
                runtime_result = ToolResult(
                    tool=handler, status=ToolStatus.CANCELLED,
                    summary="用户已拒绝该外部写操作。",
                    data={"action_id": call.call_id},
                )
            else:
                confirmation = {"action_id": call.call_id, "approved": True}
                if "confirm" in spec.args_schema:
                    arguments = {**arguments, "confirm": True}
                    call = call.model_copy(update={"arguments": arguments})
    context_kwargs = {"run_id": run_id} if run_id else {}
    state_context = state_context or {}
    if runtime_result is None:
        runtime_result = await tool_runtime.execute(call, ToolContext(
            thread_id=thread_id, user_id=user_id, query=query, plan=plan_payload,
            prior_results=prior_results, confirmation=confirmation, agent=agent,
            deadline_at=state_context.get("deadline_at"),
            latency_budget=state_context.get("latency_budget") or {},
            **context_kwargs,
        ))
    return _record_runtime_result(
        handler, call, arguments, runtime_result, results, trace, events,
        checkpoint_store=checkpoint_store, thread_id=thread_id,
        user_id=user_id, query=query,
    )


def _tool_outcome(call: ToolCall, result: ToolResult, attempt: int) -> dict[str, Any]:
    return ToolOutcome(
        call_id=call.call_id,
        tool=result.tool,
        status=result.status.value,
        arguments=call.arguments,
        summary=result.summary or "",
        error=result.error.model_dump(mode="json") if result.error else None,
        card_count=len(result.cards),
        provenance=result.provenance,
        metrics=result.metrics,
        attempt=attempt,
    ).model_dump(mode="json")


def _infer_aux_arguments(handler: str, query: str, plan: AgentPlan) -> dict[str, Any]:
    """Deterministic fallback arguments, materialized into ToolCall during planning."""
    entities = [item.strip() for item in plan.retrieval_plan.entities if item.strip()]
    lowered = query.lower()
    if handler == "feedback":
        if any(token in lowered for token in ("不喜欢", "别再", "讨厌")):
            action = "dislike"
        elif any(token in lowered for token in ("跳过", "切歌", "skip")):
            action = "skip"
        elif any(token in lowered for token in ("播放了", "听过", "played")):
            action = "played"
        else:
            action = "like"
        return {"action": action, "title": entities[0] if entities else "", "reason": query}
    if handler == "listening_history":
        window = "week" if "上周" in query or "最近一周" in query else "month" if "本月" in query or "最近一个月" in query else "recent"
        return {"window": window, "group_by": "artist" if "歌手" in query else "track", "top_k": plan.target_count or 10}
    if handler == "list_my_playlists":
        return {}
    if handler == "find_on_platform":
        platform = "youtube" if "youtube" in lowered else "bilibili" if any(token in lowered for token in ("b站", "bilibili")) else "netease"
        return {"title": entities[0] if entities else query, "artist": entities[1] if len(entities) > 1 else "", "platform": platform}
    if handler == "lyrics":
        return {"title": entities[0] if entities else query, "artist": entities[1] if len(entities) > 1 else ""}
    if handler == "audio_features":
        return {"title": entities[0] if entities else query}
    if handler == "concert_events":
        return {"artist": entities[0] if entities else query}
    return {"confirm": False}


def evaluate(agent: AudioVisualAgent, state: AgentState) -> AgentState:
    return _ensure_evaluated_state(state)


def _ensure_evaluated_state(state: AgentState) -> AgentState:
    if state.get("_evaluated"):
        return state
    try:
        plan = state["plan"]
        results = state.get("results", [])
        candidate_count = len(_collect_tracks(results))
        message = "结果可用于回答。"
        if plan.target_count and candidate_count < plan.target_count:
            message = f"候选 {candidate_count}/{plan.target_count}，不足时必须诚实说明，不编造补齐。"
    except Exception as exc:  # noqa: BLE001
        message = f"评估节点暂时不可用，继续生成保守回答：{exc}"
    return {
        **state,
        "trace": [*state.get("trace", []), f"[eval] {message}"],
        "events": [*state.get("events", []), StreamEvent(type="eval", content=message)],
        "_evaluated": True,
    }


async def reflect_async(agent: AudioVisualAgent, state: AgentState) -> AgentState:
    state = _ensure_evaluated_state(state)
    try:
        recovery = await _prepare_empty_result_recovery_async(agent, state)
        if recovery is not None:
            return recovery
        plan = state["plan"]
        if _is_knowledge_intent(plan.intent):
            return {**state, "_need_refine": False}
        if plan.intent not in {"recommend", "search", "playlist", "video"} or settings.mock_mode:
            return {**state, "_need_refine": False}
        tracks = _collect_tracks(state.get("results", []))
        constraints = _gather_constraints(agent, state)
        if not tracks or not constraints:
            return {**state, "_need_refine": False}
        reflected = await _llm_reflect_tracks_async(agent, tracks, constraints)
        return _apply_reflection_result(state, tracks, *reflected)
    except Exception as exc:  # noqa: BLE001
        return {
            **state,
            "trace": [*state.get("trace", []), f"[reflect_error] 自省节点失败，已跳过：{exc}"],
            "events": [*state.get("events", []), StreamEvent(
                type="eval", content="自省节点暂时不可用，已跳过并继续生成回答。",
                payload={"error": str(exc)},
            )],
            "_need_refine": False,
        }


def _apply_reflection_result(
    state: AgentState,
    tracks: list[Any],
    drop_indices: list[int],
    used_reflection_prompt: bool,
    llm_metrics: dict[str, float | int],
) -> AgentState:
    plan = state["plan"]
    results = state.get("results", [])
    trace = list(state.get("trace", []))
    events = list(state.get("events", []))
    context = dict(state.get("context") or {})
    context["runtime_metrics"] = merge_runtime_metrics(context.get("runtime_metrics"), llm_metrics)
    need_refine = False
    if drop_indices:
        drop_keys = {_track_key(tracks[i]) for i in drop_indices if i < len(tracks)}
        results = _drop_tracks_from_results(results, drop_keys)
        excluded = _merge_excluded_tracks(
            plan._excluded_tracks,
            [_track_to_excluded_item(tracks[i]) for i in drop_indices if i < len(tracks)],
        )
        plan._excluded_tracks = excluded
        remaining = len(_collect_tracks(results))
        # 候选补量回环默认关闭（ENABLE_REFLECT_REFINE）：它会重新跑 execute_tools + reflect，
        # 引入第 4/5 次串行往返（含联网搜索）。不足时由 _compose_intro 如实说明 shortfall。
        if (
            settings.enable_reflect_refine
            and plan.target_count
            and remaining < plan.target_count
            and state.get("_refine_count", 0) < 1
        ):
            need_refine = True
        labels = "、".join(_track_label(tracks[i]) for i in drop_indices if i < len(tracks))[:120]
        note = f"[reflect] LLM 核对剔除 {len(drop_indices)} 首违反约束候选：{labels}"
        if need_refine:
            note += "；候选不足，回环补量一次。"
    else:
        note = "[reflect] LLM 核对通过，无候选违反用户约束。"
    if used_reflection_prompt:
        context["prompt_versions"] = _merge_prompt_versions(
            context.get("prompt_versions"),
            {"candidate_reflection": CANDIDATE_REFLECTION_VERSION},
        )
        note += f" [prompt] candidate_reflection={CANDIDATE_REFLECTION_VERSION}"
    trace.append(note)
    events.append(StreamEvent(type="eval", content=note))
    return {**state, "results": results, "trace": trace, "events": events, "context": context, "_need_refine": need_refine}


_RECOVERY_INTENTS = {"recommend", "search", "playlist"}
_NO_AUTO_RECOVERY = {
    ToolStatus.AUTH_REQUIRED.value,
    ToolStatus.CONFIRMATION_REQUIRED.value,
    ToolStatus.UNSUPPORTED.value,
    ToolStatus.CANCELLED.value,
}
_NETWORK_ERROR_KINDS = {
    "timeout", "timeouterror", "connectionerror", "connecterror", "readerror",
    "remoteprotocolerror", "oserror",
}


async def _prepare_empty_result_recovery_async(
    agent: AudioVisualAgent,
    state: AgentState,
) -> AgentState | None:
    if not settings.enable_empty_result_recovery:
        return None
    attempt = int(state.get("_refine_count", 0))
    if _is_knowledge_intent(state["plan"].intent):
        return None
    if attempt >= settings.empty_result_recovery_max_attempts or state["plan"].intent not in _RECOVERY_INTENTS:
        return None
    current = [item for item in state.get("tool_outcomes", []) if int(item.get("attempt", 0)) == attempt]
    if not current:
        return None
    statuses = {str(item.get("status") or "") for item in current}
    if statuses and statuses.issubset(_NO_AUTO_RECOVERY):
        return None
    decision = _deterministic_recovery_decision(state, current)
    context = dict(state.get("context") or {})
    if decision is None and not context.get("recovery_llm_used"):
        decision = await _llm_recovery_decision_async(agent, state, current)
        context["recovery_llm_used"] = True
    if decision is None or decision.action != "retry" or not decision.calls:
        return None
    return _apply_recovery_decision(state, current, decision, context)


def _apply_recovery_decision(
    state: AgentState,
    current: list[dict[str, Any]],
    decision: RecoveryDecision,
    context: dict[str, Any],
) -> AgentState | None:
    plan = state["plan"]
    attempt = int(state.get("_refine_count", 0))

    blocked_tools = {
        str(item.get("tool") or "")
        for item in current
        if item.get("status") in {*_NO_AUTO_RECOVERY, ToolStatus.ERROR.value}
    }
    calls = [name for name in _safe_recovery_calls(decision.calls) if name not in blocked_tools]
    if not calls:
        return None
    old_query = _query_with_entities(state["query"], plan)
    search_query = decision.search_query.strip() or old_query
    retrieval = plan.retrieval_plan.model_copy(update={"search_query": search_query})
    network_calls = {"web_music_search", "web_info_search", "video_search"}
    online = any((get_handler(name) or name) in network_calls for name in calls)
    revised = plan.model_copy(update={
        "tools_needed": calls,
        "stages": [],
        "strategy": "online_first" if online else "library_first",
        "online_required": online,
        "retrieval_plan": retrieval.model_copy(update={"use_web": online}),
        "reasoning_summary": f"{plan.reasoning_summary}；恢复策略：{decision.reason}",
    })
    revised = _materialize_tool_stages(revised, state["query"], state.get("top_k", 5))
    event_payload = {
        "reason": decision.reason,
        "old_query": old_query,
        "search_query": search_query,
        "tools": calls,
        "attempt": attempt + 1,
    }
    note = f"[refine] {decision.reason}；query={search_query}；tools={'/'.join(calls)}"
    return {
        **state,
        "plan": revised,
        "context": context,
        "trace": [*state.get("trace", []), note],
        "events": [*state.get("events", []), StreamEvent(
            type="refine", content=decision.reason, payload=event_payload,
        )],
        "_need_refine": True,
    }


def _deterministic_recovery_decision(
    state: AgentState,
    outcomes: list[dict[str, Any]],
) -> RecoveryDecision | None:
    plan = state["plan"]
    tracks = _collect_tracks(state.get("results", []))
    by_tool = {str(item.get("tool")): item for item in outcomes}
    recommend_outcome = by_tool.get("recommend")
    if tracks and recommend_outcome and recommend_outcome.get("status") == ToolStatus.EMPTY.value:
        return RecoveryDecision(
            action="retry", reason="搜索候选已存在，仅重新执行推荐排序。",
            search_query=_query_with_entities(state["query"], plan), calls=["recommend"],
        )

    failures = [item for item in outcomes if item.get("status") == ToolStatus.ERROR.value]
    network_failed = any(
        item.get("tool") == "web_music_search"
        and str((item.get("error") or {}).get("kind") or "").lower() in _NETWORK_ERROR_KINDS
        for item in failures
    )
    if network_failed:
        return RecoveryDecision(
            action="retry",
            reason="在线搜索连接失败，切换到可追溯的本地检索。",
            search_query=_positive_recovery_query(state),
            calls=_local_recovery_calls(plan.intent),
        )

    web_empty = any(
        item.get("tool") == "web_music_search" and item.get("status") == ToolStatus.EMPTY.value
        for item in outcomes
    )
    all_empty = not tracks and any(item.get("status") == ToolStatus.EMPTY.value for item in outcomes)
    if web_empty or all_empty:
        attempted = {
            str((item.get("arguments") or {}).get("query") or "").strip().lower()
            for item in state.get("tool_outcomes", []) if item.get("tool") == "web_music_search"
        }
        candidates = [_positive_recovery_query(state), *plan.retrieval_plan.search_variants]
        revised_query = next(
            (value.strip() for value in candidates if value.strip() and value.strip().lower() not in attempted),
            "",
        )
        if revised_query:
            calls = ["web_music_search", *_downstream_recovery_calls(plan.intent)]
            return RecoveryDecision(
                action="retry", reason="首轮没有候选，使用正向检索词和未尝试变体重新搜索。",
                search_query=revised_query, calls=calls,
            )
        return RecoveryDecision(
            action="retry", reason="在线检索无候选，切换到可追溯的本地检索。",
            search_query=_positive_recovery_query(state), calls=_local_recovery_calls(plan.intent),
        )
    return None


def _positive_recovery_query(state: AgentState) -> str:
    plan = state["plan"]
    constraints = extract_content_negations(state["query"])
    retrieval = plan.retrieval_plan
    base = _strip_negative_query(retrieval.search_query or state["query"], constraints)
    dialogue = (state.get("context") or {}).get("dialogue_state") or {}
    parts = [
        base,
        *retrieval.entities,
        *retrieval.mood_filter,
        *retrieval.genre_filter,
        *retrieval.scenario_filter,
        *(dialogue.get("entities") or []),
        *(dialogue.get("mood_tags") or []),
        *(dialogue.get("genre_tags") or []),
        *(dialogue.get("scenario_tags") or []),
    ]
    values: list[str] = []
    seen: set[str] = set()
    blocked = {
        _constraint_key(alias)
        for constraint in constraints
        for alias in expand_content_negation(constraint)
        if _constraint_key(alias)
    }
    for part in parts:
        value = str(part or "").strip()
        key = _constraint_key(value)
        is_blocked = any(item in key or key in item for item in blocked) if key else False
        if value and key not in seen and not is_blocked:
            values.append(value)
            seen.add(key)
    return " ".join(values).strip()


def _downstream_recovery_calls(intent: str) -> list[str]:
    return {"recommend": ["recommend"], "search": ["search"], "playlist": ["playlist"]}.get(intent, [])


def _local_recovery_calls(intent: str) -> list[str]:
    return {"recommend": ["recommend"], "search": ["search"], "playlist": ["search", "playlist"]}.get(intent, [])


def _safe_recovery_calls(calls: list[str]) -> list[str]:
    safe: list[str] = []
    for name in calls[:3]:
        spec = get_tool_spec(name)
        if spec is None or spec.risk != ToolRisk.READ:
            continue
        if spec.name not in safe:
            safe.append(spec.name)
    return safe


async def _llm_recovery_decision_async(
    agent: AudioVisualAgent,
    state: AgentState,
    outcomes: list[dict[str, Any]],
) -> RecoveryDecision | None:
    if settings.mock_mode:
        return None
    allowed = [
        name for name in ("web_music_search", "search", "recommend", "playlist")
        if (spec := get_tool_spec(name)) is not None and spec.risk == ToolRisk.READ
    ]
    prompt = (
        f"用户请求：{state['query']}\n当前正向查询：{_positive_recovery_query(state)}\n"
        f"工具观察：{str(outcomes)[:2400]}\n可选只读工具：{', '.join(allowed)}\n"
        "最多重试一次。若值得改变查询或工具则 retry，否则 finalize。"
    )
    system = (
        "你是音乐 Agent 的失败恢复规划器。只输出 JSON："
        '{"action":"retry|finalize","reason":"...","search_query":"...","calls":["..."]}。'
        "禁止选择给定列表外的工具，禁止写操作。"
    )
    try:
        llm = select_llm(agent, "fast")
        payload = extract_json_dict(await llm.agenerate(prompt, system=system, temperature=0.0))
        decision = RecoveryDecision.model_validate(payload)
    except Exception:
        return None
    decision.calls = _safe_recovery_calls(decision.calls)
    return decision if decision.action == "finalize" or decision.calls else None


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


def _track_to_excluded_item(track: Any) -> dict[str, str]:
    return {
        "title": getattr(track, "title", "") or "",
        "artist": getattr(track, "artist", "") or "",
        "source": getattr(track, "source", "") or "",
        "source_id": getattr(track, "external_id", "") or getattr(track, "asset_id", "") or "",
    }


def _taste_experiment_card(item: Any) -> dict[str, Any]:
    track = item.track
    return {
        "title": track.title,
        "artist": track.artist,
        "source": track.source,
        "source_id": track.source_id,
        "genre": track.genre,
        "mood": track.mood,
        "score": track.score,
        "components": item.components or track.components,
        "reason": item.reason,
        "bucket": item.bucket,
        "expected_signal": item.expected_signal,
    }


def _merge_excluded_tracks(existing: list[dict[str, str]], new_items: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in [*existing, *new_items]:
        title = item.get("title", "").lower().strip()
        source_id = item.get("source_id", "").strip()
        if not title:
            continue
        key = (title, source_id)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


async def _llm_reflect_tracks_async(
    agent: AudioVisualAgent,
    tracks: list[Any],
    constraints: list[str],
) -> tuple[list[int], bool, dict[str, float | int]]:
    catalog = "\n".join(
        f"[{i}] {_track_label(t)} | 曲风:{'/'.join(getattr(t, 'genre', []) or [])} "
        f"情绪:{'/'.join(getattr(t, 'mood', []) or [])}"
        for i, t in enumerate(tracks[:15])
    )
    cons = "；".join(f"({i + 1}) {c}" for i, c in enumerate(constraints[:8]))
    prompt = CANDIDATE_REFLECTION_USER.format(catalog=catalog, constraints=cons)
    llm = select_llm(agent, "strong")
    try:
        raw = await llm.agenerate(prompt, system=CANDIDATE_REFLECTION_SYSTEM, temperature=0.0)
        data = extract_json_dict(raw)
        drop = data.get("drop") if isinstance(data, dict) else None
        if isinstance(drop, list):
            indices = [int(x) for x in drop if isinstance(x, (int, float)) and 0 <= int(x) < len(tracks)]
            return indices, True, capture_llm_stats(llm)
    except Exception:
        logger.debug("reflect async LLM 核对失败，跳过", exc_info=True)
    return [], True, capture_llm_stats(llm)


def _drop_tracks_from_results(results: list[dict[str, Any]], drop_keys: set[str]) -> list[dict[str, Any]]:
    """从 results 的各类型结果里移除命中 drop_keys 的曲目（reflect 拒绝的）。

    覆盖主要结果类型；未知类型不处理（graceful，finalize 仍能正常工作）。
    """
    for r in results:
        t = r.get("type")
        if t in {"recommend", "recommend_music", "daily_recommend"}:
            rec = r.get("recommendation")
            if rec is not None and hasattr(rec, "tracks"):
                rec.tracks = [tr for tr in rec.tracks if _track_key(getattr(tr, "asset", tr)) not in drop_keys]
        elif t == "playlist":
            pl = r.get("playlist")
            if pl is not None and hasattr(pl, "tracks"):
                pl.tracks = [tr for tr in pl.tracks if _track_key(tr) not in drop_keys]
        elif t == "web_music_search":
            r["tracks"] = [tr for tr in r.get("tracks", []) if _track_key(tr) not in drop_keys]
        elif t == "search":
            response = r.get("response")
            if response is not None:
                response.external = [
                    tr for tr in response.external if _track_key(tr) not in drop_keys
                ]
                response.local = [
                    tr for tr in response.local if _track_key(tr) not in drop_keys
                ]
        elif t == "journey":
            journey = r.get("journey") or {}
            for phase in journey.get("phases", []):
                phase["tracks"] = [
                    tr for tr in phase.get("tracks", [])
                    if _track_key(ExternalTrack.model_validate(tr)) not in drop_keys
                ]
    return results


async def _finalize_tail_async(
    agent: AudioVisualAgent, state: AgentState, answer_text: str,
) -> tuple[AgentAnswer, dict[str, Any], list[str]]:
    """answer_text 已知后的收尾：guard / 记忆自学习 / 目标推进 / 持久化 / 构建 AgentAnswer + final_payload。

    流式与非流式共用，副作用只跑一次。
    """
    known = collect_known_titles(state.get("results", []))
    answer_text, removed = guard_answer(answer_text, known)
    memory_updated = await agent.memory.auto_learn_from_turn_async(
        state["user_id"], state["query"], state.get("results", []),
    )
    goal = None
    if state["plan"].intent != "chat":
        goal = agent.memory.ensure_goal(state["user_id"], state["query"])
        goal = agent.memory.update_goal_progress(state["user_id"], goal, _completed_actions(state.get("results", [])))
    trace = list(state.get("trace", []))
    if removed:
        trace.append(f"[guard] 移除 {len(removed)} 个未核实歌名。")
    trace.append("[final] 输出 grounded answer。")
    runtime_metrics = dict((state.get("context") or {}).get("runtime_metrics") or {})
    if runtime_metrics:
        trace.append(f"[meta] {format_runtime_metrics(runtime_metrics)}")
    _persist_dialogue_state(agent, state)
    listed = _select_listed_tracks(state.get("results", []), state["plan"])
    aligned_cards = _aligned_cards(listed, state.get("events", [])) if listed else []
    answer = AgentAnswer(
        answer=answer_text,
        evidences=[],
        recommended_tracks=[_track_ref_from_card(card) for card in aligned_cards],
        prompt_versions=dict((state.get("context") or {}).get("prompt_versions") or {}),
        runtime_metrics=runtime_metrics,
        memory_updated=memory_updated,
        agent_trace=trace,
        pending_goal=goal.goal if goal and goal.status == "active" else None,
        goal_progress=_goal_progress(goal),
    )
    # 权威卡片：与答案文本实际列出的曲目严格一一对应，挂到 final 事件。
    # 前端收到后替换流式预览卡片，保证「文本列几首 = 底部几张卡」。
    final_payload = answer.model_dump(mode="json")
    if listed:
        final_payload["cards"] = aligned_cards
    experiment_payload = _taste_experiment_payload(state.get("results", []))
    if experiment_payload:
        final_payload["taste_experiment"] = experiment_payload
    artists_payload = _similar_artists_payload(state.get("results", []))
    if artists_payload:
        final_payload["artists"] = artists_payload
    dossier_payload = _music_dossier_payload(state.get("results", []))
    if dossier_payload:
        final_payload["dossier"] = dossier_payload
    sample_payload = _sample_dossier_payload(state.get("results", []))
    if sample_payload:
        final_payload["sample_dossier"] = sample_payload
        final_payload["sample_relations"] = sample_payload.get("relations") or []
        source_cards = sample_payload.get("source_track_cards") or []
        if source_cards:
            final_payload["cards"] = source_cards
    final_payload["trace_summary"] = _trace_summary(
        state["plan"], state.get("results", []), trace, aligned_cards,
        state.get("tool_outcomes", []), state.get("context") or {},
    )
    return answer, final_payload, trace


async def finalize_stream_async(agent: AudioVisualAgent, state: AgentState):
    """Native async finalize used by the production SSE graph."""
    try:
        context = state.get("context") or {}
        parts: list[str] = []
        async for delta in compose_answer_stream_async(
            state["query"], state.get("results", []), state["plan"],
            agent=agent,
            memory_query=context.get("memory_query", ""),
            history_text=context.get("history_text", ""),
            user_id=state["user_id"],
        ):
            if delta:
                parts.append(delta)
                yield StreamEvent(type="token", content=delta)
        answer, final_payload, _trace = await _finalize_tail_async(agent, state, "".join(parts))
        yield StreamEvent(type="final", content=answer.answer, payload=final_payload)
    except Exception as exc:  # noqa: BLE001
        fallback = _finalize_fallback(state, exc)
        final_event = next((ev for ev in fallback.get("events", []) if ev.type == "final"), None)
        yield final_event or StreamEvent(type="final", content="这轮处理出错了，请重试。", payload={})


def _finalize_fallback(state: AgentState, exc: Exception) -> AgentState:
    query = str(state.get("query") or "这次请求")
    trace = [
        *state.get("trace", []),
        f"[final_error] finalize 失败，输出保守兜底回答：{exc}",
    ]
    answer_text = f"这轮我已经尽量处理了“{query}”，但最后整理答案时遇到错误。你可以先查看上方候选结果，我没有编造额外歌曲。"
    answer = AgentAnswer(
        answer=answer_text,
        evidences=[],
        recommended_tracks=[],
        prompt_versions=dict((state.get("context") or {}).get("prompt_versions") or {}),
        runtime_metrics=dict((state.get("context") or {}).get("runtime_metrics") or {}),
        memory_updated=False,
        agent_trace=trace,
        fallback_reason=f"finalize_error: {exc}",
    )
    final_payload = answer.model_dump(mode="json")
    fallback_cards: list[dict[str, Any]] = []
    seen_cards: set[tuple[str, str, str]] = set()
    for event in state.get("events", []):
        if event.type != "candidates":
            continue
        for card in (event.payload or {}).get("cards", []):
            key = (
                str(card.get("title", "")).lower(),
                str(card.get("source", "")),
                str(card.get("source_id", "")),
            )
            if key in seen_cards:
                continue
            seen_cards.add(key)
            fallback_cards.append(card)
    if fallback_cards:
        final_payload["cards"] = fallback_cards
    final_payload["trace_summary"] = {
        "intent": getattr(state.get("plan"), "intent", "unknown"),
        "tools": [],
        "sources": [],
        "fallback": f"finalize_error: {exc}",
        "guard_removed": 0,
        "final_cards": len(fallback_cards),
    }
    return {
        **state,
        "answer": answer,
        "trace": trace,
        "events": [*state.get("events", []), StreamEvent(type="final", content=answer_text, payload=final_payload)],
    }


def _select_listed_tracks(results: list[dict[str, Any]], plan: AgentPlan) -> list[Any]:
    """返回答案文本实际会列出的那批曲目（与 compose_answer 的截断逻辑严格一致）。

    仅对会渲染确定性曲目清单的意图返回非空（recommend/search/playlist/journey）；
    chat/discuss/taste 的文本不是一行行歌名，返回 [] 表示不接管卡片，
    让前端保留流式预览卡片。
    """
    if plan.intent == "playlist":
        pl = next((r["playlist"] for r in results if r.get("type") == "playlist"), None)
        tracks = list(pl.tracks) if pl and pl.tracks else _collect_tracks(results)
        return tracks[: plan.target_count or 30]
    if plan.intent == "recommend":
        recommendation = next(
            (result.get("recommendation") for result in results if result.get("type") == "daily_recommend"),
            None,
        )
        tracks = [item.asset for item in recommendation.tracks] if recommendation else []
        return (tracks or _collect_tracks(results))[: plan.target_count or 12]
    if plan.intent == "search":
        response = next(
            (result.get("response") for result in results if result.get("type") == "search"),
            None,
        )
        tracks = [*(response.external if response else []), *(response.local if response else [])]
        return (tracks or _collect_tracks(results))[: plan.target_count or 12]
    if plan.intent == "journey":
        tracks = _collect_tracks(results)
        # 旅程的阶段数决定自然长度。未显式指定首数时必须保留全部阶段，不能套用
        # 普通推荐的 12 张默认上限，否则 final 事件会把流式阶段的完整卡片截掉。
        return tracks[:plan.target_count] if plan.target_count else tracks
    if plan.intent == "import":
        return _collect_tracks(results)[: plan.target_count or 12]
    if plan.intent == "video":
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

    shown_tracks 的累积规则：本轮是延续指令时，把本轮 shown 追加到前轮累积记录上，
    去重后封顶保存——这样第 N 轮"不要重复"能排除整个会话里已展示过的曲目，而不只是
    上一轮的。非延续（全新话题）则重置为本轮 shown。

    关键：累积只看 is_continuation，不看 rp.entities。因为延续指令会从上一轮"继承"
    实体（"多来几首"继承 The Weeknd），继承后 rp.entities 非空——若用 `not rp.entities`
    做条件，会把"继承实体"误判成"话题切换"而重置，导致前几轮的展示记录被丢、
    第三轮去重又把第一轮的歌捞回来。
    """
    plan = state["plan"]
    user_id = state["user_id"]
    if plan.intent == "chat":
        agent.memory.clear_dialogue_state(user_id)
        return
    rp = plan.retrieval_plan
    # DialogueState must not depend solely on LLM-populated plan fields.  Recover
    # deterministic genre/mood/scenario seeds from the raw turn and preserve prior
    # positive tags on a continuation when one side is incomplete.
    turn_tags = extract_tags(state["query"])
    prior_dialogue = (state.get("context") or {}).get("dialogue_state") or {}

    def merged_tags(current: list[str], derived: list[str], prior: list[str]) -> list[str]:
        values = [*current, *derived]
        if is_continuation(state["query"]):
            values.extend(prior)
        return list(dict.fromkeys(item for item in values if item))

    genre_tags = merged_tags(rp.genre_filter, turn_tags["genre"], prior_dialogue.get("genre_tags") or [])
    mood_tags = merged_tags(rp.mood_filter, turn_tags["mood"], prior_dialogue.get("mood_tags") or [])
    scenario_tags = merged_tags(
        rp.scenario_filter, turn_tags["scenario"], prior_dialogue.get("scenario_tags") or [],
    )
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
    # 跨轮累积：延续指令时并入前轮记录（继承实体也算同一话题），否则视为新话题重置。
    prev_shown = prior_dialogue.get("shown_tracks") or []
    if is_continuation(state["query"]):
        merged_shown = _merge_excluded_tracks(prev_shown, shown)
    else:
        merged_shown = shown
    merged_shown = merged_shown[:80]  # 封顶，避免长期会话排除集无限增长
    agent.memory.save_dialogue_state(
        user_id,
        intent=plan.intent,
        query=state["query"],
        entities=rp.entities,
        genre_tags=genre_tags,
        mood_tags=mood_tags,
        scenario_tags=scenario_tags,
        shown_tracks=merged_shown,
    )


async def plan_with_llm_async(
    agent: AudioVisualAgent,
    query: str,
    history_text: str = "",
    memory_text: str = "",
) -> AgentPlan | None:
    plan, _, _ = await plan_with_llm_with_meta_async(agent, query, history_text, memory_text)
    return plan


async def plan_with_llm_with_meta_async(
    agent: AudioVisualAgent,
    query: str,
    history_text: str = "",
    memory_text: str = "",
) -> tuple[AgentPlan | None, dict[str, str], dict[str, float | int]]:
    try:
        sections: list[str] = []
        if history_text.strip():
            sections.append(f"【最近对话】\n{history_text}")
        if memory_text.strip() and not settings.mock_mode:
            sections.append(
                "【长期音乐偏好（仅作软参考，不得覆盖本轮明确要求）】\n"
                f"{memory_text[:500]}"
            )
        sections.append(f"【本轮输入】\n{query}" if sections else query)
        llm = select_llm(agent, "fast")
        raw = await llm.agenerate(
            "\n\n".join(sections), system=QUERY_PLAN_SYSTEM, temperature=0.1,
        )
    except Exception:
        return None, {}, empty_runtime_metrics()
    payload = _parse_query_plan_payload(raw)
    metrics = capture_llm_stats(llm)
    if payload is None:
        return None, {}, metrics
    plan = _plan_from_query_payload(payload, query)
    return plan, {"query_plan": QUERY_PLAN_VERSION}, metrics


def _plan_from_query_payload(payload: QueryPlanPayload, query: str) -> AgentPlan:
    intent = payload.intent
    spec = get_intent(intent)
    tags = extract_tags(query)
    search_query = payload.search_query.strip()
    retrieval = RetrievalPlan(
        use_local=payload.use_local,
        use_vector=payload.use_vector,
        use_web=payload.use_web if payload.use_web is not None else spec.online_default,
        entities=payload.entities,
        genre_filter=tags["genre"],
        mood_filter=tags["mood"],
        scenario_filter=tags["scenario"],
        search_query=search_query,
        search_variants=_cap_search_variants(payload.search_variants, search_query),
        language_filter=payload.language.strip().lower(),
    )
    return AgentPlan(
        intent=intent,
        strategy=spec.strategy_for(retrieval.use_web),
        tools_needed=spec.tools_for(retrieval.use_web),
        target_count=payload.target_count or _infer_count(query),
        online_required=retrieval.use_web,
        reasoning_summary=payload.reasoning.strip() or spec.summary,
        retrieval_plan=retrieval,
    )


def _merge_prompt_versions(existing: Any, incoming: dict[str, str] | None) -> dict[str, str]:
    merged = dict(existing or {})
    for key, value in (incoming or {}).items():
        if value:
            merged[key] = value
    return merged


def _format_prompt_versions(versions: dict[str, str]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(versions.items()))


def _parse_query_plan_payload(raw: str) -> QueryPlanPayload | None:
    data = extract_json_dict(raw)
    if not isinstance(data, dict):
        return None
    try:
        return QueryPlanPayload.model_validate(data)
    except Exception:
        return None


def _keyword_retrieval_plan(query: str, use_web: bool) -> RetrievalPlan:
    tags = extract_tags(query)
    variants = _cap_search_variants(_rule_search_variants(query, tags), query)
    return RetrievalPlan(
        use_local=True,
        use_vector=bool(tags["mood"] or tags["scenario"]),
        use_web=use_web,
        genre_filter=tags["genre"],
        mood_filter=tags["mood"],
        scenario_filter=tags["scenario"],
        search_variants=variants,
    )


def _cap_search_variants(variants: list[str], primary: str = "") -> list[str]:
    seen = {primary.strip().lower()} if primary.strip() else set()
    capped: list[str] = []
    for item in variants or []:
        value = str(item or "").strip()
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        capped.append(value)
        if len(capped) >= settings.max_search_variants:
            break
    return capped


def _rule_search_variants(query: str, tags: dict[str, list[str]]) -> list[str]:
    variants: list[str] = []
    genres = tags.get("genre") or []
    moods = tags.get("mood") or []
    scenarios = tags.get("scenario") or []
    if genres or moods or scenarios:
        variants.append(" ".join([*moods[:1], *genres[:1], *scenarios[:1]]).strip())
    synonym_pairs = {
        "说唱": "rap hip hop",
        "爵士": "jazz",
        "放松": "chill relaxing",
        "深夜": "late night",
        "跑步": "running workout",
        "学习": "focus study",
        "慵懒": "lofi chill",
        "R&B": "rnb soul",
    }
    for key, variant in synonym_pairs.items():
        if key in query:
            variants.append(variant)
    return variants


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
        "taste_experiment": "taste_experiment",
        "journey": "journey",
        "import_netease_playlist": "import_netease_playlist",
        "video_search": "video_search",
        "web_info_search": "web_info_search",
        "similar_artists": "similar_artists",
    }
    return [mapping.get(result.get("type", ""), result.get("type", "")) for result in results]


def _trace_summary(
    plan: AgentPlan,
    results: list[dict[str, Any]],
    trace: list[str],
    cards: list[dict[str, Any]],
    outcomes: list[dict[str, Any]] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """稳定的 Agent 摘要，供 UI 透明度面板和 smoke 报告断言使用。"""
    tracks = _select_listed_tracks(results, plan) or _collect_tracks(results)
    sources = {getattr(track, "source", "") for track in tracks if getattr(track, "source", "")}
    experiment_cards = 0
    album_cards = 0
    artist_cards = 0
    sample_cards = 0
    for result in results:
        if result.get("type") == "taste_experiment":
            exp = result.get("experiment")
            experiment_cards = sum(len(segment.tracks) for segment in getattr(exp, "segments", []) or [])
        elif result.get("type") == "artist_albums":
            album_cards += len(result.get("albums") or [])
            if result.get("albums"):
                sources.add("netease")
        elif result.get("type") == "import_netease_playlist":
            sources.add("netease")
        elif result.get("type") == "similar_artists":
            artist_cards += len(result.get("artists") or [])
            if result.get("artists"):
                sources.add("local_library")
        elif result.get("type") == "music_dossier":
            dossier = result.get("dossier") or {}
            for citation in dossier.get("citations") or []:
                if citation.get("source"):
                    sources.add(citation.get("source"))
        elif result.get("type") == "sample_dossier":
            dossier = result.get("sample_dossier") or {}
            sample_cards += len(dossier.get("source_track_cards") or [])
            for citation in dossier.get("citations") or []:
                if citation.get("source"):
                    sources.add(citation.get("source"))
    observed_tools = [str(item.get("tool") or "") for item in outcomes or [] if item.get("tool")]
    planned = list(dict.fromkeys([
        *observed_tools,
        *(get_handler(name) or name for name in plan.tools_needed),
    ]))
    tool_statuses: dict[str, str] = {}
    for item in outcomes or []:
        tool = str(item.get("tool") or "")
        status = str(item.get("status") or "")
        if tool and status:
            tool_statuses[tool] = status
    if not tool_statuses:
        for line in trace:
            match = re.match(r"\[tool_status\]\s+tool=(\S+)\s+status=(\S+)", line)
            if match:
                tool_statuses[match.group(1)] = match.group(2)
    completed = list(dict.fromkeys([*tool_statuses, *_completed_actions(results)]))
    error_details = [
        {
            "tool": str(item.get("tool") or ""),
            "message": str((item.get("error") or {}).get("message") or "unknown error"),
        }
        for item in outcomes or [] if item.get("status") == ToolStatus.ERROR.value
    ]
    if not error_details:
        for line in trace:
            match = re.match(r"\[tool_error\]\s+(\S+)\s+失败，已跳过：(.*)", line)
            if match:
                error_details.append({"tool": match.group(1), "message": match.group(2) or "unknown error"})
    errors = list(dict.fromkeys(item["tool"] for item in error_details))
    empty = [tool for tool, status in tool_statuses.items() if status == "empty"]
    if not planned:
        execution_state = "not_planned"
    elif errors:
        execution_state = "error"
    elif empty:
        execution_state = "empty"
    elif completed:
        execution_state = "ok"
    else:
        execution_state = "planned_not_executed"
    latency_budget = _latency_budget_summary(context or {}, outcomes or [], results)
    return {
        "intent": plan.intent,
        "strategy": plan.strategy,
        "tools": completed,
        "tools_planned": planned,
        "tools_executed": completed,
        "tool_statuses": tool_statuses,
        "tool_errors": errors,
        "tool_error_details": error_details,
        "empty_results": empty,
        "tool_execution_state": execution_state,
        "sources": sorted(sources),
        "used_fallback": any("fallback" in line.lower() for line in trace),
        "guard_removed": sum(1 for line in trace if line.startswith("[guard]")),
        "reflection": any("[reflect]" in line for line in trace),
        "recovery": any("[refine]" in line for line in trace),
        "final_cards": len(cards) or experiment_cards or album_cards or artist_cards or sample_cards,
        "latency_budget": latency_budget,
    }


def _latency_budget_summary(
    context: dict[str, Any],
    outcomes: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not _is_knowledge_intent(str(context.get("intent") or "")) and not context.get("latency_budget"):
        # context may not carry intent; infer from knowledge result payloads.
        if not any(r.get("type") in {"music_dossier", "sample_dossier"} for r in results):
            return None
    started = context.get("started_at_monotonic")
    elapsed = round(max(0.0, time.monotonic() - float(started)), 3) if started else 0
    timed_out = [
        str(item.get("tool") or "")
        for item in outcomes
        if (item.get("error") or {}).get("kind") == "timeout"
    ]
    skipped = []
    for item in outcomes:
        metrics = item.get("metrics") or {}
        if metrics.get("deadline_skipped"):
            skipped.append(str(item.get("tool") or ""))
    partial = any((r.get("dossier") or {}).get("partial") for r in results if r.get("type") == "music_dossier")
    partial = partial or any((r.get("sample_dossier") or {}).get("partial") for r in results if r.get("type") == "sample_dossier")
    budget = context.get("latency_budget") or {}
    return {
        "budget_seconds": budget.get("budget_seconds", settings.knowledge_turn_budget_seconds),
        "elapsed_seconds": elapsed,
        "timed_out_tools": [t for t in timed_out if t],
        "skipped_due_to_deadline": [s for s in skipped if s],
        "partial": partial,
    }


def _music_dossier_payload(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    for result in reversed(results):
        if result.get("type") == "music_dossier" and result.get("dossier"):
            return result.get("dossier")
    return None


def _sample_dossier_payload(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    for result in reversed(results):
        if result.get("type") == "sample_dossier" and result.get("sample_dossier"):
            return result.get("sample_dossier")
    return None


def _taste_experiment_payload(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    for result in results:
        if result.get("type") == "taste_experiment":
            exp = result.get("experiment")
            if hasattr(exp, "model_dump"):
                return exp.model_dump(mode="json")
            if isinstance(exp, dict):
                return exp
    return None


def _similar_artists_payload(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for result in results:
        if result.get("type") == "similar_artists":
            return list(result.get("artists") or [])
    return []


def _compose_deterministic_answer(results: list[dict[str, Any]], plan: AgentPlan) -> str:
    if plan.intent == "sample_lookup":
        dossier = _sample_dossier_payload(results)
        if dossier:
            try:
                from app.knowledge import sample_dossier_answer
                from app.models import SampleDossier

                return sample_dossier_answer(SampleDossier.model_validate(dossier))
            except Exception:
                return str((dossier or {}).get("degraded_reason") or "已生成采样溯源结果，但整理文本时降级。")
        return "这轮没有找到可核实采样关系；我不会硬编源曲。"
    if _is_knowledge_intent(plan.intent):
        dossier = _music_dossier_payload(results)
        if dossier:
            try:
                from app.knowledge import dossier_answer
                from app.models import MusicDossier

                return dossier_answer(MusicDossier.model_validate(dossier))
            except Exception:
                return str((dossier or {}).get("summary") or "已生成音乐档案，但整理文本时降级。")
        return "这轮没有在时间预算内拿到足够音乐资料；我不会编造乐评。"
    if plan.intent == "artist_albums":
        return _compose_artist_albums_answer(results)
    if plan.intent == "similar_artists":
        return _compose_similar_artists_answer(results)
    if plan.intent == "taste_experiment":
        return _compose_taste_experiment_answer(results)
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
    return "这轮没有拿到可交付的结构化结果。"


async def compose_answer_stream_async(
    query: str,
    results: list[dict[str, Any]],
    plan: AgentPlan,
    agent: AudioVisualAgent | None = None,
    memory_query: str = "",
    history_text: str = "",
    user_id: str = "",
):
    llm = select_llm(agent, "default") if agent is not None else None

    async def stream_llm(prompt: str, fallback: str, temp: float = settings.dialog_temperature):
        got = False
        if llm is not None:
            try:
                async for piece in llm.agenerate_stream(prompt, temperature=temp):
                    if piece:
                        got = True
                        yield piece
            except Exception:
                logger.debug("compose_answer_stream_async: LLM 流式失败，回退兜底", exc_info=True)
        if not got and fallback:
            yield fallback

    intent = plan.intent
    if intent == "chat":
        async for piece in stream_llm(
            _chat_prompt(query, agent, history_text, user_id),
            "你好，我在。有什么音乐上的事可以帮你?",
        ):
            yield piece
        return
    if intent == "discuss":
        prompt, refuse = _discussion_prompt(query, _collect_tracks(results), history_text)
        if prompt is None:
            yield refuse
            return
        async for piece in stream_llm(prompt, refuse):
            yield piece
        return
    if intent == "video":
        tracks = _collect_tracks(results)
        if not tracks:
            yield "这轮没有搜到视频结果。"
            return
        prompt, fallback = _video_intro_prompt(query, tracks, history_text)
        async for piece in stream_llm(prompt, fallback):
            yield piece
        yield "\n" + "\n".join(_video_list_lines(tracks))
        return
    if intent == "artist_info":
        prompt, fallback, source_urls = _artist_info_prompt(query, results, history_text)
        if prompt is None:
            yield "抱歉，暂时没找到相关信息。"
            return
        async for piece in stream_llm(prompt, fallback):
            yield piece
        if source_urls:
            yield "\n\n📎 参考来源：\n" + "\n".join(f"- {url}" for url in source_urls[:3])
        return
    if intent in {"artist_albums", "similar_artists", "taste_experiment", "taste", "journey"} or _is_knowledge_intent(intent):
        yield _compose_deterministic_answer(results, plan)
        return

    tracks = _select_listed_tracks(results, plan) or _collect_tracks(results)
    if not tracks:
        yield "这轮没有拿到可追溯的音乐候选；我不会用未核实歌名硬凑结果。"
        return
    tracks = tracks[: plan.target_count or (30 if intent == "playlist" else 12)]
    shortfall = bool(plan.target_count and len(tracks) < plan.target_count)
    prompt, fallback = _intro_prompt(query, tracks, plan, memory_query, shortfall, history_text)
    async for piece in stream_llm(prompt, fallback):
        yield piece
    if intent == "playlist":
        playlist = next((r.get("playlist") for r in results if r.get("type") == "playlist"), None)
        if playlist is not None:
            header = f"歌单《{playlist.name}》"
            if playlist.description:
                header += f"：{playlist.description}"
            yield "\n" + header
    yield "\n"
    for idx, track in enumerate(tracks, start=1):
        yield f"{idx}. 《{getattr(track, 'title', '')}》 - {getattr(track, 'artist', '') or '未知'}（{getattr(track, 'source', 'local')}）\n"


def _intro_prompt(
    query: str,
    tracks: list[Any],
    plan: AgentPlan,
    memory_query: str,
    shortfall: bool,
    history_text: str = "",
) -> tuple[str, str]:
    """构造推荐引言的 prompt + 确定性兜底文本（供非流式 _compose_intro 与流式共用）。"""
    fallback = f"我按在线优先策略整理了 {len(tracks)} 个可追溯候选："
    if shortfall:
        fallback += f"\n说明：你要求 {plan.target_count} 首，但当前候选只有 {len(tracks)} 首。"
    titles_preview = "、".join(getattr(t, "title", "") for t in tracks[:5])
    artists_preview = "、".join(dict.fromkeys(
        (getattr(t, "artist", "") or "").strip() for t in tracks[:8]
        if (getattr(t, "artist", "") or "").strip()
    ))
    mem_hint = f"用户偏好：{memory_query[:150]}" if memory_query else "暂无明确偏好记录"
    prompt = (
        f"用户请求：{query}\n"
        f"我已找到 {len(tracks)} 首真实候选，前几首：{titles_preview}\n"
        f"候选中实际出现的艺人：{artists_preview or '无明确艺人'}\n"
        f"{mem_hint}\n\n"
        "请写一句自然、有温度的推荐开场白（80字内），只能围绕用户本轮请求和实际候选。"
        "不要提最近对话、上一轮话题、旧歌手或旧专辑；本轮任务是独立的。"
        "记忆只作为弱背景，当前任务约束优先。除非用户本轮明确点名某艺人，且该艺人确实出现在候选中，"
        "否则不得声称会加入、延续或重点推荐该艺人的歌曲。"
        "不要列歌名，不要编造任何歌曲，不要用书名号。只输出这一句话。"
    )
    return prompt, fallback


def _chat_prompt(query: str, agent: AudioVisualAgent | None, history_text: str = "", user_id: str = "") -> str:
    """构造 chat 回复的 prompt（注入用户画像作为事实锚，供流式/非流式共用）。"""
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
    return (
        f"{history_hint}"
        f"{taste_hint}"
        f"用户说：{query}\n\n"
        "你是用户的私人音乐搭子，用中文自然、友好地回复。"
        "如果用户只是在打招呼，友好回应并提示你可以帮他做什么音乐相关的事。"
        "如果用户提到了某个歌手/歌曲/风格，可以结合用户偏好简短聊聊你的看法。"
        "不要每次用同一句话，语气要自然口语化。50-100字。\n"
        "不要编造排名、发行时间、销量等你不确定的具体数据。"
    )


def _discussion_prompt(
    query: str,
    tracks: list[Any],
    history_text: str = "",
) -> tuple[str | None, str]:
    """构造讨论回复的 prompt。无真实曲目时返回 (None, 拒绝模板)——反幻觉：拒绝编造。"""
    refuse = "我暂时没找到关于这个话题的可靠音乐数据，不想编，怕误导你。你可以试试换个关键词再问我。"
    if not tracks:
        return None, refuse
    items = []
    for t in tracks[:10]:
        title = getattr(t, "title", "")
        artist = getattr(t, "artist", "") or ""
        items.append(f"《{title}》{artist}")
    track_hint = f"已搜到的真实曲目（网易云验证过）：{'、'.join(items)}\n"
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
    return prompt, refuse


def _video_intro_prompt(query: str, tracks: list[Any], history_text: str = "") -> tuple[str, str]:
    """构造视频推荐开场白的 prompt + 兜底文本（供流式/非流式共用）。"""
    fallback = "我帮你搜到了这些视频："
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
    return prompt, fallback


def _video_list_lines(tracks: list[Any]) -> list[str]:
    """视频清单（确定性拼接，流式/非流式共用）。"""
    return [
        f"{idx}. 《{getattr(track, 'title', '')}》 - {getattr(track, 'artist', '') or '未知'}"
        f"（{getattr(track, 'source', 'local')}）"
        for idx, track in enumerate(tracks[:10], start=1)
    ]


def _compose_artist_albums_answer(
    results: list[dict[str, Any]],
) -> str:
    """artist_albums 意图回答：列出真实专辑清单。

    专辑名来自网易云回查、可追溯。用「专辑《名》」格式书写，且 collect_known_titles 已把
    专辑名纳入白名单——两层保险确保 Answer Guard 不会把专辑名当幻觉歌名删掉。
    """
    albums: list[dict[str, Any]] = []
    for r in results:
        if r.get("type") == "artist_albums":
            albums.extend(r.get("albums") or [])
    if not albums:
        return "暂时没拿到这位歌手的专辑，可能是接口限流，稍后再试一次。"
    artist = albums[0].get("artist") or ""
    if artist:
        intro = f"这是 {artist} 的 {len(albums)} 张专辑，点任意一张就能整张播放："
    else:
        intro = f"找到 {len(albums)} 张专辑，点任意一张就能整张播放："
    lines = []
    for idx, a in enumerate(albums, start=1):
        name = a.get("name", "")
        count = a.get("track_count")
        tail = f"（{count} 首）" if count else ""
        lines.append(f"{idx}. 专辑《{name}》 - {a.get('artist') or artist or '未知'}{tail}")
    return f"{intro}\n" + "\n".join(lines)


def _compose_similar_artists_answer(results: list[dict[str, Any]]) -> str:
    result = next((item for item in results if item.get("type") == "similar_artists"), None)
    artists = list((result or {}).get("artists") or [])
    seed = str((result or {}).get("seed_artist") or "这位歌手")
    if not artists:
        return f"当前曲库里没有足够标签支持判断与 {seed} 相似的歌手；我不会凭印象硬列名单。"
    lines = [f"按曲库中真实歌曲的曲风与情绪标签，和 {seed} 接近的歌手有："]
    for index, artist in enumerate(artists, start=1):
        reason = artist.get("reason") or "曲库标签相近"
        tracks = "、".join(artist.get("representative_tracks") or [])
        suffix = f"；曲库代表作：{tracks}" if tracks else ""
        lines.append(f"{index}. {artist.get('name', '')}（{reason}{suffix}）")
    return "\n".join(lines)


def _compose_taste_experiment_answer(results: list[dict[str, Any]]) -> str:
    exp = next((r.get("experiment") for r in results if r.get("type") == "taste_experiment"), None)
    if exp is None:
        return "这轮没有生成可追溯的品味实验候选；我不会硬凑结果。"
    lines = [
        f"我做了一个 Taste Lab 品味实验：{getattr(exp, 'hypothesis', '')}",
        "三档候选已经准备好，听完、跳过、喜欢或标记太安全/太远后，就能生成实验报告：",
    ]
    label_map = {"safe": "安全区", "stretch": "轻微越界", "bold": "大胆探索"}
    for segment in getattr(exp, "segments", []) or []:
        names = "、".join(f"《{item.track.title}》" for item in segment.tracks[:4])
        label = getattr(segment, "label", "") or label_map.get(segment.name, segment.name)
        lines.append(f"- {label}：{len(segment.tracks)} 首。{names or '暂无候选'}")
    result_summary = getattr(exp, "result_summary", "")
    if result_summary:
        lines.append(result_summary)
    return "\n".join(lines)


def _artist_info_prompt(
    query: str,
    results: list[dict[str, Any]],
    history_text: str = "",
) -> tuple[str | None, str, list[str]]:
    """构造 artist_info 的 prompt + 兜底（搜索摘要）+ 来源链接。无搜索结果返回 (None, '', [])。"""
    search_results = next(
        (r.get("search_results", []) for r in results if r.get("type") == "web_info_search"),
        [],
    )
    if not search_results:
        return None, "", []
    # 把搜索摘要拼接成上下文
    context_parts: list[str] = []
    for i, item in enumerate(search_results[:5], 1):
        title = item.get("title", "")
        content = item.get("content", "")
        if content:
            context_parts.append(f"[{i}] {title}\n{content}")
    search_context = "\n\n".join(context_parts)
    source_urls = [item["url"] for item in search_results if item.get("url")]
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
    return prompt, search_context, source_urls


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
