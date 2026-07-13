from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any

from app.answer import infer_count as _infer_count
from app.config import settings
from app.context import ContextBudgetManager, ContextSource
from app.graph._shared import (
    _format_prompt_versions,
    _is_knowledge_intent,
    _merge_prompt_versions,
)
from app.graph.continuation import (
    _apply_dialogue_continuation,
    _constraint_key,
    _query_with_entities,
    _strip_negative_query,
    _without_negative_constraints,
)
from app.graph.tag_rules import extract_tags
from app.intents import (
    extract_content_negations,
    get_intent,
    is_allowed_multi_intent_pair,
    match_intent_by_keywords,
)
from app.llm.observability import capture_llm_stats, empty_runtime_metrics
from app.llm.structured import extract_json_dict
from app.models import AgentPlan, QueryPlanPayload, RetrievalPlan, StreamEvent, ToolStage
from app.prompts import QUERY_PLAN_SYSTEM, QUERY_PLAN_VERSION
from app.tools.contracts import ToolCall
from app.tools.registry import get_handler

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

    turn_deadline_at = time.monotonic() + settings.turn_budget_seconds
    # ``deadline_at`` 是 ToolRuntime 消费的硬 deadline。普通路径预留最终组装时间；
    # 知识意图会在规划完成后以自己的更长预算覆盖它。
    execution_deadline_at = max(
        time.monotonic() + 0.05,
        turn_deadline_at - min(settings.turn_finalize_reserve_seconds, settings.turn_budget_seconds),
    )
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
        "turn_deadline_at": turn_deadline_at,
        "deadline_at": execution_deadline_at,
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
    profile_text = (
        agent.profile_context_text(state["user_id"])
        if state.get("user_id") and hasattr(agent, "profile_context_text")
        else ""
    )
    # 从 nodes 模块读取，使外部对 nodes.plan_with_llm_with_meta_async 的 monkeypatch 生效。
    from app.graph.nodes import plan_with_llm_with_meta_async
    timeout = None
    deadline_at = context.get("deadline_at")
    if deadline_at:
        timeout = max(0.0, float(deadline_at) - time.monotonic())
    try:
        if timeout is not None:
            if timeout <= 0:
                raise TimeoutError
            plan, prompt_versions, runtime_metrics = await asyncio.wait_for(
                plan_with_llm_with_meta_async(
                    agent,
                    state["query"],
                    context.get("history_text", ""),
                    context.get("memory_query", ""),
                    profile_text=profile_text,
                ),
                timeout=timeout,
            )
        else:
            plan, prompt_versions, runtime_metrics = await plan_with_llm_with_meta_async(
                agent,
                state["query"],
                context.get("history_text", ""),
                context.get("memory_query", ""),
                profile_text=profile_text,
            )
    except TimeoutError:
        logger.warning("规划阶段达到请求 deadline，改用确定性计划")
        plan, prompt_versions, runtime_metrics = None, {}, empty_runtime_metrics()
        state = {
            **state,
            "trace": [*state.get("trace", []), "[budget] 规划超时，改用确定性计划。"],
            "events": [*state.get("events", []), StreamEvent(
                type="eval", content="规划耗时过长，已使用快速确定性路径。",
                payload={"planning_timed_out": True},
            )],
        }
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
    if settings.enable_multi_intent and plan.is_multi_intent:
        plan = _merge_multi_intent_stages(plan, query, state.get("top_k", 5))
    context = dict(state.get("context") or {})
    # primary 是知识意图，或（多意图下）某个 sub_plan 是知识意图，都要挂知识延迟预算，
    # 让 build_music_dossier / web_knowledge_search 拿到 deadline，避免跑满无界。
    _has_knowledge = _is_knowledge_intent(plan.intent) or any(
        _is_knowledge_intent(sp.intent) for sp in plan.sub_plans
    )
    if _has_knowledge:
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
        # 强搜索 provider 取代 review_search 作主检索：claims+sources+citations，web 空时内部回退 review_search。
        web_knowledge = ToolCall(name="web_knowledge_search", arguments=_planned_arguments("web_knowledge_search", query, plan, top_k))
        build = ToolCall(name="build_music_dossier", arguments=_planned_arguments("build_music_dossier", query, plan, top_k))
        return plan.model_copy(update={
            "tools_needed": ["resolve_music_entity", "music_metadata_lookup", "web_knowledge_search", "build_music_dossier"],
            "stages": [
                ToolStage(calls=[resolve], parallel=False),
                ToolStage(calls=[metadata, web_knowledge], parallel=True),
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


def _merge_multi_intent_stages(plan: AgentPlan, query: str, top_k: int) -> AgentPlan:
    """把 primary 与各 sub_plan 的工具链合并成一组并行 stages（wall-time ≈ max）。

    做法：先给每个 sub_plan 各自 materialize stages，再按「深度」把各链同层 stage
    压进同一个合并 stage——第 N 层合并 stage 只依赖第 <N 层的结果（各链内部依赖天然满足），
    于是两条链在墙钟上并行推进，总耗时 ≈ 最长单链，而非各链之和。

    共享工具去重：同一合并层里若出现 (name, arguments) 完全一致的调用（典型
    resolve_music_entity），只保留一个；arguments 不一致则各跑各的（正确但多一次廉价调用）。
    plan.intent 保持 primary（让 composer 的 track ladder 继续走）；tools_needed 取并集。
    """
    sub_materialized = [_materialize_tool_stages(sp, query, top_k) for sp in plan.sub_plans]
    chains: list[list[ToolStage]] = [list(plan.stages or []), *[list(sp.stages or []) for sp in sub_materialized]]
    depth = max((len(chain) for chain in chains), default=0)

    merged_stages: list[ToolStage] = []
    for level in range(depth):
        calls: list[ToolCall] = []
        seen: set[tuple[str, str]] = set()
        for chain in chains:
            if level >= len(chain):
                continue
            for call in chain[level].calls:
                key = (call.name, json.dumps(call.arguments, sort_keys=True, ensure_ascii=False, default=str))
                if key in seen:
                    continue
                seen.add(key)
                calls.append(call)
        if calls:
            merged_stages.append(ToolStage(calls=calls, parallel=len(calls) > 1))

    tools_union: list[str] = []
    for stage in merged_stages:
        for call in stage.calls:
            if call.name not in tools_union:
                tools_union.append(call.name)
    # sub_plans 更新为已 materialize 的版本，供后续 composition 读取其 stages/retrieval。
    return plan.model_copy(update={"stages": merged_stages, "tools_needed": tools_union, "sub_plans": sub_materialized})


def _planned_arguments(name: str, query: str, plan: AgentPlan, top_k: int) -> dict[str, Any]:
    handler = get_handler(name) or name
    count = plan.target_count or top_k
    entity_query = _query_with_entities(query, plan)
    if handler == "recommend":
        return {"query": query, "search_query": entity_query, "top_k": count}
    if handler == "search":
        return {"query": entity_query, "include_external": True}
    if handler == "web_music_search":
        return {"query": entity_query, "top_k": count}
    if handler == "playlist":
        args: dict[str, Any] = {"instruction": entity_query}
        if plan.target_count is not None:
            args["target_count"] = plan.target_count
        return args
    if handler == "playlist_repair":
        return {"instruction": query, "target": "上一轮歌单或推荐"}
    if handler == "taste_experiment":
        return {"prompt": query, "total": plan.target_count or 12}
    if handler == "taste_shift_detector":
        return {"window_recent_days": 30, "window_baseline_days": 90}
    if handler == "music_fact_check":
        return {"query": query}
    if handler == "recommend_explainer":
        return {"query": query}
    if handler in {
        "resolve_music_entity", "music_metadata_lookup", "review_search", "web_knowledge_search", "build_music_dossier",
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
        from app.graph.nodes import _infer_aux_arguments
        return _infer_aux_arguments(handler, query, plan)
    return {}


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


async def plan_with_llm_async(
    agent: AudioVisualAgent,
    query: str,
    history_text: str = "",
    memory_text: str = "",
    profile_text: str = "",
) -> AgentPlan | None:
    plan, _, _ = await plan_with_llm_with_meta_async(agent, query, history_text, memory_text, profile_text)
    return plan


async def plan_with_llm_with_meta_async(
    agent: AudioVisualAgent,
    query: str,
    history_text: str = "",
    memory_text: str = "",
    profile_text: str = "",
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
        # 画像仪表盘（app/profile/）：与听歌历史互补，带场景偏好/探索风格/画像级排除，
        # 让 LLM 规划 search_query/entities 时参考「专辑品位」。同样软参考、不覆盖本轮明确要求。
        if profile_text.strip() and not settings.mock_mode:
            sections.append(
                "【用户画像·品味仪表盘（软参考，不得覆盖本轮明确要求）】\n"
                f"{profile_text[:400]}"
            )
        sections.append(f"【本轮输入】\n{query}" if sections else query)
        # 从 nodes 模块读取，使外部对 nodes.select_llm 的 monkeypatch 生效。
        from app.graph.nodes import select_llm
        llm = select_llm(agent, "fast")
        raw = await llm.agenerate(
            "\n\n".join(sections), system=QUERY_PLAN_SYSTEM, temperature=0.1,
        )
    except Exception as _plan_exc:
        import httpx as _httpx
        if isinstance(_plan_exc, _httpx.HTTPStatusError) and _plan_exc.response.status_code in (401, 403):
            logger.error(
                "[LLM_AUTH_ERROR] LLM API key 无效或无权限（%s %s）——检查 .env 的 LLM_API_KEY",
                _plan_exc.response.status_code, _plan_exc.response.text[:120],
            )
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
    exclusions = extract_content_negations(query)
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
        excluded_terms=exclusions,
    )
    return AgentPlan(
        intent=intent,
        strategy=spec.strategy_for(retrieval.use_web),
        tools_needed=spec.tools_for(retrieval.use_web),
        target_count=payload.target_count or _infer_count(query),
        online_required=retrieval.use_web,
        reasoning_summary=payload.reasoning.strip() or spec.summary,
        retrieval_plan=retrieval,
        sub_plans=_build_secondary_sub_plans(payload, intent, query),
    )


def _build_secondary_sub_plans(payload: QueryPlanPayload, primary_intent: str, query: str) -> list[AgentPlan]:
    """把 LLM 检测到的 secondary 意图转成一个 sub AgentPlan（≤1 个）。

    三重闸门：flag 关 / 无 secondary / pair 不在白名单 → 返回空列表（=单意图今天行为）。
    sub_plan 自己不再递归带 secondary（双意图上限）。
    """
    secondary = payload.secondary
    if not settings.enable_multi_intent or secondary is None or not secondary.intent:
        return []
    if not is_allowed_multi_intent_pair(primary_intent, secondary.intent):
        return []
    spec = get_intent(secondary.intent)
    entities = list(secondary.entities) or list(payload.entities)
    search_query = secondary.search_query.strip() or (entities[0] if entities else query)
    retrieval = RetrievalPlan(
        use_local=False,
        use_vector=False,
        use_web=spec.online_default,
        entities=entities,
        search_query=search_query,
    )
    return [AgentPlan(
        intent=secondary.intent,
        strategy=spec.strategy_for(retrieval.use_web),
        tools_needed=spec.tools_for(retrieval.use_web),
        online_required=retrieval.use_web,
        reasoning_summary=spec.summary,
        retrieval_plan=retrieval,
    )]


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
