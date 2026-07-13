"""反射 / 空结果恢复 / 候选过滤 stage：从 nodes.py 拆出。

nodes.py 仅作为门面 re-export 本模块的符号，业务逻辑收敛于此。
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app.answer import collect_tracks as _collect_tracks
from app.config import settings
from app.graph._shared import _is_knowledge_intent, _merge_prompt_versions
from app.graph.budget import (
    _apply_turn_budget_degradation,
    _finalize_due_to_budget,
    _turn_budget_exceeded,
)
from app.graph.continuation import (
    _constraint_key,
    _merge_excluded_tracks,
    _query_with_entities,
    _strip_negative_query,
)
from app.graph.planning import _materialize_tool_stages
from app.intents import expand_content_negation, extract_content_negations
from app.llm.observability import capture_llm_stats, merge_runtime_metrics
from app.llm.structured import extract_json_dict
from app.models import AgentPlan, ExternalTrack, RecoveryDecision, StreamEvent
from app.prompts.reflect import (
    CANDIDATE_REFLECTION_SYSTEM,
    CANDIDATE_REFLECTION_USER,
    CANDIDATE_REFLECTION_VERSION,
)
from app.tools.contracts import ToolRisk, ToolStatus
from app.tools.registry import get_handler, get_tool_spec

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.agent import AudioVisualAgent
    from app.graph.state import AgentState


def route_after_reflect(state: AgentState) -> str:
    """reflect 后条件路由：候选不足且仍可补量时回 execute_tools。"""
    return "refine" if state.get("_need_refine") else "finalize"

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
    state = _apply_turn_budget_degradation(state)
    if _turn_budget_exceeded(state):
        return _finalize_due_to_budget(state)
    try:
        # 从 nodes 模块读取，使外部对 nodes._prepare_empty_result_recovery_async 的 monkeypatch 生效。
        from app.graph.nodes import _prepare_empty_result_recovery_async
        recovery = await _prepare_empty_result_recovery_async(agent, state)
        if recovery is not None:
            return recovery
        plan = state["plan"]
        if _is_knowledge_intent(plan.intent):
            return await _knowledge_reflect_async(agent, state)
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

async def _knowledge_reflect_async(agent: AudioVisualAgent, state: AgentState) -> AgentState:
    """知识意图自省（Reflexion）：确定性核对工具结果，决定是否补量重试。

    知识链路此前在 reflect 里被直接短路（passthrough），工具失败（resolve 空 / 档案降级）一路带到
    finalize。这里补一步**确定性核对**——零 LLM、零延迟——产出可观测判定：
      - resolve 空、档案靠 parametric 兜底 → 标注「已兜底」（答案可用，仅缺实体富化）；
      - 档案真正降级（partial 且非 parametric、正文是机械兜底/空）→ 若开 ``enable_knowledge_refine`` 且有
        预算，回 execute_tools 用清洗后的实体名重试一次 resolve（治「resolve 首轮空、原句带'的音乐路线'
        搜不到实体」）。重试有界（≤1 次）且受单轮预算墙约束。
    """
    plan = state["plan"]
    attempt = int(state.get("_refine_count", 0))
    outcomes = [o for o in state.get("tool_outcomes", []) if int(o.get("attempt", 0)) == attempt]
    by_tool = {str(o.get("tool") or ""): o for o in outcomes}
    resolve_outcome = by_tool.get("resolve_music_entity")
    resolve_empty = bool(resolve_outcome and resolve_outcome.get("status") == ToolStatus.EMPTY.value)

    dossier_result = next(
        (r for r in state.get("results", []) if r.get("type") in {"music_dossier", "music_compare"}),
        None,
    )
    dossier = (dossier_result or {}).get("dossier") or {}
    is_parametric = bool(dossier.get("is_parametric"))
    summary = str(dossier.get("summary") or "")
    # 真正降级：partial 且非 parametric（parametric 是完整直答，不算降级），正文为机械兜底/空。
    degraded = bool(dossier.get("partial")) and not is_parametric and (
        not summary or "未能合成" in summary or "证据归属不一致" in summary
    )

    if degraded:
        verdict = "[reflect][knowledge] 知识结果不足：档案降级、无可用正文"
    elif resolve_empty and is_parametric:
        verdict = "[reflect][knowledge] resolve 本轮空，已由 parametric 直答兜底（缺实体富化：career/library 命中弱）"
    elif resolve_empty:
        verdict = "[reflect][knowledge] resolve 本轮空，无实体可用"
    else:
        verdict = "[reflect][knowledge] 结果充分"

    trace = [*state.get("trace", []), verdict]
    events = [*state.get("events", []), StreamEvent(type="eval", content=verdict)]

    # 补量重试：仅档案真正降级 + 开关开 + 没重试过 + 有预算。重跑知识链路较重（~20-40s），故默认关。
    if (
        degraded
        and settings.enable_knowledge_refine
        and attempt < 1
        and not _turn_budget_exceeded(state)
    ):
        entities = list(getattr(plan.retrieval_plan, "entities", []) or [])
        cleaned = next((e.strip() for e in entities if e and e.strip()), state["query"])
        revised = _revise_knowledge_plan_for_retry(plan, cleaned, state.get("top_k", 5))
        note = f"{verdict}；回环重试 resolve（query={cleaned}）"
        return {
            **state,
            "plan": revised,
            "trace": [*state.get("trace", []), note],
            "events": [*events, StreamEvent(
                type="refine",
                content="知识结果不足，用清洗后的实体名重试 resolve。",
                payload={"search_query": cleaned, "attempt": attempt + 1},
            )],
            "_need_refine": True,
        }
    return {**state, "trace": trace, "events": events, "_need_refine": False}

def _revise_knowledge_plan_for_retry(plan: AgentPlan, cleaned_query: str, top_k: int) -> AgentPlan:
    """重试知识链路：把检索词换成清洗后的实体名，重建 [resolve→metadata→web_knowledge→build] stages。"""
    retrieval = plan.retrieval_plan.model_copy(update={"search_query": cleaned_query})
    revised = plan.model_copy(update={
        "retrieval_plan": retrieval,
        "reasoning_summary": f"{plan.reasoning_summary}；知识自省：resolve 首轮空，用实体名「{cleaned_query}」重试。",
    })
    return _materialize_tool_stages(revised, cleaned_query, top_k)

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
    if (
        decision is None
        and not context.get("recovery_llm_used")
        and context.get("budget_degrade_level") != "soft"
        and context.get("budget_degrade_level") != "hard"
    ):
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
    degrade_level = ((state.get("context") or {}).get("budget_degrade_level") or "").lower()
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
        if degrade_level == "hard":
            return RecoveryDecision(
                action="retry",
                reason="剩余预算不足，跳过在线重搜，直接切到可追溯的本地检索。",
                search_query=_positive_recovery_query(state),
                calls=_local_recovery_calls(plan.intent),
            )
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
        # 从 nodes 模块读取，使外部对 nodes.select_llm 的 monkeypatch 生效。
        from app.graph.nodes import select_llm
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
    # 从 nodes 模块读取，使外部对 nodes.select_llm 的 monkeypatch 生效。
    from app.graph.nodes import select_llm
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
