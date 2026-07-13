"""工具执行 stage：把工具调用、运行时记录、web 兜底从 nodes.py 拆出。

nodes.py 仅作为门面 re-export 本模块的符号，业务逻辑收敛于此。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.answer import collect_tracks as _collect_tracks
from app.graph.planning import _planned_arguments
from app.models import AgentPlan, StreamEvent, ToolOutcome, ToolStage
from app.tools.contracts import ToolCall, ToolError, ToolResult, ToolRisk, ToolStatus
from app.tools.registry import get_handler, get_tool_spec

if TYPE_CHECKING:
    from app.agent import AudioVisualAgent
    from app.graph.state import AgentState


async def web_fallback_async(agent: AudioVisualAgent, state: AgentState) -> AgentState:
    plan = state["plan"]
    query = state["query"]
    user_id = state["user_id"]
    top_k = state.get("top_k", 5)
    results = list(state.get("results", []))
    trace = [*state.get("trace", []), "[web_fallback] 本地候选不足，触发联网兜底补搜。"]
    events = [*state.get("events", []), StreamEvent(type="eval", content="本地候选不足，联网兜底补搜。")]
    outcomes = list(state.get("tool_outcomes", []))
    call = ToolCall(name="web_music_search", arguments=_planned_arguments("web_music_search", query, plan, top_k))
    local_results: list[dict[str, Any]] = []
    runtime_result = await _run_tool_async_safely(
        agent,
        call,
        plan,
        query,
        user_id,
        top_k,
        results,
        local_results,
        trace,
        events,
        state.get("thread_id") or f"{user_id}:default",
        state.get("run_id") or "",
        bool(state.get("_interrupt_enabled")),
    )
    results.extend(local_results)
    if runtime_result is not None:
        outcomes.append(_tool_outcome(call, runtime_result, state.get("_refine_count", 0)))
    return {
        **state,
        "results": results,
        "trace": trace,
        "events": events,
        "tool_outcomes": outcomes,
        "_need_web_fallback": False,
    }


def route_after_execute(state: AgentState) -> str:
    """条件路由：候选不足 → web_fallback，否则 → reflect。"""
    return "web_fallback" if state.get("_need_web_fallback") else "reflect"


def _needs_web_fallback(plan: AgentPlan, results: list[dict[str, Any]], executed: set[str]) -> bool:
    if "web_music_search" in executed or not plan.online_required:
        return False
    if plan.intent not in {"recommend", "search", "playlist"}:
        return False
    verified = [
        t for t in _collect_tracks(results) if getattr(t, "source", "local") in {"netease", "bilibili", "youtube"}
    ]
    need = plan.target_count or 3
    return len(verified) < need


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
        events.append(
            StreamEvent(
                type="error",
                content=f"{handler} 暂时不可用，已跳过该工具。",
                payload={"tool": handler, "error": error_message},
            )
        )
    else:
        trace.append(f"[{handler}] {summary}")
        if handler == "import_netease_playlist":
            trace.append(f"[import] {summary}")
    trace.append(
        f"[tool_status] tool={handler} status={runtime_result.status.value} candidates={len(runtime_result.cards)}"
    )
    if runtime_result.status == ToolStatus.CONFIRMATION_REQUIRED:
        checkpoint_store.put(
            call.call_id,
            thread_id,
            user_id,
            handler,
            arguments,
            query,
        )
        events.append(
            StreamEvent(
                type="confirmation_required",
                content=summary,
                payload={"action_id": call.call_id, "tool": handler, "arguments": arguments},
            )
        )
    if handler == "artist_albums":
        for album in runtime_result.data.get("albums", []):
            events.append(StreamEvent(type="album_card", content=album.get("name", ""), payload={"album": album}))
    if handler == "similar_artists":
        for artist in runtime_result.data.get("artists", []):
            events.append(StreamEvent(type="artist_card", content=artist.get("name", ""), payload=artist))
    if handler == "build_music_dossier" and runtime_result.data.get("dossier"):
        events.append(
            StreamEvent(
                type="dossier",
                content=runtime_result.data.get("answer", "已生成音乐档案。"),
                payload={"dossier": runtime_result.data.get("dossier")},
            )
        )
        for artist in runtime_result.data.get("artist_cards", []) or []:
            events.append(StreamEvent(type="artist_card", content=artist.get("name", ""), payload=artist))
        dossier = runtime_result.data.get("dossier") or {}
        entity = dossier.get("entity") or {}
        result_type = str(runtime_result.data.get("type") or "")
        # 专辑解读：把这张专辑本身作为卡片下发（封面/曲目/风格），出现在乐评下方。
        # 对比(music_compare)不下发：它的 entity 是两个比较对象之一，且 resolve 失败时类型推断常把
        # 艺人名误判成 album（实测 The Weeknd/drake → type=album）→ 出「The Weeknd / 未知歌手」错卡。
        if entity.get("type") == "album" and result_type != "music_compare":
            ext = entity.get("external_ids") or {}
            key_tracks = dossier.get("key_tracks") or []
            main_album = {
                "id": ext.get("netease_album") or ext.get("musicbrainz") or ext.get("spotify") or "",
                "name": entity.get("name", ""),
                "artist": entity.get("artist", ""),
                "image": entity.get("image", ""),
                "track_count": len(key_tracks) or None,
                "tracks": [
                    {
                        "title": t.get("title", ""),
                        "artist": t.get("artist") or entity.get("artist", ""),
                        "source": t.get("source", "local"),
                        "source_id": t.get("source_id", ""),
                    }
                    for t in key_tracks
                ],
                "genres": dossier.get("style_tags") or [],
            }
            if main_album["name"]:
                events.append(StreamEvent(type="album_card", content=main_album["name"], payload={"album": main_album}))
        # 聆听路线的真实专辑作为可播放/可收藏卡片下发（复用前端 album_card 渲染）。
        for album in dossier.get("related_albums", []):
            events.append(StreamEvent(type="album_card", content=album.get("name", ""), payload={"album": album}))
    if handler == "build_sample_dossier" and runtime_result.data.get("sample_dossier"):
        events.append(
            StreamEvent(
                type="sample_relations",
                content=runtime_result.data.get("answer", "已生成采样溯源结果。"),
                payload={
                    "sample_dossier": runtime_result.data.get("sample_dossier"),
                    "relations": runtime_result.data.get("sample_relations") or [],
                    "source_cards": runtime_result.data.get("source_cards") or [],
                },
            )
        )
    if runtime_result.cards:
        payload: dict[str, Any] = {"count": len(runtime_result.cards), "cards": runtime_result.cards}
        if handler == "taste_experiment":
            experiment = runtime_result.data.get("experiment")
            payload["taste_experiment"] = (
                experiment.model_dump(mode="json") if hasattr(experiment, "model_dump") else experiment
            )
        if handler == "journey":
            payload["journey"] = runtime_result.data.get("journey")
        events.append(
            StreamEvent(type="candidates", content=f"{handler} {len(runtime_result.cards)} 个结果", payload=payload)
        )
    events.append(
        StreamEvent(
            type="tool_result",
            content=summary,
            payload={"tool": handler, "status": runtime_result.status.value},
        )
    )
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
            agent,
            call,
            plan,
            query,
            user_id,
            top_k,
            shared_results,
            local_results,
            local_trace,
            local_events,
            thread_id,
            run_id,
            interrupt_enabled,
            state_context,
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
            agent,
            call,
            plan,
            query,
            user_id,
            top_k,
            prior_results,
            results,
            trace,
            events,
            thread_id,
            run_id,
            interrupt_enabled,
            state_context,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        if any(base.__name__ == "GraphBubbleUp" for base in type(exc).__mro__):
            raise
        handler = get_handler(call.name) or call.name
        result = ToolResult(
            tool=handler,
            status=ToolStatus.ERROR,
            error=ToolError(kind=type(exc).__name__, message=str(exc)),
        )
        trace.extend(
            [
                f"[tool_error] {handler} 失败，已跳过：{exc}",
                f"[tool_status] tool={handler} status=error candidates=0",
            ]
        )
        events.append(
            StreamEvent(
                type="error",
                content=f"{handler} 暂时不可用，已跳过该工具。",
                payload={"tool": handler, "error": str(exc)},
            )
        )
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
    from app.services.tools import checkpoint_store, tool_runtime
    from app.tools.contracts import ToolContext

    arguments = call.arguments or _planned_arguments(handler, query, plan, top_k)
    call = call.model_copy(update={"name": handler, "arguments": arguments})
    plan_payload = plan.model_dump(mode="json")
    plan_payload["_excluded_tracks"] = getattr(plan, "_excluded_tracks", None) or []
    plan_payload["_excluded_artists"] = getattr(plan, "_excluded_artists", None) or []
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
                "action_id": call.call_id,
                "tool": handler,
                "arguments": arguments,
                "query": query[:200],
            }
            created = checkpoint_store.put(
                call.call_id,
                thread_id,
                user_id,
                handler,
                arguments,
                query,
            )
            if created:
                get_stream_writer()(
                    StreamEvent(
                        type="confirmation_required",
                        content="这是外部账号写操作，需要明确确认后执行。",
                        payload=action_payload,
                    ).model_dump(mode="json")
                )
            decision = interrupt(action_payload)
            approved = (
                isinstance(decision, dict)
                and decision.get("action_id") == call.call_id
                and decision.get("approved") is True
            )
            if not approved:
                runtime_result = ToolResult(
                    tool=handler,
                    status=ToolStatus.CANCELLED,
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
    latency_budget = dict(state_context.get("latency_budget") or {})
    budget_degrade_level = state_context.get("budget_degrade_level")
    if budget_degrade_level:
        latency_budget["budget_degrade_level"] = budget_degrade_level
    if runtime_result is None:
        runtime_result = await tool_runtime.execute(
            call,
            ToolContext(
                thread_id=thread_id,
                user_id=user_id,
                query=query,
                plan=plan_payload,
                prior_results=prior_results,
                confirmation=confirmation,
                agent=agent,
                deadline_at=state_context.get("deadline_at"),
                latency_budget=latency_budget,
                **context_kwargs,
            ),
        )
    return _record_runtime_result(
        handler,
        call,
        arguments,
        runtime_result,
        results,
        trace,
        events,
        checkpoint_store=checkpoint_store,
        thread_id=thread_id,
        user_id=user_id,
        query=query,
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
        window = (
            "week"
            if "上周" in query or "最近一周" in query
            else "month"
            if "本月" in query or "最近一个月" in query
            else "recent"
        )
        return {
            "window": window,
            "group_by": "artist" if "歌手" in query else "track",
            "top_k": plan.target_count or 10,
        }
    if handler == "list_my_playlists":
        return {}
    if handler == "find_on_platform":
        platform = (
            "youtube"
            if "youtube" in lowered
            else "bilibili"
            if any(token in lowered for token in ("b站", "bilibili"))
            else "netease"
        )
        return {
            "title": entities[0] if entities else query,
            "artist": entities[1] if len(entities) > 1 else "",
            "platform": platform,
        }
    if handler == "lyrics":
        return {"title": entities[0] if entities else query, "artist": entities[1] if len(entities) > 1 else ""}
    if handler == "audio_features":
        return {"title": entities[0] if entities else query}
    if handler == "concert_events":
        return {"artist": entities[0] if entities else query}
    return {"confirm": False}
