from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from app.compound import is_compound_task
from app.graph.decompose import SubTask, decompose_compound_async, summarize_subtasks
from app.graph.nodes import (
    execute_tools_async,
    finalize_stream_async,
    load_context,
    plan_intent_async,
    reflect_async,
    route_after_execute,
    route_after_reflect,
    web_fallback_async,
)
from app.graph.state import AgentState
from app.llm.observability import empty_runtime_metrics, format_runtime_metrics, merge_runtime_metrics
from app.llm.routing import select_llm
from app.models import AgentAnswer, StreamEvent
from app.prompts import COMPOUND_SYNTH_SYSTEM, COMPOUND_SYNTH_USER, COMPOUND_SYNTH_VERSION

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.agent import AudioVisualAgent


class AgentGraphRunner:
    def __init__(self, agent: AudioVisualAgent) -> None:
        self.agent = agent
        self.compiled_stream = _try_build_streaming_langgraph(agent)
        self._checkpoint_connection = None

    async def initialize_checkpointing(self, path: str) -> None:
        if self._checkpoint_connection is not None:
            return
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        from app.tools.checkpoint_serde import SanitizingCheckpointSerializer

        connection = await aiosqlite.connect(path)
        self._checkpoint_connection = connection
        self.compiled_stream = _try_build_streaming_langgraph(
            agent=self.agent,
            checkpointer=AsyncSqliteSaver(connection, serde=SanitizingCheckpointSerializer()),
        )

    async def close(self) -> None:
        if self._checkpoint_connection is not None:
            await self._checkpoint_connection.close()
            self._checkpoint_connection = None

    @property
    def checkpointing_ready(self) -> bool:
        return self._checkpoint_connection is not None and self.compiled_stream is not None

    async def resume(
        self,
        *,
        thread_id: str,
        action_id: str,
        approved: bool,
    ) -> AsyncIterator[StreamEvent]:
        if not self.checkpointing_ready:
            raise RuntimeError("LangGraph checkpointing is not available")
        from langgraph.types import Command

        config = {"configurable": {"thread_id": thread_id}}
        command = Command(resume={"action_id": action_id, "approved": approved})
        async for chunk in self.compiled_stream.astream(command, config=config, stream_mode="custom"):
            if isinstance(chunk, StreamEvent):
                yield chunk
            elif isinstance(chunk, dict):
                yield StreamEvent.model_validate(chunk)

    async def ainvoke(
        self,
        user_id: str,
        asset_id: str | None,
        query: str,
        history: list[dict[str, Any]] | None = None,
        top_k: int = 5,
        thread_id: str | None = None,
        run_id: str | None = None,
    ) -> AgentAnswer:
        final: StreamEvent | None = None
        async for event in self.astream(
            user_id, asset_id, query, history=history, top_k=top_k,
            thread_id=thread_id, run_id=run_id,
        ):
            if event.type == "final":
                final = event
        if final is None:
            raise RuntimeError("LangGraph completed without a final event")
        return AgentAnswer.model_validate(final.payload)

    async def astream(
        self,
        user_id: str,
        asset_id: str | None,
        query: str,
        history: list[dict[str, Any]] | None = None,
        top_k: int = 5,
        thread_id: str | None = None,
        run_id: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """The sole production execution path."""
        resolved_thread = thread_id or f"{user_id}:default"
        if is_compound_task(query):
            async for event in self._astream_compound(
                user_id, asset_id, query, history or [], top_k, resolved_thread, run_id or "",
            ):
                yield event
            return
        async for event in self._astream_single(
            user_id, asset_id, query, history or [], top_k, resolved_thread, run_id or "",
        ):
            yield event

    async def _astream_single(
        self,
        user_id: str,
        asset_id: str | None,
        query: str,
        history: list[dict[str, Any]],
        top_k: int,
        thread_id: str,
        run_id: str,
    ) -> AsyncIterator[StreamEvent]:
        state: AgentState = {
            "user_id": user_id,
            "run_id": run_id,
            "thread_id": thread_id,
            "asset_id": asset_id,
            "query": query,
            "history": history,
            "top_k": top_k,
            "_refine_count": 0,
            "_interrupt_enabled": self.checkpointing_ready,
        }
        if self.compiled_stream is None:
            raise RuntimeError("LangGraph is unavailable; no secondary orchestrator is configured")
        config = {"configurable": {"thread_id": state["thread_id"]}}
        try:
            async for chunk in self.compiled_stream.astream(state, config=config, stream_mode="custom"):
                if isinstance(chunk, StreamEvent):
                    yield chunk
                elif isinstance(chunk, dict):
                    yield StreamEvent.model_validate(chunk)
        except Exception:  # noqa: BLE001
            logger.exception("LangGraph stream execution failed")
            trace = ["[graph_error] 本轮图执行失败，已输出保守兜底回答。"]
            answer = AgentAnswer(
                answer="这轮处理时遇到临时问题，我没有生成未经核实的音乐结果。请稍后重试，或换一个更具体的描述。",
                evidences=[],
                agent_trace=trace,
                fallback_reason="graph_execution_failed",
                trace_summary={"intent": "unknown", "tools": [], "sources": [], "fallback": "graph_execution_failed"},
            )
            yield StreamEvent(type="error", content="处理暂时失败，已返回保守结果。")
            yield StreamEvent(type="final", content=answer.answer, payload=answer.model_dump(mode="json"))

    async def _astream_compound(
        self,
        user_id: str,
        asset_id: str | None,
        query: str,
        history: list[dict[str, Any]],
        top_k: int,
        thread_id: str,
        run_id: str,
    ) -> AsyncIterator[StreamEvent]:
        subtasks, decompose_versions, decompose_metrics = await decompose_compound_async(
            self.agent, query, history,
        )
        yield StreamEvent(type="plan", content=summarize_subtasks(subtasks))
        scratchpad: dict[str, Any] = {}
        answers: list[AgentAnswer] = []
        cards: list[dict[str, Any]] = []
        trace = [f"[compound_plan] {summarize_subtasks(subtasks)}"]
        for index, task in enumerate(subtasks, start=1):
            task_query = _hydrate_subtask_query(task, scratchpad)
            final: StreamEvent | None = None
            async for event in self._astream_single(
                user_id, asset_id, task_query, history, top_k,
                f"{thread_id}:sub:{index}", f"{run_id}:sub:{index}" if run_id else "",
            ):
                if event.type == "final":
                    final = event
                else:
                    yield event
            if final is None:
                continue
            answer = AgentAnswer.model_validate(final.payload)
            answers.append(answer)
            cards.extend((final.payload or {}).get("cards") or [])
            _update_compound_scratchpad(scratchpad, task, answer, index, len(subtasks))
            trace.append(f"[compound_step] {index}/{len(subtasks)} {task.intent}: {task_query}")
            trace.extend(f"[subtask {index}] {line}" for line in answer.agent_trace)
        text, synth_versions, synth_metrics = await _compose_compound_answer_async(
            self.agent, query, subtasks, answers,
        )
        last = answers[-1] if answers else AgentAnswer(answer="", evidences=[])
        versions = _merge_prompt_version_maps(
            *[answer.prompt_versions for answer in answers], decompose_versions, synth_versions,
        )
        metrics = merge_runtime_metrics(
            empty_runtime_metrics(), *[answer.runtime_metrics for answer in answers],
            decompose_metrics, synth_metrics,
        )
        trace.append(f"[meta] {format_runtime_metrics(metrics)}")
        answer = AgentAnswer(
            answer=text,
            evidences=[ev for item in answers for ev in item.evidences][:8],
            recommended_segments=last.recommended_segments,
            recommended_tracks=_merge_compound_recommended_tracks(answers),
            prompt_versions=versions,
            runtime_metrics=metrics,
            memory_updated=any(item.memory_updated for item in answers),
            agent_trace=trace,
            pending_goal=last.pending_goal,
            goal_progress=last.goal_progress,
            fallback_reason=last.fallback_reason,
        )
        payload = answer.model_dump(mode="json")
        merged_cards = _dedupe_cards(cards)
        if merged_cards:
            payload["cards"] = merged_cards
        yield StreamEvent(type="final", content=text, payload=payload)


def build_agent_graph(agent: AudioVisualAgent) -> AgentGraphRunner:
    return AgentGraphRunner(agent)


def _try_build_streaming_langgraph(agent: AudioVisualAgent, checkpointer=None):
    try:
        from langgraph.config import get_stream_writer
        from langgraph.graph import END, StateGraph
    except Exception:
        return None

    def emitting(node):
        def wrapped(state: AgentState) -> AgentState:
            before = len(state.get("events", []))
            next_state = node(agent, state)
            writer = get_stream_writer()
            for event in next_state.get("events", [])[before:]:
                writer(event.model_dump(mode="json"))
            return next_state

        return wrapped

    def emitting_async(node):
        async def wrapped(state: AgentState) -> AgentState:
            before = len(state.get("events", []))
            next_state = await node(agent, state)
            writer = get_stream_writer()
            for event in next_state.get("events", [])[before:]:
                writer(event.model_dump(mode="json"))
            return next_state

        return wrapped

    async def stream_finalize(state: AgentState) -> AgentState:
        writer = get_stream_writer()
        final_event = None
        async for event in finalize_stream_async(agent, state):
            writer(event.model_dump(mode="json"))
            final_event = event
        if final_event and final_event.type == "final":
            try:
                answer = AgentAnswer.model_validate(final_event.payload)
                return {**state, "answer": answer}
            except Exception:
                pass
        return state

    graph = StateGraph(AgentState)
    graph.add_node("load_context", emitting(load_context))
    graph.add_node("plan_intent", emitting_async(plan_intent_async))
    graph.add_node("execute_tools", emitting_async(execute_tools_async))
    graph.add_node("web_fallback", emitting_async(web_fallback_async))
    graph.add_node("reflect", emitting_async(reflect_async))
    graph.add_node("finalize", stream_finalize)
    graph.set_entry_point("load_context")
    graph.add_edge("load_context", "plan_intent")
    graph.add_edge("plan_intent", "execute_tools")
    graph.add_conditional_edges("execute_tools", lambda state: route_after_execute(state), {"web_fallback": "web_fallback", "reflect": "reflect"})
    graph.add_edge("web_fallback", "reflect")
    graph.add_conditional_edges("reflect", lambda state: route_after_reflect(state), {"refine": "execute_tools", "finalize": "finalize"})
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def _hydrate_subtask_query(task: SubTask, scratchpad: dict[str, Any]) -> str:
    query = task.query
    if not task.depends_on_prev:
        return query
    lines = [query]
    last_query = (scratchpad.get("last_query") or "").strip()
    last_summary = (scratchpad.get("last_summary") or "").strip()
    last_answer = (scratchpad.get("last_answer") or "").strip()
    if last_query:
        lines.append(f"上一步任务：{last_query[:160]}")
    if last_summary:
        lines.append(f"上一步摘要：{last_summary[:160]}")
    elif last_answer:
        lines.append(f"上一步结果参考：{last_answer[:240]}")
    return "\n\n".join(lines)


def _update_compound_scratchpad(
    scratchpad: dict[str, Any],
    task: SubTask,
    answer: AgentAnswer,
    index: int,
    total: int,
) -> None:
    scratchpad["last_query"] = task.query
    scratchpad["last_answer"] = answer.answer
    scratchpad["last_goal_progress"] = answer.goal_progress
    summary = answer.answer.strip().splitlines()[0] if answer.answer.strip() else ""
    scratchpad["last_summary"] = summary
    scratchpad["completed_subtasks"] = index
    scratchpad["remaining_subtasks"] = max(total - index, 0)


async def _compose_compound_answer_async(
    agent: AudioVisualAgent,
    query: str,
    subtasks: list[Any],
    answers: list[AgentAnswer],
) -> tuple[str, dict[str, str], dict[str, float | int]]:
    if not answers:
        return "这轮没有拿到可交付结果。", {}, empty_runtime_metrics()
    if len(answers) == 1:
        return answers[0].answer, {}, empty_runtime_metrics()
    synthesized, runtime_metrics = await _synthesize_compound_answer_async(agent, query, subtasks, answers)
    if synthesized:
        return synthesized, {"compound_synth": COMPOUND_SYNTH_VERSION}, runtime_metrics
    return _compose_compound_fallback(query, subtasks, answers), {}, runtime_metrics


async def _synthesize_compound_answer_async(
    agent: AudioVisualAgent,
    query: str,
    subtasks: list[Any],
    answers: list[AgentAnswer],
) -> tuple[str, dict[str, float | int]]:
    llm = select_llm(agent, "strong")
    if llm is None:
        return "", empty_runtime_metrics()
    try:
        text = await llm.agenerate(
            COMPOUND_SYNTH_USER(query, _format_compound_subtask_block(subtasks, answers)),
            system=COMPOUND_SYNTH_SYSTEM,
            temperature=0.2,
        )
    except Exception:
        from app.llm.observability import capture_llm_stats

        return "", capture_llm_stats(llm)
    from app.llm.observability import capture_llm_stats

    return text.strip(), capture_llm_stats(llm)


def _format_compound_subtask_block(subtasks: list[Any], answers: list[AgentAnswer]) -> str:
    chunks: list[str] = []
    for index, (task, answer) in enumerate(zip(subtasks, answers, strict=False), start=1):
        summary = answer.answer.strip().splitlines()[0] if answer.answer.strip() else "本步没有拿到明确结果。"
        lines = [
            f"子任务 {index}（{task.intent}）",
            f"任务：{task.query}",
            f"结果摘要：{summary[:180]}",
        ]
        if answer.answer.strip():
            lines.append(f"主要答复：{answer.answer.strip()[:600]}")
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks)


def _compose_compound_fallback(query: str, subtasks: list[Any], answers: list[AgentAnswer]) -> str:
    lines = [f"我把“{query}”拆成了 {len(subtasks)} 步来处理："]
    for index, (task, answer) in enumerate(zip(subtasks, answers, strict=False), start=1):
        lines.append(f"{index}. {task.query}")
        lines.append(answer.answer)
    return "\n".join(lines)


def _dedupe_cards(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for card in items:
        key = (
            str(card.get("title", "")).lower(),
            str(card.get("source", "")),
            str(card.get("source_id", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(card)
    return merged


def _merge_compound_recommended_tracks(answers: list[AgentAnswer]) -> list[Any]:
    merged: list[Any] = []
    seen: set[tuple[str, str, str]] = set()
    for answer in answers:
        for track in answer.recommended_tracks:
            key = (track.title.lower(), track.source, track.source_id)
            if not track.title or key in seen:
                continue
            seen.add(key)
            merged.append(track)
    return merged


def _merge_prompt_version_maps(*maps: dict[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for version_map in maps:
        for key, value in (version_map or {}).items():
            if value:
                merged[key] = value
    return merged


def _format_prompt_versions(versions: dict[str, str]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(versions.items()))
