from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from app.graph.decompose import SubTask, decompose_compound, decompose_compound_with_meta, summarize_subtasks
from app.graph.nodes import (
    evaluate,
    execute_tools,
    finalize,
    load_context,
    plan_intent,
    reflect,
    route_after_execute,
    route_after_reflect,
    web_fallback,
)
from app.graph.state import AgentState
from app.llm.observability import empty_runtime_metrics, format_runtime_metrics, merge_runtime_metrics
from app.llm.routing import select_llm
from app.models import AgentAnswer, StreamEvent
from app.prompts import COMPOUND_SYNTH_SYSTEM, COMPOUND_SYNTH_USER, COMPOUND_SYNTH_VERSION

if TYPE_CHECKING:
    from app.agent import AudioVisualAgent


class AgentGraphRunner:
    def __init__(self, agent: AudioVisualAgent) -> None:
        self.agent = agent
        self.compiled = _try_build_langgraph(agent)

    def invoke(
        self,
        user_id: str,
        asset_id: str | None,
        query: str,
        history: list[dict[str, Any]] | None = None,
        top_k: int = 5,
    ) -> AgentAnswer:
        return self._invoke_state(user_id, asset_id, query, history=history, top_k=top_k)["answer"]

    def invoke_compound(
        self,
        user_id: str,
        asset_id: str | None,
        query: str,
        history: list[dict[str, Any]] | None = None,
        top_k: int = 5,
    ) -> AgentAnswer:
        subtasks, compound_prompt_versions, compound_runtime_metrics = decompose_compound_with_meta(self.agent, query, history)
        scratchpad: dict[str, Any] = {}
        states: list[AgentState] = []
        answers: list[AgentAnswer] = []
        compound_trace = [f"[compound_plan] {summarize_subtasks(subtasks)}"]

        for index, task in enumerate(subtasks, start=1):
            task_query = _hydrate_subtask_query(task, scratchpad)
            state = self._invoke_state(user_id, asset_id, task_query, history=history, top_k=top_k)
            answer = state["answer"]
            states.append(state)
            answers.append(answer)
            _update_scratchpad(scratchpad, task, state, index, len(subtasks))
            compound_trace.append(f"[compound_step] {index}/{len(subtasks)} {task.intent}: {task_query}")
            compound_trace.extend(f"[subtask {index}] {line}" for line in answer.agent_trace)

        final_answer, synth_versions, synth_runtime_metrics = _compose_compound_answer(self.agent, query, subtasks, answers)
        last = answers[-1] if answers else AgentAnswer(answer="", evidences=[])
        merged_prompt_versions = _merge_prompt_version_maps(
            *[ans.prompt_versions for ans in answers],
            compound_prompt_versions,
            synth_versions,
        )
        merged_runtime_metrics = merge_runtime_metrics(
            empty_runtime_metrics(),
            *[ans.runtime_metrics for ans in answers],
            compound_runtime_metrics,
            synth_runtime_metrics,
        )
        if merged_prompt_versions:
            compound_trace.append(f"[prompt] {_format_prompt_versions(merged_prompt_versions)}")
        compound_trace.append(f"[meta] {format_runtime_metrics(merged_runtime_metrics)}")
        compound_answer = AgentAnswer(
            answer=final_answer,
            evidences=[ev for ans in answers for ev in ans.evidences][:8],
            recommended_segments=last.recommended_segments,
            recommended_tracks=_merge_compound_recommended_tracks(answers),
            prompt_versions=merged_prompt_versions,
            runtime_metrics=merged_runtime_metrics,
            memory_updated=any(ans.memory_updated for ans in answers),
            agent_trace=compound_trace,
            pending_goal=last.pending_goal,
            goal_progress=last.goal_progress,
            fallback_reason=last.fallback_reason,
        )
        cards = _merge_compound_cards(states)
        if cards:
            setattr(compound_answer, "_compound_cards", cards)
        return compound_answer

    def stream(
        self,
        user_id: str,
        asset_id: str | None,
        query: str,
        history: list[dict[str, Any]] | None = None,
        top_k: int = 5,
    ) -> Iterable[StreamEvent]:
        """同步生成器节点级流式：逐节点执行，每跑完一个节点就吐出它新增的事件。

        事件顺序：plan → tool_start → candidates(先吐候选卡片) → tool_result
        → eval → final，对齐 SoulTuner 候选先于解释文本的体验。
        """
        state: AgentState = {
            "user_id": user_id,
            "asset_id": asset_id,
            "query": query,
            "history": history or [],
            "top_k": top_k,
            "_refine_count": 0,
        }
        emitted = 0
        try:
            def advance(node):
                nonlocal state, emitted
                state = node(self.agent, state)
                events = state.get("events", [])
                while emitted < len(events):
                    yield events[emitted]
                    emitted += 1

            for event in advance(load_context):
                yield event
            for event in advance(plan_intent):
                yield event
            while True:
                for event in advance(execute_tools):
                    yield event
                if route_after_execute(state) == "web_fallback":
                    for event in advance(web_fallback):
                        yield event
                for event in advance(evaluate):
                    yield event
                for event in advance(reflect):
                    yield event
                if route_after_reflect(state) != "refine":
                    break
            for event in advance(finalize):
                yield event
        except Exception as exc:  # noqa: BLE001
            yield StreamEvent(type="error", content=f"处理出错：{exc}")

    def stream_compound(
        self,
        user_id: str,
        asset_id: str | None,
        query: str,
        history: list[dict[str, Any]] | None = None,
        top_k: int = 5,
    ) -> Iterable[StreamEvent]:
        subtasks = decompose_compound(self.agent, query, history)
        yield StreamEvent(type="plan", content=summarize_subtasks(subtasks))
        answer = self.invoke_compound(user_id, asset_id, query, history=history, top_k=top_k)
        final_payload = answer.model_dump(mode="json")
        cards = getattr(answer, "_compound_cards", [])
        if cards:
            final_payload["cards"] = cards
        yield StreamEvent(type="final", content=answer.answer, payload=final_payload)

    def _invoke_state(
        self,
        user_id: str,
        asset_id: str | None,
        query: str,
        history: list[dict[str, Any]] | None = None,
        top_k: int = 5,
    ) -> AgentState:
        state: AgentState = {
            "user_id": user_id,
            "asset_id": asset_id,
            "query": query,
            "history": history or [],
            "top_k": top_k,
            "_refine_count": 0,
        }
        if self.compiled is not None:
            return self.compiled.invoke(state)
        return _fallback_invoke(self.agent, state)


def build_agent_graph(agent: AudioVisualAgent) -> AgentGraphRunner:
    return AgentGraphRunner(agent)


def _fallback_invoke(agent: AudioVisualAgent, state: AgentState) -> AgentState:
    """无 langgraph 时的等价执行：复刻条件路由（execute → [web_fallback] → evaluate → finalize）。"""
    state = load_context(agent, state)
    state = plan_intent(agent, state)
    while True:
        state = execute_tools(agent, state)
        if route_after_execute(state) == "web_fallback":
            state = web_fallback(agent, state)
        state = evaluate(agent, state)
        state = reflect(agent, state)
        if route_after_reflect(state) != "refine":
            break
    state = finalize(agent, state)
    return state


def _try_build_langgraph(agent: AudioVisualAgent):
    try:
        from langgraph.graph import END, StateGraph
    except Exception:
        return None

    graph = StateGraph(AgentState)
    graph.add_node("load_context", lambda state: load_context(agent, state))
    graph.add_node("plan_intent", lambda state: plan_intent(agent, state))
    graph.add_node("execute_tools", lambda state: execute_tools(agent, state))
    graph.add_node("web_fallback", lambda state: web_fallback(agent, state))
    graph.add_node("evaluate", lambda state: evaluate(agent, state))
    graph.add_node("reflect", lambda state: reflect(agent, state))
    graph.add_node("finalize", lambda state: finalize(agent, state))
    graph.set_entry_point("load_context")
    graph.add_edge("load_context", "plan_intent")
    graph.add_edge("plan_intent", "execute_tools")
    graph.add_conditional_edges(
        "execute_tools",
        lambda state: route_after_execute(state),
        {"web_fallback": "web_fallback", "evaluate": "evaluate"},
    )
    graph.add_edge("web_fallback", "evaluate")
    graph.add_edge("evaluate", "reflect")
    graph.add_conditional_edges(
        "reflect",
        lambda state: route_after_reflect(state),
        {"refine": "execute_tools", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph.compile()


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


def _update_scratchpad(
    scratchpad: dict[str, Any],
    task: SubTask,
    state: AgentState,
    index: int,
    total: int,
) -> None:
    answer = state["answer"]
    scratchpad["last_query"] = task.query
    scratchpad["last_answer"] = answer.answer
    scratchpad["last_goal_progress"] = answer.goal_progress
    scratchpad["last_cards"] = _extract_final_cards(state)
    summary = answer.answer.strip().splitlines()[0] if answer.answer.strip() else ""
    scratchpad["last_summary"] = summary
    scratchpad["completed_subtasks"] = index
    scratchpad["remaining_subtasks"] = max(total - index, 0)


def _compose_compound_answer(
    agent: AudioVisualAgent,
    query: str,
    subtasks: list[Any],
    answers: list[AgentAnswer],
) -> tuple[str, dict[str, str], dict[str, float | int]]:
    if not answers:
        return "这轮没有拿到可交付结果。", {}, empty_runtime_metrics()
    if len(answers) == 1:
        return answers[0].answer, {}, empty_runtime_metrics()
    synthesized, runtime_metrics = _synthesize_compound_answer(agent, query, subtasks, answers)
    if synthesized:
        return synthesized, {"compound_synth": COMPOUND_SYNTH_VERSION}, runtime_metrics
    return _compose_compound_fallback(query, subtasks, answers), {}, runtime_metrics


def _synthesize_compound_answer(
    agent: AudioVisualAgent,
    query: str,
    subtasks: list[Any],
    answers: list[AgentAnswer],
) -> tuple[str, dict[str, float | int]]:
    llm = select_llm(agent, "strong")
    if llm is None:
        return "", empty_runtime_metrics()
    try:
        text = llm.generate(
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
    for index, (task, answer) in enumerate(zip(subtasks, answers), start=1):
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
    for index, (task, answer) in enumerate(zip(subtasks, answers), start=1):
        lines.append(f"{index}. {task.query}")
        lines.append(answer.answer)
    return "\n".join(lines)


def _extract_final_cards(state: AgentState) -> list[dict[str, Any]]:
    for event in reversed(state.get("events", [])):
        if event.type != "final":
            continue
        cards = (event.payload or {}).get("cards")
        if isinstance(cards, list):
            return cards
        return []
    return []


def _merge_compound_cards(states: list[AgentState]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for state in states:
        for card in _extract_final_cards(state):
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
