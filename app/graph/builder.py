from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from app.graph.nodes import (
    evaluate,
    execute_tools,
    finalize,
    load_context,
    plan_intent,
    reflect,
    route_after_execute,
    web_fallback,
)
from app.graph.state import AgentState
from app.models import AgentAnswer, StreamEvent

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
        state: AgentState = {
            "user_id": user_id,
            "asset_id": asset_id,
            "query": query,
            "history": history or [],
            "top_k": top_k,
        }
        if self.compiled is not None:
            out = self.compiled.invoke(state)
        else:
            out = _fallback_invoke(self.agent, state)
        return out["answer"]

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
        }
        emitted = 0
        try:
            for node in self._node_sequence():
                state = node(self.agent, state)
                events = state.get("events", [])
                while emitted < len(events):
                    yield events[emitted]
                    emitted += 1
        except Exception as exc:  # noqa: BLE001 - 流式中任何节点失败都要给前端一个 error 事件
            yield StreamEvent(type="error", content=f"处理出错：{exc}")

    @staticmethod
    def _node_sequence():
        """节点执行序列（含 web_fallback 条件路由的同步等价展开）。"""
        def execute_then_maybe_fallback(agent, state):
            state = execute_tools(agent, state)
            if route_after_execute(state) == "web_fallback":
                state = web_fallback(agent, state)
            return state

        return [load_context, plan_intent, execute_then_maybe_fallback, evaluate, reflect, finalize]


def build_agent_graph(agent: AudioVisualAgent) -> AgentGraphRunner:
    return AgentGraphRunner(agent)


def _fallback_invoke(agent: AudioVisualAgent, state: AgentState) -> AgentState:
    """无 langgraph 时的等价执行：复刻条件路由（execute → [web_fallback] → evaluate → finalize）。"""
    state = load_context(agent, state)
    state = plan_intent(agent, state)
    state = execute_tools(agent, state)
    if route_after_execute(state) == "web_fallback":
        state = web_fallback(agent, state)
    state = evaluate(agent, state)
    state = reflect(agent, state)
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
    # 条件边：候选不足 → web_fallback 补搜，否则直接评估。
    graph.add_conditional_edges(
        "execute_tools",
        lambda state: route_after_execute(state),
        {"web_fallback": "web_fallback", "evaluate": "evaluate"},
    )
    graph.add_edge("web_fallback", "evaluate")
    graph.add_edge("evaluate", "reflect")
    graph.add_edge("reflect", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()
