from __future__ import annotations

from app.models import AgentPlan


def next_tool(plan: AgentPlan, completed: set[str]) -> str:
    for tool in plan.tools_needed:
        if tool not in completed:
            return tool
    return "finalize"


def should_continue(completed: set[str], plan: AgentPlan) -> bool:
    return any(tool not in completed for tool in plan.tools_needed)
