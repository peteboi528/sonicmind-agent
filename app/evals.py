from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agent import AudioVisualAgent
from app.config import settings
from app.graph.nodes import _persist_dialogue_state, load_context, plan_intent_async, reflect_async
from app.models import AgentPlan
from app.storage import JsonStore


class GoldenAssertion(BaseModel):
    path: str
    op: Literal["eq", "contains", "excludes", "truthy"]
    value: Any = None


class GoldenCase(BaseModel):
    id: str
    mode: Literal["plan", "recovery", "multiturn_plan"] = "plan"
    query: str
    prior_query: str = ""
    memory_query: str = ""
    dialogue_state: dict[str, Any] = Field(default_factory=dict)
    plan: dict[str, Any] | None = None
    outcomes: list[dict[str, Any]] = Field(default_factory=list)
    assertions: list[GoldenAssertion]


class EvalCaseResult(BaseModel):
    id: str
    passed: bool
    passed_assertions: int
    total_assertions: int
    failures: list[str] = Field(default_factory=list)
    snapshot: dict[str, Any] = Field(default_factory=dict)


class EvalReport(BaseModel):
    passed: bool
    score: float
    passed_assertions: int
    total_assertions: int
    cases: list[EvalCaseResult]


def load_golden_cases(path: str | Path) -> list[GoldenCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [GoldenCase.model_validate(item) for item in payload]


def run_golden_evals(path: str | Path, *, deterministic: bool = True) -> EvalReport:
    cases = load_golden_cases(path)
    previous_key = settings.llm_api_key
    if deterministic:
        settings.llm_api_key = ""
    try:
        with TemporaryDirectory(prefix="sonicmind-eval-") as root:
            agent = AudioVisualAgent(JsonStore(Path(root) / "store"))
            results = asyncio.run(_run_cases(agent, cases))
    finally:
        settings.llm_api_key = previous_key
    passed = sum(item.passed_assertions for item in results)
    total = sum(item.total_assertions for item in results)
    score = passed / total if total else 1.0
    return EvalReport(
        passed=passed == total,
        score=round(score, 4),
        passed_assertions=passed,
        total_assertions=total,
        cases=results,
    )


async def _run_cases(agent: AudioVisualAgent, cases: list[GoldenCase]) -> list[EvalCaseResult]:
    return [await _run_case(agent, case) for case in cases]


async def _run_case(agent: AudioVisualAgent, case: GoldenCase) -> EvalCaseResult:
    if case.mode == "plan":
        snapshot = await _plan_snapshot(agent, case)
    elif case.mode == "multiturn_plan":
        snapshot = await _multiturn_plan_snapshot(agent, case)
    else:
        snapshot = await _recovery_snapshot(agent, case)
    failures: list[str] = []
    for assertion in case.assertions:
        actual = _get_path(snapshot, assertion.path)
        if not _matches(actual, assertion.op, assertion.value):
            failures.append(f"{assertion.path} {assertion.op} {assertion.value!r}; actual={actual!r}")
    total = len(case.assertions)
    return EvalCaseResult(
        id=case.id,
        passed=not failures,
        passed_assertions=total - len(failures),
        total_assertions=total,
        failures=failures,
        snapshot=snapshot,
    )


async def _plan_snapshot(agent: AudioVisualAgent, case: GoldenCase) -> dict[str, Any]:
    state = load_context(
        agent,
        {
            "user_id": f"eval:{case.id}",
            "asset_id": None,
            "query": case.query,
            "history": [],
            "top_k": 5,
        },
    )
    state["context"]["dialogue_state"] = case.dialogue_state
    state["context"]["memory_query"] = case.memory_query
    output = await plan_intent_async(agent, state)
    plan = output["plan"]
    effective = []
    for stage in plan.stages:
        for call in stage.calls:
            value = call.arguments.get("query") or call.arguments.get("instruction") or call.arguments.get("artist")
            if value:
                effective.append(str(value))
    return {
        "intent": plan.intent,
        "tools": plan.tools_needed,
        "entities": plan.retrieval_plan.entities,
        "search_query": plan.retrieval_plan.search_query,
        "excluded_terms": plan.retrieval_plan.excluded_terms,
        "effective_query": " | ".join(effective),
        "online_required": plan.online_required,
    }


async def _multiturn_plan_snapshot(agent: AudioVisualAgent, case: GoldenCase) -> dict[str, Any]:
    if not case.prior_query:
        raise ValueError(f"Multiturn case {case.id} requires prior_query")
    user_id = f"eval:{case.id}"
    first = load_context(
        agent,
        {
            "user_id": user_id,
            "asset_id": None,
            "query": case.prior_query,
            "history": [],
            "top_k": 5,
        },
    )
    first = await plan_intent_async(agent, first)
    _persist_dialogue_state(agent, first)
    second = load_context(
        agent,
        {
            "user_id": user_id,
            "asset_id": None,
            "query": case.query,
            "history": [],
            "top_k": 5,
        },
    )
    second = await plan_intent_async(agent, second)
    plan = second["plan"]
    return {
        "intent": plan.intent,
        "tools": plan.tools_needed,
        "entities": plan.retrieval_plan.entities,
        "search_query": plan.retrieval_plan.search_query,
        "excluded_terms": plan.retrieval_plan.excluded_terms,
        "online_required": plan.online_required,
        "persisted_genres": second["context"]["dialogue_state"].get("genre_tags", []),
        "persisted_moods": second["context"]["dialogue_state"].get("mood_tags", []),
    }


async def _recovery_snapshot(agent: AudioVisualAgent, case: GoldenCase) -> dict[str, Any]:
    if case.plan is None:
        raise ValueError(f"Recovery case {case.id} requires plan")
    state = {
        "user_id": f"eval:{case.id}",
        "query": case.query,
        "top_k": 5,
        "plan": AgentPlan.model_validate(case.plan),
        "results": [],
        "tool_outcomes": case.outcomes,
        "trace": [],
        "events": [],
        "context": {"dialogue_state": case.dialogue_state, "memory_query": case.memory_query},
        "_refine_count": 0,
    }
    output = await reflect_async(agent, state)
    plan = output["plan"]
    return {
        "need_refine": output.get("_need_refine", False),
        "tools": plan.tools_needed,
        "search_query": plan.retrieval_plan.search_query,
        "online_required": plan.online_required,
    }


def _get_path(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def _matches(actual: Any, op: str, expected: Any) -> bool:
    if op == "eq":
        return actual == expected
    if op == "truthy":
        return bool(actual)
    if op == "contains":
        return expected in actual if actual is not None else False
    if op == "excludes":
        return expected not in actual if actual is not None else True
    return False
