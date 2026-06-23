from __future__ import annotations

import asyncio
import inspect
import time
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import ValidationError

from app.tools.contracts import ToolCall, ToolContext, ToolError, ToolResult, ToolRisk, ToolStatus
from app.tools.registry import ToolSpec, get_tool_spec
from app.tools.trace import LocalTraceStore

_KNOWLEDGE_TOOL_LIMITS = {
    "resolve_music_entity": "knowledge_source_timeout_seconds",
    "music_metadata_lookup": "knowledge_source_timeout_seconds",
    "review_search": "knowledge_review_timeout_seconds",
    "build_music_dossier": "knowledge_llm_timeout_seconds",
    "sample_relation_search": "knowledge_review_timeout_seconds",
    "locate_sample_sources": "knowledge_source_timeout_seconds",
    "build_sample_dossier": "knowledge_llm_timeout_seconds",
}


class ToolRuntime:
    def __init__(self, *, trace_store: LocalTraceStore | None = None) -> None:
        self.trace_store = trace_store
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        spec = get_tool_spec(call.name)
        if spec is None:
            return ToolResult(tool=call.name, status=ToolStatus.UNSUPPORTED, error=ToolError(kind="unknown_tool", message=f"Unknown tool: {call.name}"))
        started = time.monotonic()
        started_at = datetime.now(UTC).isoformat()
        retries = 0
        span_id = uuid4().hex
        try:
            arguments = spec.args_model.model_validate(call.arguments).model_dump(exclude_none=True)
        except ValidationError as exc:
            return self._finish(spec, context, span_id, started, started_at, retries, ToolResult(tool=spec.name, status=ToolStatus.ERROR, error=ToolError(kind="validation_error", message=str(exc))))

        if spec.risk == ToolRisk.EXTERNAL_WRITE and not self._confirmed(call, context):
            action_id = call.call_id
            result = ToolResult(
                tool=spec.name,
                status=ToolStatus.CONFIRMATION_REQUIRED,
                summary="这是外部账号写操作，需要明确确认后执行。",
                data={"action_id": action_id, "tool": spec.name, "arguments": arguments},
            )
            return self._finish(spec, context, span_id, started, started_at, retries, result)
        handler = spec.async_handler or spec.handler
        if handler is None:
            result = ToolResult(tool=spec.name, status=ToolStatus.UNSUPPORTED, error=ToolError(kind="missing_handler", message=f"No handler registered for {spec.name}"))
            return self._finish(spec, context, span_id, started, started_at, retries, result)

        timeout_seconds, skipped = self._effective_timeout(spec, context)
        if skipped:
            result = ToolResult(
                tool=spec.name,
                status=ToolStatus.EMPTY,
                summary=f"{spec.name} 因知识链路时间预算不足被跳过。",
                data={"type": spec.name, "skipped_due_to_deadline": [spec.name]},
                metrics={"deadline_skipped": True},
            )
            return self._finish(spec, context, span_id, started, started_at, retries, result)

        semaphore = self._semaphores.setdefault(spec.name, asyncio.Semaphore(spec.max_concurrency))
        attempts = spec.max_retries + 1 if spec.idempotent and spec.risk == ToolRisk.READ else 1
        async with semaphore:
            for attempt in range(attempts):
                try:
                    if inspect.iscoroutinefunction(handler):
                        value = await asyncio.wait_for(handler(arguments, context), timeout=timeout_seconds)
                    else:
                        value = await asyncio.wait_for(
                            asyncio.to_thread(handler, arguments, context), timeout=timeout_seconds
                        )
                    result = value if isinstance(value, ToolResult) else ToolResult(tool=spec.name, status=ToolStatus.OK, data=value or {})
                    return self._finish(spec, context, span_id, started, started_at, retries, result)
                except asyncio.CancelledError:
                    result = ToolResult(tool=spec.name, status=ToolStatus.CANCELLED, summary="工具调用已取消。")
                    return self._finish(spec, context, span_id, started, started_at, retries, result)
                except TimeoutError:
                    if spec.name in _KNOWLEDGE_TOOL_LIMITS:
                        result = ToolResult(
                            tool=spec.name,
                            status=ToolStatus.EMPTY,
                            summary=f"{spec.name} 在知识链路预算内未返回，已降级继续。",
                            data={"type": spec.name, "timed_out_tools": [spec.name]},
                            metrics={"deadline_skipped": False, "timeout_as_degraded": True},
                        )
                        return self._finish(spec, context, span_id, started, started_at, retries, result)
                    result = ToolResult(
                        tool=spec.name,
                        status=ToolStatus.ERROR,
                        error=ToolError(
                            kind="timeout",
                            message=f"{spec.name} timed out after {timeout_seconds:.1f}s",
                            retryable=False,
                        ),
                    )
                    return self._finish(spec, context, span_id, started, started_at, retries, result)
                except (ConnectionError, OSError) as exc:
                    if attempt + 1 < attempts:
                        retries += 1
                        await asyncio.sleep(min(0.2 * (2**attempt), 1.0))
                        continue
                    result = ToolResult(tool=spec.name, status=ToolStatus.ERROR, error=ToolError(kind=type(exc).__name__, message=str(exc), retryable=True))
                    return self._finish(spec, context, span_id, started, started_at, retries, result)
                except Exception as exc:  # noqa: BLE001
                    result = ToolResult(tool=spec.name, status=ToolStatus.ERROR, error=ToolError(kind=type(exc).__name__, message=str(exc)))
                    return self._finish(spec, context, span_id, started, started_at, retries, result)
        raise AssertionError("unreachable")

    async def close(self) -> None:
        return None

    @staticmethod
    def _confirmed(call: ToolCall, context: ToolContext) -> bool:
        confirmation = context.confirmation or {}
        return confirmation.get("action_id") == call.call_id and confirmation.get("approved") is True

    @staticmethod
    def _effective_timeout(spec: ToolSpec, context: ToolContext) -> tuple[float, bool]:
        timeout = spec.timeout_seconds
        if spec.name in _KNOWLEDGE_TOOL_LIMITS:
            from app.config import settings

            timeout = min(timeout, float(getattr(settings, _KNOWLEDGE_TOOL_LIMITS[spec.name])))
        remaining: float | None = None
        if context.deadline_at:
            remaining = max(0.0, context.deadline_at - time.monotonic())
            timeout = min(timeout, remaining)
        skipped = (
            spec.name in _KNOWLEDGE_TOOL_LIMITS
            and spec.name not in {"build_music_dossier", "build_sample_dossier"}
            and remaining is not None
            and remaining < 1.0
        )
        return max(0.05, timeout), skipped

    def _finish(self, spec: ToolSpec, context: ToolContext, span_id: str, started: float, started_at: str, retries: int, result: ToolResult) -> ToolResult:
        duration_ms = (time.monotonic() - started) * 1000
        result.metrics.update({"duration_ms": round(duration_ms, 2), "retries": retries})
        if context.deadline_at:
            result.metrics["deadline_remaining_ms"] = round(max(0.0, context.deadline_at - time.monotonic()) * 1000, 2)
        if self.trace_store:
            self.trace_store.span(
                span_id=span_id, run_id=context.run_id, name=spec.name, kind="tool",
                status=result.status.value, started_at=started_at, duration_ms=duration_ms,
                retries=retries,
                attrs={
                    "risk": spec.risk.value,
                    "source": spec.source,
                    "cards": len(result.cards),
                    "error_kind": result.error.kind if result.error else "",
                    "error_message": result.error.message if result.error else "",
                },
            )
        return result
