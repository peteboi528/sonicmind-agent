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
    "music_metadata_lookup": "knowledge_metadata_timeout_seconds",
    "review_search": "knowledge_review_timeout_seconds",
    "build_music_dossier": "knowledge_dossier_timeout_seconds",
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
            return ToolResult(
                tool=call.name,
                status=ToolStatus.UNSUPPORTED,
                error=ToolError(kind="unknown_tool", message=f"Unknown tool: {call.name}"),
            )
        started = time.monotonic()
        started_at = datetime.now(UTC).isoformat()
        retries = 0
        span_id = uuid4().hex
        try:
            arguments = spec.args_model.model_validate(call.arguments).model_dump(exclude_none=True)
        except ValidationError as exc:
            return self._finish(
                spec,
                context,
                span_id,
                started,
                started_at,
                retries,
                ToolResult(
                    tool=spec.name, status=ToolStatus.ERROR, error=ToolError(kind="validation_error", message=str(exc))
                ),
            )

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
            result = ToolResult(
                tool=spec.name,
                status=ToolStatus.UNSUPPORTED,
                error=ToolError(kind="missing_handler", message=f"No handler registered for {spec.name}"),
            )
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

        deadline = started + timeout_seconds
        if context.deadline_at is not None:
            deadline = min(deadline, context.deadline_at)
        handler_context = context.model_copy(update={"deadline_at": deadline})
        semaphore = self._semaphores.setdefault(spec.name, asyncio.Semaphore(spec.max_concurrency))
        attempts = spec.max_retries + 1 if spec.idempotent and spec.risk == ToolRisk.READ else 1
        acquired_at: float | None = None
        pending: asyncio.Task | None = None
        phase = "queue"

        def release_after_worker(task: asyncio.Task) -> None:
            semaphore.release()
            if not task.cancelled():
                task.exception()

        def run_sync() -> ToolResult:
            if time.monotonic() >= deadline:
                raise TimeoutError
            return handler(arguments, handler_context)

        try:
            if deadline <= time.monotonic():
                raise TimeoutError
            async with asyncio.timeout(deadline - time.monotonic()):
                await semaphore.acquire()
                acquired_at = time.monotonic()
                for attempt in range(attempts):
                    phase = "execution"
                    if time.monotonic() >= deadline:
                        raise TimeoutError
                    try:
                        if inspect.iscoroutinefunction(handler):
                            value = await handler(arguments, handler_context)
                        else:
                            pending = asyncio.create_task(asyncio.to_thread(run_sync))
                            value = await asyncio.shield(pending)
                        result = (
                            value
                            if isinstance(value, ToolResult)
                            else ToolResult(tool=spec.name, status=ToolStatus.OK, data=value or {})
                        )
                        break
                    except TimeoutError:
                        raise
                    except (ConnectionError, OSError) as exc:
                        delay = min(0.2 * (2**attempt), 1.0)
                        if attempt + 1 >= attempts or time.monotonic() + delay >= deadline:
                            result = ToolResult(
                                tool=spec.name,
                                status=ToolStatus.ERROR,
                                error=ToolError(kind=type(exc).__name__, message=str(exc), retryable=True),
                            )
                            break
                        phase = "retry_backoff"
                        retries += 1
                        await asyncio.sleep(delay)
        except asyncio.CancelledError:
            self._finish(
                spec, context, span_id, started, started_at, retries,
                ToolResult(
                    tool=spec.name, status=ToolStatus.CANCELLED,
                    metrics={
                        "cancel_phase": phase,
                        "queue_wait_ms": round(((acquired_at or time.monotonic()) - started) * 1000, 2),
                        "worker_still_running": bool(pending and not pending.done()),
                    },
                ),
            )
            raise
        except TimeoutError:
            metrics = {"timeout_phase": phase, "worker_still_running": bool(pending and not pending.done())}
            if spec.name in _KNOWLEDGE_TOOL_LIMITS:
                result = ToolResult(
                    tool=spec.name,
                    status=ToolStatus.EMPTY,
                    summary=f"{spec.name} 在知识链路预算内未返回，已降级继续。",
                    data={"type": spec.name, "timed_out_tools": [spec.name]},
                    metrics={**metrics, "deadline_skipped": False, "timeout_as_degraded": True},
                )
            else:
                result = ToolResult(
                    tool=spec.name,
                    status=ToolStatus.ERROR,
                    error=ToolError(
                        kind="timeout", message=f"{spec.name} timed out in {phase} after {timeout_seconds:.1f}s"
                    ),
                    metrics=metrics,
                )
        except Exception as exc:  # noqa: BLE001
            result = ToolResult(
                tool=spec.name,
                status=ToolStatus.ERROR,
                error=ToolError(kind=type(exc).__name__, message=str(exc)),
            )
        finally:
            if acquired_at is not None:
                if pending is None:
                    semaphore.release()
                else:
                    # A timed-out sync call holds its permit until its thread exits.
                    pending.add_done_callback(release_after_worker)
        result.metrics.update(
            {
                "queue_wait_ms": round(((acquired_at or time.monotonic()) - started) * 1000, 2),
                "budget_ms": round(timeout_seconds * 1000, 2),
            }
        )
        return self._finish(spec, context, span_id, started, started_at, retries, result)

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
        skipped = (remaining is not None and remaining <= 0) or (
            spec.name in _KNOWLEDGE_TOOL_LIMITS
            and spec.name not in {"build_music_dossier", "build_sample_dossier"}
            and remaining is not None
            and remaining < 1.0
        )
        return max(0.05, timeout), skipped

    def _finish(
        self,
        spec: ToolSpec,
        context: ToolContext,
        span_id: str,
        started: float,
        started_at: str,
        retries: int,
        result: ToolResult,
    ) -> ToolResult:
        duration_ms = (time.monotonic() - started) * 1000
        result.metrics.update({"duration_ms": round(duration_ms, 2), "retries": retries})
        if context.deadline_at:
            result.metrics["deadline_remaining_ms"] = round(max(0.0, context.deadline_at - time.monotonic()) * 1000, 2)
        if self.trace_store:
            self.trace_store.span(
                span_id=span_id,
                run_id=context.run_id,
                name=spec.name,
                kind="tool",
                status=result.status.value,
                started_at=started_at,
                duration_ms=duration_ms,
                retries=retries,
                attrs={
                    "risk": spec.risk.value,
                    "source": spec.source,
                    "cards": len(result.cards),
                    "queue_wait_ms": result.metrics.get("queue_wait_ms"),
                    "budget_ms": result.metrics.get("budget_ms"),
                    "timeout_phase": result.metrics.get("timeout_phase"),
                    "cancel_phase": result.metrics.get("cancel_phase"),
                    "worker_still_running": result.metrics.get("worker_still_running"),
                    "error_kind": result.error.kind if result.error else "",
                    "error_message": result.error.message if result.error else "",
                },
            )
        return result
