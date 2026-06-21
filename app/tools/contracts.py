from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ToolStatus(StrEnum):
    OK = "ok"
    EMPTY = "empty"
    ERROR = "error"
    AUTH_REQUIRED = "auth_required"
    CONFIRMATION_REQUIRED = "confirmation_required"
    UNSUPPORTED = "unsupported"
    CANCELLED = "cancelled"


class ToolRisk(StrEnum):
    READ = "read"
    LOCAL_WRITE = "local_write"
    EXTERNAL_WRITE = "external_write"


class ToolCall(BaseModel):
    call_id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolContext(BaseModel):
    """Serializable request context plus excluded in-process service handles."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str = Field(default_factory=lambda: uuid4().hex)
    thread_id: str
    user_id: str
    query: str
    asset_id: str | None = None
    plan: dict[str, Any] | None = None
    prior_results: list[dict[str, Any]] = Field(default_factory=list)
    confirmation: dict[str, Any] | None = None
    deadline_at: float | None = None
    latency_budget: dict[str, Any] = Field(default_factory=dict)
    agent: Any = Field(default=None, exclude=True, repr=False)


class ToolError(BaseModel):
    kind: str
    message: str
    retryable: bool = False


class ToolResult(BaseModel):
    tool: str
    status: ToolStatus
    data: dict[str, Any] = Field(default_factory=dict)
    summary: str | None = None
    cards: list[dict[str, Any]] = Field(default_factory=list)
    provenance: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: ToolError | None = None
