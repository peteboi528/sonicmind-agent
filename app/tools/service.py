from __future__ import annotations

from app.config import settings
from app.tools.checkpoints import ActionCheckpointStore
from app.tools.handlers import install_default_handlers
from app.tools.runtime import ToolRuntime
from app.tools.trace import LocalTraceStore

install_default_handlers()

trace_store = LocalTraceStore(
    settings.agent_trace_path,
    retention_days=settings.agent_retention_days,
    enabled=settings.local_tracing,
)
checkpoint_store = ActionCheckpointStore(
    settings.agent_checkpoint_path,
    retention_days=settings.agent_retention_days,
)
tool_runtime = ToolRuntime(trace_store=trace_store)

