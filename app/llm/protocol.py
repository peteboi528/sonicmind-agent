from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMError(Exception):
    """Raised when a plain LLM completion request fails."""


@dataclass
class ToolCall:
    """LLM 发起的一次工具调用。"""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """统一的 LLM 响应封装。"""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # 可观测性字段
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    model: str = ""
    tier: str = ""
    finish_reason: str = "stop"  # stop / tool_calls / length / error
    error: str | None = None


class LLMProvider(Protocol):
    async def agenerate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        thinking: bool | None = None,
    ) -> str: ...

    def agenerate_stream(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        thinking: bool | None = None,
    ) -> AsyncIterator[str]: ...

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        thinking: bool | None = None,
    ) -> str: ...

    def generate_stream(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        thinking: bool | None = None,
    ) -> Iterator[str]:
        """单次补全的流式版本：逐块 yield content 增量，首 token 可在几百毫秒到达。"""
        ...

    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.7, thinking: bool | None = None) -> str:
        """多轮对话接口，messages 格式为 OpenAI messages 列表。"""
        ...

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.3,
        tool_choice: str = "auto",
        thinking: bool | None = None,
    ) -> LLMResponse:
        """带工具调用的多轮对话。LLM 可能返回工具调用或最终文本。

        tool_choice: 'auto' / 'required' / 'none' / 具体工具名
        """
        ...
