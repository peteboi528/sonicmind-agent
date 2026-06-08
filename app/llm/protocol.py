from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


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
    finish_reason: str = "stop"  # stop / tool_calls / length / error
    error: str | None = None


class LLMProvider(Protocol):
    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.7) -> str: ...

    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.7) -> str:
        """多轮对话接口，messages 格式为 OpenAI messages 列表。"""
        ...

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.3,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        """带工具调用的多轮对话。LLM 可能返回工具调用或最终文本。

        tool_choice: 'auto' / 'required' / 'none' / 具体工具名
        """
        ...
