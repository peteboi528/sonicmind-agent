from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from app.config import settings
from app.llm.mock import MockLLM
from app.llm.protocol import LLMProvider, LLMResponse, ToolCall
from app.prompts import AGENT_SYSTEM_PROMPT

SYSTEM_PROMPT = AGENT_SYSTEM_PROMPT


class OpenAICompatibleLLM:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.7) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return self._call(messages, temperature)

    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.7) -> str:
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)
        return self._call(messages, temperature)

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.3,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        try:
            data = self._post(payload)
        except Exception as exc:
            return LLMResponse(content="", finish_reason="error", error=str(exc))

        try:
            choice = data["choices"][0]
            msg = choice.get("message", {})
            finish = choice.get("finish_reason", "stop")
            content = msg.get("content") or ""
            tool_calls_raw = msg.get("tool_calls") or []
            tool_calls: list[ToolCall] = []
            for tc in tool_calls_raw:
                fn = tc.get("function", {})
                args_raw = fn.get("arguments") or "{}"
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    name=fn.get("name", ""),
                    arguments=args,
                ))
            usage = data.get("usage") or {}
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                finish_reason=finish,
            )
        except (KeyError, IndexError, TypeError) as exc:
            return LLMResponse(content="", finish_reason="error", error=f"解析响应失败: {exc}")

    def _call(self, messages: list[dict[str, Any]], temperature: float) -> str:
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        try:
            data = self._post(payload)
            return data["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            return f"LLM 请求失败: {exc}"

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))


def build_llm() -> LLMProvider:
    if settings.mock_mode:
        return MockLLM()
    return OpenAICompatibleLLM(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )
