from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from app.config import settings
from app.llm.mock import MockLLM
from app.llm.protocol import LLMError, LLMProvider, LLMResponse, ToolCall
from app.prompts import AGENT_SYSTEM_PROMPT

SYSTEM_PROMPT = AGENT_SYSTEM_PROMPT


def _extract_message_text(msg: dict[str, Any]) -> str:
    """从一条 message 里取最终文本。

    推理模型（deepseek-v4-flash 等）把答案放在 reasoning_content，content 常为空。
    优先用 content；为空时回退 reasoning_content（去掉明显的思考引导前缀）。
    """
    content = (msg.get("content") or "").strip()
    if content:
        return content
    reasoning = (msg.get("reasoning_content") or "").strip()
    return reasoning


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
            "top_p": 0.95,
            "tools": tools,
            "tool_choice": tool_choice,
            "max_tokens": settings.llm_max_tokens,
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
            # 推理模型（如 deepseek-v4-flash）会把最终文本放进 reasoning_content，
            # content 为空。仅当没有工具调用、且 content 为空时，回退到 reasoning 文本，
            # 否则用户会拿到空回复 → 触发各处模板兜底（"对话僵硬"的总开关之一）。
            if not content and not tool_calls_raw:
                content = _extract_message_text(msg)
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
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.95,
            "max_tokens": settings.llm_max_tokens,
        }
        try:
            data = self._post(payload)
            return _extract_message_text(data["choices"][0]["message"])
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            raise LLMError(str(exc)) from exc

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
        # 网络抖动/超时重试一次，避免偶发失败被当成"LLM 没响应"而降级到模板。
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=settings.llm_timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                last_exc = exc
                continue
        raise last_exc if last_exc else RuntimeError("LLM request failed")


def build_llm() -> LLMProvider:
    if settings.mock_mode:
        return MockLLM()
    if _local_endpoint_unavailable(settings.llm_base_url):
        return MockLLM()
    return OpenAICompatibleLLM(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )


def _local_endpoint_unavailable(base_url: str) -> bool:
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname
    if host not in {"localhost", "127.0.0.1", "::1"}:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return False
    except OSError:
        return True
