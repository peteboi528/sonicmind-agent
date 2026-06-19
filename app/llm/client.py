from __future__ import annotations

import json
import socket
import time
import urllib.parse
from collections.abc import Iterator
from typing import Any

import httpx

from app.config import settings
from app.llm.mock import MockLLM
from app.llm.protocol import LLMError, LLMProvider, LLMResponse, ToolCall
from app.prompts import AGENT_SYSTEM_PROMPT

SYSTEM_PROMPT = AGENT_SYSTEM_PROMPT

# 进程级共享连接池：fast/strong/default 三档实例复用同一组 keep-alive 连接，
# 避免每次 LLM 调用都重新做 TLS 握手（每轮 3~4 次调用，能省下可观的建连延迟）。
# httpx.Client 对并发请求是线程安全的（SSE 端点会把图跑在线程池里，多请求会重叠）。
_LLM_HTTP_CLIENT: httpx.Client | None = None


def _shared_http_client() -> httpx.Client:
    global _LLM_HTTP_CLIENT
    if _LLM_HTTP_CLIENT is None:
        _LLM_HTTP_CLIENT = httpx.Client(headers={"User-Agent": "MusicAgent/1.0"})
    return _LLM_HTTP_CLIENT


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
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        tier: str = "default",
        thinking: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.tier = tier
        # DeepSeek-V4 等模型「思考模式」默认开启：每次先产出 reasoning_content（用户看不到却要等）。
        # 对意图分类/候选自查/闲聊开场这类结构化或自然语言小任务，推理几乎没有收益，纯属拖时间 + 烧 token。
        # 故默认关闭；仅对真正需要推理的调用按 thinking=True 显式开启。
        self.thinking = thinking
        self.last_stats: dict[str, float | int | str] = {}

    def _with_thinking(self, payload: dict[str, Any], override: bool | None) -> dict[str, Any]:
        """把思考模式开关注入请求体。

        DeepSeek-V4 服务端默认 thinking=enabled，必须显式传 {type: disabled} 才能关掉。
        override=None 时用实例默认；否则按调用显式覆盖。开启时可附带 reasoning_effort。
        """
        enabled = self.thinking if override is None else bool(override)
        body = dict(payload)
        body["thinking"] = {"type": "enabled" if enabled else "disabled"}
        if enabled and settings.llm_reasoning_effort:
            body["reasoning_effort"] = settings.llm_reasoning_effort
        return body

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        thinking: bool | None = None,
    ) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return self._call(messages, temperature, thinking)

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        thinking: bool | None = None,
    ) -> str:
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)
        return self._call(messages, temperature, thinking)

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.3,
        tool_choice: str = "auto",
        thinking: bool | None = None,
    ) -> LLMResponse:
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)
        payload: dict[str, Any] = self._with_thinking({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.95,
            "tools": tools,
            "tool_choice": tool_choice,
            "max_tokens": settings.llm_max_tokens,
        }, thinking)
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
            latency_ms = float((data.get("_meta") or {}).get("latency_ms", 0.0))
            estimated_cost = _estimate_cost_usd(
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
            )
            self.last_stats = {
                "llm_calls": 1,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
                "latency_ms": round(latency_ms, 2),
                "estimated_cost_usd": estimated_cost,
                "model": self.model,
                "tier": self.tier,
            }
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                latency_ms=latency_ms,
                estimated_cost_usd=estimated_cost,
                model=self.model,
                tier=self.tier,
                finish_reason=finish,
            )
        except (KeyError, IndexError, TypeError) as exc:
            return LLMResponse(content="", finish_reason="error", error=f"解析响应失败: {exc}")

    def generate_stream(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        thinking: bool | None = None,
    ) -> Iterator[str]:
        """单次补全的流式版本：逐块 yield content 增量，首 token 可在几百毫秒到达。

        给 finalize 的答案生成用：用户不再干等整段算完才看到第一个字。
        思考关闭（默认）时无 reasoning_content，content delta 即是正文；思考开启时
        只 yield content，跳过 reasoning_content（不把思考链展示给用户）。
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        yield from self._stream(messages, temperature, thinking)

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        thinking: bool | None = None,
    ) -> Iterator[str]:
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)
        yield from self._stream(messages, temperature, thinking)

    def _stream(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        thinking: bool | None,
    ) -> Iterator[str]:
        payload = self._with_thinking({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.95,
            "max_tokens": settings.llm_max_tokens,
            "stream": True,
        }, thinking)
        client = _shared_http_client()
        timeout = httpx.Timeout(settings.llm_timeout_seconds, connect=settings.llm_connect_timeout)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"
        started = time.perf_counter()
        usage: dict[str, Any] = {}
        # 流式不重试：半截流无法干净续传；失败由上层降级到模板兜底。
        try:
            with client.stream("POST", url, json=payload, headers=headers, timeout=timeout) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[len("data:"):].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {}) or {}
                    piece = delta.get("content")
                    if piece:
                        yield piece
        except httpx.HTTPError as exc:
            raise LLMError(str(exc)) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        self.last_stats = {
            "llm_calls": 1,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
            "latency_ms": round(latency_ms, 2),
            "estimated_cost_usd": _estimate_cost_usd(
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
            ),
            "model": self.model,
            "tier": self.tier,
        }

    def _call(self, messages: list[dict[str, Any]], temperature: float, thinking: bool | None = None) -> str:
        payload = self._with_thinking({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.95,
            "max_tokens": settings.llm_max_tokens,
        }, thinking)
        try:
            data = self._post(payload)
            usage = data.get("usage") or {}
            latency_ms = float((data.get("_meta") or {}).get("latency_ms", 0.0))
            self.last_stats = {
                "llm_calls": 1,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
                "latency_ms": round(latency_ms, 2),
                "estimated_cost_usd": _estimate_cost_usd(
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                ),
                "model": self.model,
                "tier": self.tier,
            }
            return _extract_message_text(data["choices"][0]["message"])
        except (httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
            raise LLMError(str(exc)) from exc

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /chat/completions，走进程级共享连接池（复用 TLS，省握手）。

        重试策略：仅对瞬时网络错误（连接重置 / 5xx / 429）按 llm_max_retries 重试并退避；
        超时与 4xx 直接抛——超时多半是慢生成，重试只会加倍等待；4xx（鉴权/格式）重试无意义。
        """
        client = _shared_http_client()
        timeout = httpx.Timeout(settings.llm_timeout_seconds, connect=settings.llm_connect_timeout)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"
        last_exc: Exception | None = None
        for attempt in range(settings.llm_max_retries + 1):
            try:
                started = time.perf_counter()
                resp = client.post(url, json=payload, headers=headers, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                data["_meta"] = {"latency_ms": (time.perf_counter() - started) * 1000}
                return data
            except httpx.HTTPStatusError as exc:
                # 5xx / 429 瞬时可重试；其余 4xx（鉴权/请求格式错误）直接抛
                if exc.response.status_code in (429, 500, 502, 503, 504) and attempt < settings.llm_max_retries:
                    last_exc = exc
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise
            except (httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                if attempt < settings.llm_max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise
            except httpx.TimeoutException:
                # 慢生成导致的超时：重试只加倍等待，直接抛让上层降级
                raise
        raise last_exc if last_exc else RuntimeError("LLM request failed")


def _estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    input_cost = (prompt_tokens / 1_000_000) * settings.llm_input_price_per_1m_tokens
    output_cost = (completion_tokens / 1_000_000) * settings.llm_output_price_per_1m_tokens
    return round(input_cost + output_cost, 8)


def build_llm(tier: str | None = None) -> LLMProvider:
    if settings.mock_mode:
        return MockLLM()
    if _local_endpoint_unavailable(settings.llm_base_url):
        return MockLLM()
    resolved_tier = tier or "default"
    return OpenAICompatibleLLM(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=_model_for_tier(resolved_tier),
        tier=resolved_tier,
        thinking=settings.llm_thinking,
    )


def _model_for_tier(tier: str) -> str:
    if tier == "fast":
        return settings.llm_fast_model
    if tier == "strong":
        return settings.llm_strong_model
    return settings.llm_model


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
