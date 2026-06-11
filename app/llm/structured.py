from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def parse_json_safe(text: str, model: type[T], fallback: T) -> T:
    """从 LLM 输出中提取并验证 JSON，失败返回 fallback。"""
    raw = _extract_json_object(text) or _extract_json_array(text)
    if raw is None:
        return fallback
    try:
        data = json.loads(raw)
        return model.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        return fallback


def parse_json_list_safe(text: str, item_model: type[T]) -> list[T]:
    """从 LLM 输出中提取 JSON 数组，逐项验证，跳过无效项。"""
    raw = _extract_json_array(text)
    if raw is None:
        return []
    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            return []
    except json.JSONDecodeError:
        return []
    results: list[T] = []
    for item in items:
        try:
            results.append(item_model.model_validate(item))
        except ValidationError:
            continue
    return results


def extract_json_dict(text: str) -> dict[str, Any] | None:
    """从 LLM 输出中提取第一个 JSON 对象，返回 dict 或 None。"""
    raw = _extract_json_object(text)
    if raw is None:
        return None
    try:
        result = json.loads(raw)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        return None


def extract_json_list(text: str) -> list[Any] | None:
    """从 LLM 输出中提取第一个 JSON 数组，返回 list 或 None。"""
    raw = _extract_json_array(text)
    if raw is None:
        return None
    try:
        result = json.loads(raw)
        return result if isinstance(result, list) else None
    except json.JSONDecodeError:
        return None


def _extract_json_object(text: str) -> str | None:
    return _extract_balanced(text, "{", "}")


def _extract_json_array(text: str) -> str | None:
    return _extract_balanced(text, "[", "]")


def _strip_code_fence(text: str) -> str:
    """剥掉 ```json ... ``` 围栏，返回围栏内内容；无围栏则原样返回。

    只取第一个围栏块的内容，避免贪婪 .* 跨多个块误吞。
    """
    fence = re.search(r"```(?:json|JSON)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    return text


def _extract_balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    """字符串感知的括号深度扫描器：返回第一个配平的 JSON 片段。

    旧实现用 find(open) + rfind(close)，遇到「多个 JSON 片段」或「字符串内
    含括号/转义」会切出非法或过长的串。这里逐字符扫描，跟踪字符串状态与转义，
    只在括号深度归零时收口，确保切出的是单个语法配平的片段。

    优先在 fenced code block 内查找；找不到再扫全文。
    """
    fenced = _strip_code_fence(text)
    found = _scan_balanced(fenced, open_ch, close_ch)
    if found is not None:
        return found
    if fenced is not text:
        return _scan_balanced(text, open_ch, close_ch)
    return None


def _scan_balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            # 仅在字符串内部转义有意义；串外的反斜杠忽略即可。
            escaped = in_string
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
