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
    # 优先匹配 ```json ... ``` 代码块
    block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if block:
        return block.group(1)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return text[start:end]
    return None


def _extract_json_array(text: str) -> str | None:
    block = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if block:
        return block.group(1)
    start = text.find("[")
    end = text.rfind("]") + 1
    if start != -1 and end > start:
        return text[start:end]
    return None
