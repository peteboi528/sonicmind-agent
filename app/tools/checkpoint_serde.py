from __future__ import annotations

from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

_SECRET_KEYS = {
    "api_key", "apikey", "authorization", "cookie", "cookies", "credential",
    "credentials", "password", "secret", "token", "access_token", "refresh_token",
}
_BINARY_KEYS = {"audio", "raw_audio", "audio_bytes", "blob", "content_bytes", "file_bytes"}
_MAX_HISTORY_ITEMS = 20
_MAX_COLLECTION_ITEMS = 100
_MAX_STRING_LENGTH = 4000


def sanitize_checkpoint_value(value: Any, *, parent_key: str = "") -> Any:
    """Remove data that must never be persisted while preserving graph model types."""
    if isinstance(value, dict):
        is_lyrics = str(value.get("type") or "").lower() == "lyrics"
        contains_lyrics = _contains_lyrics_result(value.get("results"))
        sanitized: dict[Any, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in _SECRET_KEYS:
                sanitized[key] = "[redacted]"
            elif normalized in _BINARY_KEYS:
                sanitized[key] = "[omitted]"
            elif is_lyrics and normalized in {"lines", "lyrics", "text", "content"}:
                sanitized["line_count" if normalized == "lines" else key] = len(item) if isinstance(item, list) else 0
            elif contains_lyrics and normalized == "answer":
                sanitized[key] = _redact_answer_model(item)
            elif contains_lyrics and normalized == "events":
                sanitized[key] = [_redact_lyrics_event(event) for event in item[-_MAX_COLLECTION_ITEMS:]]
            else:
                sanitized[key] = sanitize_checkpoint_value(item, parent_key=normalized)
        return sanitized
    if isinstance(value, (list, tuple)):
        limit = _MAX_HISTORY_ITEMS if parent_key == "history" else _MAX_COLLECTION_ITEMS
        items = [sanitize_checkpoint_value(item, parent_key=parent_key) for item in value[-limit:]]
        return tuple(items) if isinstance(value, tuple) else items
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "[binary omitted]"
    if isinstance(value, str) and len(value) > _MAX_STRING_LENGTH:
        return value[:_MAX_STRING_LENGTH] + "…"
    if isinstance(value, BaseModel):
        updates = {
            name: sanitize_checkpoint_value(getattr(value, name), parent_key=name)
            for name in type(value).model_fields
        }
        return value.model_copy(update=updates)
    return value


def _contains_lyrics_result(results: Any) -> bool:
    return isinstance(results, list) and any(
        isinstance(item, dict) and str(item.get("type") or "").lower() == "lyrics"
        for item in results
    )


def _redact_answer_model(value: Any) -> Any:
    if isinstance(value, BaseModel) and hasattr(value, "answer"):
        return value.model_copy(update={"answer": "[lyrics response omitted from checkpoint]"})
    return "[lyrics response omitted from checkpoint]"


def _redact_lyrics_event(value: Any) -> Any:
    if isinstance(value, BaseModel) and getattr(value, "type", "") in {"token", "final"}:
        return value.model_copy(update={"content": "[lyrics omitted]", "payload": {}})
    if isinstance(value, dict) and value.get("type") in {"token", "final"}:
        return {**value, "content": "[lyrics omitted]", "payload": {}}
    return sanitize_checkpoint_value(value)


class SanitizingCheckpointSerializer:
    def __init__(self) -> None:
        self._inner = JsonPlusSerializer()

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return self._inner.dumps_typed(sanitize_checkpoint_value(obj))

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        return self._inner.loads_typed(data)
