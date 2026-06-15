from __future__ import annotations

from typing import Any


def select_llm(owner: Any, tier: str = "default") -> Any:
    """Select a tiered LLM from an agent-like object, falling back to the default."""
    default_llm = _existing_attr(owner, "llm")
    original_default = _existing_attr(owner, "_llm_default_ref") or default_llm
    if tier == "fast":
        tier_llm = _existing_attr(owner, "llm_fast")
        if default_llm is not original_default and tier_llm is original_default:
            return default_llm
        return tier_llm or default_llm
    if tier == "strong":
        tier_llm = _existing_attr(owner, "llm_strong")
        if default_llm is not original_default and tier_llm is original_default:
            return default_llm
        return tier_llm or default_llm
    return default_llm


def _existing_attr(owner: Any, name: str) -> Any:
    try:
        attrs = vars(owner)
    except TypeError:
        return getattr(owner, name, None)
    if name in attrs:
        return attrs[name]
    return getattr(type(owner), name, None)
