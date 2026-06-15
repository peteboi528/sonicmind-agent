from __future__ import annotations

from typing import Any


def empty_runtime_metrics() -> dict[str, float | int]:
    return {
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_ms": 0.0,
        "estimated_cost_usd": 0.0,
        "default_llm_calls": 0,
        "fast_llm_calls": 0,
        "strong_llm_calls": 0,
    }


def merge_runtime_metrics(
    base: dict[str, float | int] | None,
    *updates: dict[str, Any] | None,
) -> dict[str, float | int]:
    merged = dict(empty_runtime_metrics())
    if base:
        for key, value in base.items():
            merged[key] = value
    for update in updates:
        if not update:
            continue
        merged["llm_calls"] = int(merged.get("llm_calls", 0)) + int(update.get("llm_calls", 0) or 0)
        merged["prompt_tokens"] = int(merged.get("prompt_tokens", 0)) + int(update.get("prompt_tokens", 0) or 0)
        merged["completion_tokens"] = int(merged.get("completion_tokens", 0)) + int(update.get("completion_tokens", 0) or 0)
        merged["total_tokens"] = int(merged.get("total_tokens", 0)) + int(update.get("total_tokens", 0) or 0)
        merged["latency_ms"] = round(float(merged.get("latency_ms", 0.0)) + float(update.get("latency_ms", 0.0) or 0.0), 2)
        merged["estimated_cost_usd"] = round(
            float(merged.get("estimated_cost_usd", 0.0)) + float(update.get("estimated_cost_usd", 0.0) or 0.0),
            8,
        )
        merged["default_llm_calls"] = int(merged.get("default_llm_calls", 0)) + int(update.get("default_llm_calls", 0) or 0)
        merged["fast_llm_calls"] = int(merged.get("fast_llm_calls", 0)) + int(update.get("fast_llm_calls", 0) or 0)
        merged["strong_llm_calls"] = int(merged.get("strong_llm_calls", 0)) + int(update.get("strong_llm_calls", 0) or 0)
    return merged


def capture_llm_stats(llm: Any) -> dict[str, float | int]:
    stats = getattr(llm, "last_stats", None) or {}
    metrics = {
        "llm_calls": int(stats.get("llm_calls", 0) or 0),
        "prompt_tokens": int(stats.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(stats.get("completion_tokens", 0) or 0),
        "total_tokens": int(stats.get("total_tokens", 0) or 0),
        "latency_ms": round(float(stats.get("latency_ms", 0.0) or 0.0), 2),
        "estimated_cost_usd": round(float(stats.get("estimated_cost_usd", 0.0) or 0.0), 8),
    }
    return merge_runtime_metrics(metrics, tier_call_metrics(str(stats.get("tier") or ""), metrics["llm_calls"]))


def tier_call_metrics(tier: str | None, calls: int = 1) -> dict[str, int]:
    if calls <= 0:
        return {}
    normalized = (tier or "default").strip().lower()
    if normalized == "fast":
        return {"fast_llm_calls": calls}
    if normalized == "strong":
        return {"strong_llm_calls": calls}
    return {"default_llm_calls": calls}


def format_runtime_metrics(metrics: dict[str, float | int] | None) -> str:
    data = metrics or empty_runtime_metrics()
    tier_bits = []
    for label, key in (
        ("default", "default_llm_calls"),
        ("fast", "fast_llm_calls"),
        ("strong", "strong_llm_calls"),
    ):
        calls = int(data.get(key, 0) or 0)
        if calls:
            tier_bits.append(f"{label}:{calls}")
    tier_text = f", tiers={','.join(tier_bits)}" if tier_bits else ""
    return (
        f"llm_calls={int(data.get('llm_calls', 0))}, "
        f"tokens={int(data.get('total_tokens', 0))}"
        f"({int(data.get('prompt_tokens', 0))}+{int(data.get('completion_tokens', 0))}), "
        f"latency_ms={round(float(data.get('latency_ms', 0.0)), 2)}, "
        f"cost_usd={round(float(data.get('estimated_cost_usd', 0.0)), 8)}"
        f"{tier_text}"
    )
