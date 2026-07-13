from __future__ import annotations

from types import SimpleNamespace

from app.agent import AudioVisualAgent
from app.config import settings
from app.llm.client import OpenAICompatibleLLM, build_llm
from app.llm.mock import MockLLM
from app.llm.observability import capture_llm_stats, format_runtime_metrics, merge_runtime_metrics, tier_call_metrics
from app.llm.routing import select_llm
from app.storage import JsonStore


def test_build_llm_selects_configured_model_tiers(monkeypatch):
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_base_url", "https://llm.example.test/v1")
    monkeypatch.setattr(settings, "llm_model", "balanced-model")
    monkeypatch.setattr(settings, "llm_fast_model", "fast-model")
    monkeypatch.setattr(settings, "llm_strong_model", "strong-model")

    fast = build_llm("fast")
    strong = build_llm("strong")
    default = build_llm()

    assert isinstance(fast, OpenAICompatibleLLM)
    assert isinstance(strong, OpenAICompatibleLLM)
    assert isinstance(default, OpenAICompatibleLLM)
    assert fast.model == "fast-model"
    assert fast.tier == "fast"
    assert strong.model == "strong-model"
    assert strong.tier == "strong"
    assert default.model == "balanced-model"
    assert default.tier == "default"


def test_build_llm_uses_mock_when_api_key_missing_and_endpoint_is_remote(monkeypatch):
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "llm_base_url", "https://llm.example.test/v1")

    llm = build_llm()

    assert isinstance(llm, MockLLM)


def test_agent_reuses_default_llm_when_tier_models_match(monkeypatch, tmp_path):
    sentinel = object()
    calls: list[str | None] = []

    def fake_build_llm(tier: str | None = None):
        calls.append(tier)
        return sentinel

    monkeypatch.setattr(settings, "llm_model", "same-model")
    monkeypatch.setattr(settings, "llm_fast_model", "same-model")
    monkeypatch.setattr(settings, "llm_strong_model", "same-model")
    monkeypatch.setattr("app.agent.build_llm", fake_build_llm)

    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))

    assert calls == [None]
    assert agent.llm is sentinel
    assert agent.llm_fast is sentinel
    assert agent.llm_strong is sentinel


def test_agent_builds_distinct_tier_llms_when_models_differ(monkeypatch, tmp_path):
    calls: list[str | None] = []

    def fake_build_llm(tier: str | None = None):
        calls.append(tier)
        return f"llm:{tier or 'default'}"

    monkeypatch.setattr(settings, "llm_model", "balanced-model")
    monkeypatch.setattr(settings, "llm_fast_model", "fast-model")
    monkeypatch.setattr(settings, "llm_strong_model", "strong-model")
    monkeypatch.setattr("app.agent.build_llm", fake_build_llm)

    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))

    assert calls == [None, "fast", "strong"]
    assert select_llm(agent, "fast") == "llm:fast"
    assert select_llm(agent, "strong") == "llm:strong"
    assert select_llm(agent) == "llm:default"


def test_runtime_metrics_track_model_tiers():
    llm = SimpleNamespace(
        last_stats={
            "llm_calls": 1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "latency_ms": 12.3,
            "estimated_cost_usd": 0.00001,
            "tier": "strong",
        }
    )

    metrics = capture_llm_stats(llm)

    assert metrics["llm_calls"] == 1
    assert metrics["strong_llm_calls"] == 1
    assert "tiers=strong:1" in format_runtime_metrics(metrics)


def test_runtime_metrics_merge_tier_counts():
    metrics = merge_runtime_metrics(
        {},
        {"llm_calls": 1, **tier_call_metrics("fast")},
        {"llm_calls": 1, **tier_call_metrics("strong")},
    )

    assert metrics["llm_calls"] == 2
    assert metrics["fast_llm_calls"] == 1
    assert metrics["strong_llm_calls"] == 1
    assert "tiers=fast:1,strong:1" in format_runtime_metrics(metrics)
