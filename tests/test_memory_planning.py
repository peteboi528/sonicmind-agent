from __future__ import annotations

import asyncio

from app.agent import AudioVisualAgent
from app.config import settings
from app.graph import nodes
from app.models import AgentPlan, RetrievalPlan
from app.storage import JsonStore


def _plan(intent: str = "recommend", query: str = "") -> AgentPlan:
    return AgentPlan(
        intent=intent,
        tools_needed=["web_music_search", "recommend"] if intent == "recommend" else ["search"],
        retrieval_plan=RetrievalPlan(search_query=query, use_web=True),
    )


def test_broad_recommendation_gets_genre_and_mood_memory_seeds():
    plan, seeds = nodes._inject_preference_seeds(
        _plan(query="随便来点好听的"),
        "随便来点好听的",
        {"memory_query": "长期偏好 R&B 暗黑氛围 The Weeknd"},
    )

    assert seeds == ["R&B", "暗黑"]
    assert "R&B" in plan.retrieval_plan.search_query
    assert "暗黑" in plan.retrieval_plan.search_query
    assert "The Weeknd" not in plan.retrieval_plan.search_query


def test_current_mood_keeps_only_complementary_memory_genre():
    plan, seeds = nodes._inject_preference_seeds(
        _plan(query="深夜 放松"),
        "来点深夜放松的歌曲",
        {"memory_query": "R&B 暗黑 律动"},
    )

    assert seeds == ["R&B"]
    assert plan.retrieval_plan.search_query == "深夜 放松 R&B"


def test_explicit_entity_or_genre_is_not_redirected_by_memory():
    explicit_genre, genre_seeds = nodes._inject_preference_seeds(
        _plan(query="古典 专注"),
        "来点古典专注音乐",
        {"memory_query": "R&B 暗黑"},
    )
    entity_plan = _plan(query="Lana Del Rey")
    entity_plan.retrieval_plan.entities = ["Lana Del Rey"]
    explicit_entity, entity_seeds = nodes._inject_preference_seeds(
        entity_plan,
        "推荐 Lana Del Rey",
        {"memory_query": "说唱 律动"},
    )

    assert genre_seeds == []
    assert explicit_genre.retrieval_plan.search_query == "古典 专注"
    assert entity_seeds == []
    assert explicit_entity.retrieval_plan.search_query == "Lana Del Rey"


def test_negated_memory_tag_is_never_reintroduced():
    plan, seeds = nodes._inject_preference_seeds(
        _plan(query="深夜"),
        "不要R&B，来点深夜音乐",
        {"memory_query": "R&B 暗黑"},
    )

    assert "R&B" not in seeds
    assert "R&B" not in plan.retrieval_plan.search_query
    assert seeds == ["暗黑"]


def test_semantic_recall_is_injected_before_tool_stages(tmp_path, monkeypatch):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    async def fake_plan(*_args, **_kwargs):
        return _plan(query="深夜 放松"), {}, {}

    monkeypatch.setattr(nodes, "plan_with_llm_with_meta_async", fake_plan)
    monkeypatch.setattr(agent.memory, "recall_episodes", lambda *_args, **_kwargs: ["三周前偏好：R&B 暗黑"])
    state = nodes.load_context(agent, {
        "user_id": "u-memory-plan",
        "asset_id": None,
        "query": "来点深夜放松的歌曲",
        "history": [],
        "top_k": 3,
    })

    out = asyncio.run(nodes.plan_intent_async(agent, state))

    assert out["plan"].retrieval_plan.search_query == "深夜 放松 R&B"
    assert out["plan"].stages[0].calls[0].arguments["query"] == "深夜 放松 R&B"
    assert any("记忆检索种子：R&B" in line for line in out["trace"])


def test_real_planner_prompt_marks_memory_as_soft_context(tmp_path, monkeypatch):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    monkeypatch.setattr(settings, "llm_api_key", "fake-key")
    captured = {"prompt": ""}

    class FakeLLM:
        async def agenerate(self, prompt, **_kwargs):
            captured["prompt"] = prompt
            return (
                '{"intent":"recommend","entities":[],"use_local":true,'
                '"use_vector":true,"use_web":true,"search_query":"深夜 放松",'
                '"search_variants":[],"language":"","target_count":null,"reasoning":"场景推荐"}'
            )

    monkeypatch.setattr(nodes, "select_llm", lambda *_args: FakeLLM())

    plan, _, _ = asyncio.run(nodes.plan_with_llm_with_meta_async(
        agent, "来点深夜放松的歌曲", "", "R&B 暗黑 The Weeknd",
    ))

    assert plan is not None
    assert "长期音乐偏好（仅作软参考" in captured["prompt"]
    assert "R&B 暗黑 The Weeknd" in captured["prompt"]
