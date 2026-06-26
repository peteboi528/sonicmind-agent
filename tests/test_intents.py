"""Intent Registry 测试：每个意图有完整元数据，关键词 fallback 与校验正确。

回归重点：discuss 不再触发 Pydantic Literal 500（AgentPlan.intent 改 str + validator）。
"""
from __future__ import annotations

import pytest

from app.intents import (
    INTENT_REGISTRY,
    get_intent,
    intent_prompt_block,
    is_valid_intent,
    match_intent_by_keywords,
    valid_intents,
)
from app.models import AgentPlan


class TestRegistryCompleteness:
    def test_every_intent_has_summary_and_prompt_desc(self):
        for name, spec in INTENT_REGISTRY.items():
            assert spec.name == name
            assert spec.summary, name
            assert spec.prompt_desc, name

    def test_non_chat_taste_intents_have_tools(self):
        # chat 无工具；其余意图至少有一个工具或联网搜索
        for name, spec in INTENT_REGISTRY.items():
            if name == "chat":
                assert spec.tools_for(True) == []
                continue
            assert spec.tools_for(spec.online_default), name

    def test_every_intent_has_strategy(self):
        for spec in INTENT_REGISTRY.values():
            assert spec.strategy_for(True) in {"online_first", "library_first", "memory_only", "no_search"}
            assert spec.strategy_for(False) in {"online_first", "library_first", "memory_only", "no_search"}

    def test_search_first_intents_prepend_web(self):
        for name in ("recommend", "search", "playlist", "discuss"):
            tools = INTENT_REGISTRY[name].tools_for(use_web=True)
            assert tools[0] == "web_music_search", name

    def test_no_web_omits_web_search(self):
        tools = INTENT_REGISTRY["recommend"].tools_for(use_web=False)
        assert "web_music_search" not in tools
        assert tools == ["recommend"]

    def test_artist_albums_uses_album_tool_without_song_search(self):
        tools = INTENT_REGISTRY["artist_albums"].tools_for(use_web=True)
        assert tools == ["artist_albums"]
        assert "web_music_search" not in tools


class TestValidation:
    def test_valid_intents_set(self):
        assert valid_intents() == set(INTENT_REGISTRY)

    def test_is_valid_intent(self):
        assert is_valid_intent("recommend")
        assert is_valid_intent("discuss")
        assert not is_valid_intent("nonsense")

    def test_get_intent_unknown_falls_back_to_chat(self):
        assert get_intent("nonsense").name == "chat"


class TestAgentPlanIntentCoercion:
    def test_discuss_does_not_raise(self):
        # 历史回归：discuss 曾因漏写 Literal 触发 500
        plan = AgentPlan(intent="discuss")
        assert plan.intent == "discuss"

    def test_unknown_intent_coerced_to_chat(self):
        plan = AgentPlan(intent="totally_new_intent")
        assert plan.intent == "chat"

    def test_all_registry_intents_accepted(self):
        for name in INTENT_REGISTRY:
            assert AgentPlan(intent=name).intent == name

    def test_none_intent_defaults_chat(self):
        plan = AgentPlan(intent=None)  # type: ignore[arg-type]
        assert plan.intent == "chat"


class TestKeywordFallback:
    @pytest.mark.parametrize("query,expected", [
        ("给我推荐几首歌", "recommend"),
        ("搜索周杰伦", "search"),
        ("做一个 chill 歌单", "playlist"),
        ("分析下我的品味", "taste"),
        ("导入这个网易云歌单", "import"),
        ("来个音乐旅程 热身到冲刺", "journey"),
        ("推荐 The Weeknd 的专辑", "artist_albums"),
        ("周杰伦有哪几张专辑", "artist_albums"),
        ("Asen 牛逼吗", "discuss"),
        ("这张专辑先听哪几首", "album_deep_dive"),
        ("这张专辑，我最想让你先听《Self Control》和《White Ferrari》。", "album_deep_dive"),
    ])
    def test_keyword_matches(self, query, expected):
        assert match_intent_by_keywords(query) == expected

    def test_no_match_returns_none(self):
        assert match_intent_by_keywords("今天天气真好") is None

    def test_import_priority_over_playlist(self):
        # "导入歌单" 同时含 import 和 playlist 信号，import 优先
        assert match_intent_by_keywords("帮我导入这个歌单") == "import"

    def test_playlist_beats_generic_journey_phase_words(self):
        assert match_intent_by_keywords("帮我做 8 首适合跑步冲刺的歌单") == "playlist"


class TestPromptBlock:
    def test_prompt_block_lists_all_intents(self):
        block = intent_prompt_block()
        for spec in INTENT_REGISTRY.values():
            assert spec.prompt_desc in block

    def test_prompt_block_has_count(self):
        assert "选一" in intent_prompt_block()
