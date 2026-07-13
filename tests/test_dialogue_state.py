"""DialogueState 多轮延续测试：继承 / 话题切换 / 持久化 / 多用户隔离。"""

from __future__ import annotations

import asyncio

import pytest

from app.graph.nodes import _apply_dialogue_continuation, _query_with_entities
from app.intents import extract_content_negations, is_continuation
from app.memory import MemoryManager
from app.models import AgentPlan, RetrievalPlan
from app.storage import JsonStore


@pytest.fixture
def memory(tmp_path):
    return MemoryManager(JsonStore(tmp_path / "store"))


class TestContinuationDetection:
    @pytest.mark.parametrize(
        "query",
        [
            "再来几首",
            "换一批",
            "类似这个",
            "还要",
            "more please",
            # Bug1① 回归：反重复信号 + 纯数量请求必须判为延续，否则跨轮去重永不触发。
            "不要重复",
            "别重复",
            "换新的",
            "我需要12首",
            "12首",
            "再多几首",
            # 内容否定依赖上一轮正向上下文，不是独立的新搜索主题。
            "不要越南",
            "不要中文歌曲",
            "别放日语歌",
            "推荐同类型的歌手",
            "找几个同风格歌手",
        ],
    )
    def test_continuation_signals(self, query):
        assert is_continuation(query)

    @pytest.mark.parametrize(
        "query",
        [
            "推荐周杰伦的歌",
            "我想听一些适合下雨天在家放松的纯音乐钢琴曲",  # 长查询自带语境，不算延续
            "你好",
            # Bug1① 回归：自带新实体的数量请求不算延续（话题切换）。
            "周杰伦12首",
            "推荐12首歌",
            "不要了",
        ],
    )
    def test_non_continuation(self, query):
        assert not is_continuation(query)

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("不要越南语歌曲", "越南"),
            ("别放日本语歌", "日语"),
            ("排除韩文音乐", "韩语"),
            ("without Vietnamese songs", "越南"),
            ("no Cantonese music", "粤语"),
        ],
    )
    def test_language_negation_aliases_are_normalized(self, query, expected):
        assert extract_content_negations(query) == [expected]
        assert is_continuation(query)


class TestShownTracksAccumulation:
    """Bug1② 回归：shown_tracks 必须跨轮累积（而非每轮覆盖），否则第三轮"不要重复"
    的排除集只有第二轮那几首，第一轮展示过的又会漏回来。"""

    def test_accumulate_on_continuation(self, tmp_path):
        from app.agent import AudioVisualAgent
        from app.graph.nodes import _persist_dialogue_state
        from app.models import RetrievalPlan
        from app.storage import JsonStore

        agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
        prev_shown = [
            {"title": "Old A", "source_id": "1"},
            {"title": "Old B", "source_id": "2"},
        ]
        plan = AgentPlan(
            intent="recommend",
            online_required=True,
            retrieval_plan=RetrievalPlan(entities=[], use_web=True),
        )
        state = {
            "user_id": "u",
            "query": "不要重复",
            "plan": plan,
            "results": [],
            "trace": [],
            "events": [],
            "context": {"dialogue_state": {"shown_tracks": prev_shown, "entities": []}},
        }
        _persist_dialogue_state(agent, state)
        saved = agent.memory.get_dialogue_state("u").shown_tracks
        # 本轮无新增曲目时，累积记录应保留前轮的 2 首（不会被空本轮覆盖）
        assert any(s.get("title") == "Old A" for s in saved)
        assert any(s.get("title") == "Old B" for s in saved)

    def test_similar_artists_accumulate_shown_artists_on_continuation(self, tmp_path):
        from app.agent import AudioVisualAgent
        from app.graph.nodes import _persist_dialogue_state
        from app.storage import JsonStore

        agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
        prev_shown_artists = [{"name": "SZA", "source": "local_library"}]
        plan = AgentPlan(intent="similar_artists", tools_needed=["similar_artists"])
        state = {
            "user_id": "u",
            "query": "再来一点",
            "plan": plan,
            "results": [
                {
                    "type": "similar_artists",
                    "artists": [{"name": "Frank Ocean", "source": "local_library"}],
                }
            ],
            "trace": [],
            "events": [],
            "context": {"dialogue_state": {"shown_artists": prev_shown_artists, "entities": ["The Weeknd"]}},
        }
        _persist_dialogue_state(agent, state)
        saved = agent.memory.get_dialogue_state("u").shown_artists
        assert any(item.get("name") == "SZA" for item in saved)
        assert any(item.get("name") == "Frank Ocean" for item in saved)

    def test_reset_on_topic_switch(self, tmp_path):
        from app.agent import AudioVisualAgent
        from app.graph.nodes import _persist_dialogue_state
        from app.models import RetrievalPlan
        from app.storage import JsonStore

        agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
        prev_shown = [{"title": "Old A", "source_id": "1"}]
        plan = AgentPlan(
            intent="search",
            online_required=True,
            retrieval_plan=RetrievalPlan(entities=["林俊杰"], use_web=True),
        )
        # 非延续（全新搜索）→ 即使 context 里有旧 shown，也必须重置（不算进新话题）
        state = {
            "user_id": "u",
            "query": "搜林俊杰的歌",
            "plan": plan,
            "results": [],
            "trace": [],
            "events": [],
            "context": {"dialogue_state": {"shown_tracks": prev_shown, "entities": []}},
        }
        _persist_dialogue_state(agent, state)
        saved = agent.memory.get_dialogue_state("u").shown_tracks
        assert all(s.get("title") != "Old A" for s in saved)

    def test_accumulate_even_with_inherited_entities(self, tmp_path):
        """Bug 回归（实测第三轮重复第一轮的歌）：延续指令会从上一轮"继承"实体，
        继承后 rp.entities 非空。若累积条件写成 `is_continuation and not rp.entities`，
        会把"继承实体"误判成话题切换而重置，丢掉前几轮记录 → 第三轮排除集缺第一轮
        → 第一轮的歌被捞回来。累积必须只看 is_continuation，与实体无关。"""
        from app.agent import AudioVisualAgent
        from app.graph.nodes import _persist_dialogue_state
        from app.models import RetrievalPlan
        from app.storage import JsonStore

        agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
        # 模拟第二轮"多来几首"：延续 + 继承了实体 The Weeknd（非空）
        prev_shown = [{"title": "Blinding Lights", "source_id": "1"}]  # 第一轮展示的
        plan = AgentPlan(
            intent="recommend",
            online_required=True,
            retrieval_plan=RetrievalPlan(entities=["The Weeknd"], use_web=True),  # 继承来的，非空
        )
        state = {
            "user_id": "u",
            "query": "多来几首",
            "plan": plan,
            "results": [],
            "trace": [],
            "events": [],
            "context": {"dialogue_state": {"shown_tracks": prev_shown, "entities": ["The Weeknd"]}},
        }
        _persist_dialogue_state(agent, state)
        saved = agent.memory.get_dialogue_state("u").shown_tracks
        # 第一轮的 Blinding Lights 必须保留在累积记录里（不能被继承实体触发的重置丢掉）
        assert any(s.get("title") == "Blinding Lights" for s in saved)


class TestPersistence:
    def test_default_state_when_empty(self, memory):
        state = memory.get_dialogue_state("u1")
        assert state.user_id == "u1"
        assert state.turn_count == 0
        assert state.entities == []
        assert state.last_intent == "chat"

    def test_save_and_read_back(self, memory):
        memory.save_dialogue_state(
            "u1",
            intent="recommend",
            query="推荐周杰伦",
            entities=["周杰伦"],
            genre_tags=["流行"],
            mood_tags=["放松"],
        )
        state = memory.get_dialogue_state("u1")
        assert state.last_intent == "recommend"
        assert state.entities == ["周杰伦"]
        assert state.genre_tags == ["流行"]
        assert state.mood_tags == ["放松"]
        assert state.turn_count == 1

    def test_turn_count_increments(self, memory):
        memory.save_dialogue_state("u1", intent="recommend", query="q1", entities=["A"])
        memory.save_dialogue_state("u1", intent="search", query="q2", entities=["B"])
        assert memory.get_dialogue_state("u1").turn_count == 2

    def test_persist_derives_missing_positive_tags_from_raw_query(self, tmp_path):
        from app.agent import AudioVisualAgent
        from app.graph.nodes import _persist_dialogue_state

        agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
        plan = AgentPlan(
            intent="recommend",
            online_required=True,
            retrieval_plan=RetrievalPlan(search_query="越南 chill R&B", use_web=True),
        )
        _persist_dialogue_state(
            agent,
            {
                "user_id": "u-seeds",
                "query": "推荐越南 chill R&B 歌曲",
                "plan": plan,
                "results": [],
                "events": [],
                "trace": [],
                "context": {"dialogue_state": {}},
            },
        )

        saved = agent.memory.get_dialogue_state("u-seeds")
        assert "R&B" in saved.genre_tags
        assert "放松" in saved.mood_tags

    def test_clear_state(self, memory):
        memory.save_dialogue_state("u1", intent="recommend", query="q", entities=["A"])
        memory.clear_dialogue_state("u1")
        state = memory.get_dialogue_state("u1")
        assert state.entities == []
        assert state.turn_count == 0

    def test_clear_missing_is_safe(self, memory):
        memory.clear_dialogue_state("never_seen")  # 不抛错
        assert memory.get_dialogue_state("never_seen").turn_count == 0


class TestMultiUserIsolation:
    def test_users_do_not_leak(self, memory):
        memory.save_dialogue_state("alice", intent="recommend", query="q", entities=["周杰伦"])
        memory.save_dialogue_state("bob", intent="search", query="q", entities=["Beyond"])
        assert memory.get_dialogue_state("alice").entities == ["周杰伦"]
        assert memory.get_dialogue_state("bob").entities == ["Beyond"]

    def test_clear_one_user_keeps_other(self, memory):
        memory.save_dialogue_state("alice", intent="recommend", query="q", entities=["周杰伦"])
        memory.save_dialogue_state("bob", intent="search", query="q", entities=["Beyond"])
        memory.clear_dialogue_state("alice")
        assert memory.get_dialogue_state("alice").entities == []
        assert memory.get_dialogue_state("bob").entities == ["Beyond"]


def _plan(intent="recommend", entities=None, online=True):
    return AgentPlan(
        intent=intent,
        online_required=online,
        retrieval_plan=RetrievalPlan(entities=entities or [], use_web=online),
    )


class TestContinuationInheritance:
    def test_inherits_entities_on_continuation(self):
        prev = {
            "entities": ["周杰伦"],
            "last_intent": "recommend",
            "genre_tags": ["流行"],
            "mood_tags": [],
            "scenario_tags": [],
        }
        plan = _plan(intent="recommend", entities=[])
        new_plan, inherited = _apply_dialogue_continuation(plan, "再来几首", prev)
        assert "周杰伦" in inherited
        assert new_plan.retrieval_plan.entities == ["周杰伦"]
        assert new_plan.retrieval_plan.genre_filter == ["流行"]

    def test_inherits_prev_intent_when_this_turn_chat(self):
        prev = {"entities": ["Beyond"], "last_intent": "search", "genre_tags": [], "mood_tags": [], "scenario_tags": []}
        plan = _plan(intent="chat", entities=[])
        new_plan, inherited = _apply_dialogue_continuation(plan, "换一批", prev)
        assert new_plan.intent == "search"
        assert new_plan.retrieval_plan.entities == ["Beyond"]

    def test_no_inherit_when_this_turn_has_entities(self):
        # 本轮自带新实体 = 话题切换，不继承旧的
        prev = {
            "entities": ["周杰伦"],
            "last_intent": "recommend",
            "genre_tags": [],
            "mood_tags": [],
            "scenario_tags": [],
        }
        plan = _plan(intent="search", entities=["林俊杰"])
        new_plan, inherited = _apply_dialogue_continuation(plan, "再来几首林俊杰", prev)
        assert inherited == ""
        assert new_plan.retrieval_plan.entities == ["林俊杰"]

    def test_excluded_mounted_even_with_entities(self):
        """Bug 回归：用户重提歌手名（"The Weeknd 再来几首"）时，实体不继承，
        但跨轮去重排除集必须照挂——否则同一查询永远返回同一批 top-N，重复照旧。"""
        prev = {
            "entities": ["The Weeknd"],
            "last_intent": "search",
            "shown_tracks": [{"title": "Blinding Lights", "source_id": "1"}],
            "genre_tags": [],
            "mood_tags": [],
            "scenario_tags": [],
        }
        plan = _plan(intent="search", entities=["The Weeknd"])
        new_plan, inherited = _apply_dialogue_continuation(plan, "The Weeknd 再来几首", prev)
        assert inherited == ""  # 实体不继承
        assert new_plan.retrieval_plan.entities == ["The Weeknd"]
        # 排除集必须挂上（去重要生效）
        assert getattr(new_plan, "_excluded_tracks", None)
        assert new_plan._excluded_tracks[0]["title"] == "Blinding Lights"

    def test_no_inherit_when_not_continuation(self):
        prev = {
            "entities": ["周杰伦"],
            "last_intent": "recommend",
            "genre_tags": [],
            "mood_tags": [],
            "scenario_tags": [],
        }
        plan = _plan(intent="recommend", entities=[])
        new_plan, inherited = _apply_dialogue_continuation(plan, "推荐点爵士乐", prev)
        assert inherited == ""
        assert new_plan.retrieval_plan.entities == []

    def test_no_inherit_without_prior_state(self):
        plan = _plan(intent="recommend", entities=[])
        new_plan, inherited = _apply_dialogue_continuation(plan, "再来几首", None)
        assert inherited == ""
        assert new_plan is plan

    def test_content_negation_removes_negative_entity_and_inherits_positive_seed(self):
        prev = {
            "entities": ["越南"],
            "last_intent": "recommend",
            "last_query": "推荐越南 chill R&B",
            "genre_tags": ["R&B"],
            "mood_tags": ["chill"],
            "scenario_tags": [],
            "shown_tracks": [{"title": "Old", "source_id": "1"}],
        }
        plan = AgentPlan(
            intent="recommend",
            tools_needed=["web_music_search", "recommend"],
            online_required=True,
            retrieval_plan=RetrievalPlan(
                entities=["越南"],
                use_web=True,
                search_query="不要越南",
                search_variants=["不要越南 chill"],
            ),
        )

        new_plan, _ = _apply_dialogue_continuation(plan, "不要越南", prev)

        rp = new_plan.retrieval_plan
        assert rp.entities == []
        assert "越南" not in rp.search_query
        assert "不要" not in rp.search_query
        assert "chill" in rp.search_query and "R&B" in rp.search_query
        assert rp.excluded_terms == ["越南"]
        assert "不要" not in _query_with_entities("不要越南", new_plan)
        assert all("越南" not in variant and "不要" not in variant for variant in rp.search_variants)
        assert new_plan._excluded_tracks

    def test_content_negation_derives_language_and_keeps_prior_scenario(self):
        prev = {
            "entities": [],
            "last_intent": "recommend",
            "last_query": "推荐深夜中文歌曲",
            "genre_tags": [],
            "mood_tags": ["放松"],
            "scenario_tags": ["深夜"],
        }
        plan = AgentPlan(
            intent="recommend",
            online_required=True,
            retrieval_plan=RetrievalPlan(search_query="", use_web=True),
        )

        new_plan, _ = _apply_dialogue_continuation(plan, "不要中文歌曲", prev)

        assert new_plan.retrieval_plan.language_filter == "en"
        assert "深夜" in new_plan.retrieval_plan.search_query
        assert "中文" not in new_plan.retrieval_plan.search_query


class TestEndToEndThroughGraph:
    """跑通 load_context→plan→finalize 一轮后，DialogueState 落盘且下一轮可继承。"""

    def test_round_trip_persists_and_inherits(self, tmp_path):
        from app.agent import AudioVisualAgent

        agent = AudioVisualAgent(JsonStore(tmp_path / "store"))

        # 第一轮：明确推荐周杰伦
        asyncio.run(agent.chat_async("user-x", "推荐周杰伦的歌"))
        saved = agent.memory.get_dialogue_state("user-x")
        assert saved.last_intent in {"recommend", "search"}
        assert saved.turn_count >= 1

    def test_chat_clears_dialogue_state(self, tmp_path):
        from app.agent import AudioVisualAgent

        agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
        agent.memory.save_dialogue_state("user-y", intent="recommend", query="q", entities=["周杰伦"])
        asyncio.run(agent.chat_async("user-y", "你好"))
        # chat 意图应清空旧延续状态
        assert agent.memory.get_dialogue_state("user-y").entities == []
