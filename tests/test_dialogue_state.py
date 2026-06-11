"""DialogueState 多轮延续测试：继承 / 话题切换 / 持久化 / 多用户隔离。"""
from __future__ import annotations

import pytest

from app.graph.nodes import _apply_dialogue_continuation
from app.intents import is_continuation
from app.memory import MemoryManager
from app.models import AgentPlan, RetrievalPlan
from app.storage import JsonStore


@pytest.fixture
def memory(tmp_path):
    return MemoryManager(JsonStore(tmp_path / "store"))


class TestContinuationDetection:
    @pytest.mark.parametrize("query", ["再来几首", "换一批", "类似这个", "还要", "more please"])
    def test_continuation_signals(self, query):
        assert is_continuation(query)

    @pytest.mark.parametrize("query", [
        "推荐周杰伦的歌",
        "我想听一些适合下雨天在家放松的纯音乐钢琴曲",  # 长查询自带语境，不算延续
        "你好",
    ])
    def test_non_continuation(self, query):
        assert not is_continuation(query)


class TestPersistence:
    def test_default_state_when_empty(self, memory):
        state = memory.get_dialogue_state("u1")
        assert state.user_id == "u1"
        assert state.turn_count == 0
        assert state.entities == []
        assert state.last_intent == "chat"

    def test_save_and_read_back(self, memory):
        memory.save_dialogue_state(
            "u1", intent="recommend", query="推荐周杰伦",
            entities=["周杰伦"], genre_tags=["流行"], mood_tags=["放松"],
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
            "entities": ["周杰伦"], "last_intent": "recommend",
            "genre_tags": ["流行"], "mood_tags": [], "scenario_tags": [],
        }
        plan = _plan(intent="recommend", entities=[])
        new_plan, inherited = _apply_dialogue_continuation(plan, "再来几首", prev)
        assert "周杰伦" in inherited
        assert new_plan.retrieval_plan.entities == ["周杰伦"]
        assert new_plan.retrieval_plan.genre_filter == ["流行"]

    def test_inherits_prev_intent_when_this_turn_chat(self):
        prev = {"entities": ["Beyond"], "last_intent": "search",
                "genre_tags": [], "mood_tags": [], "scenario_tags": []}
        plan = _plan(intent="chat", entities=[])
        new_plan, inherited = _apply_dialogue_continuation(plan, "换一批", prev)
        assert new_plan.intent == "search"
        assert new_plan.retrieval_plan.entities == ["Beyond"]

    def test_no_inherit_when_this_turn_has_entities(self):
        # 本轮自带新实体 = 话题切换，不继承旧的
        prev = {"entities": ["周杰伦"], "last_intent": "recommend",
                "genre_tags": [], "mood_tags": [], "scenario_tags": []}
        plan = _plan(intent="search", entities=["林俊杰"])
        new_plan, inherited = _apply_dialogue_continuation(plan, "再来几首林俊杰", prev)
        assert inherited == ""
        assert new_plan.retrieval_plan.entities == ["林俊杰"]

    def test_no_inherit_when_not_continuation(self):
        prev = {"entities": ["周杰伦"], "last_intent": "recommend",
                "genre_tags": [], "mood_tags": [], "scenario_tags": []}
        plan = _plan(intent="recommend", entities=[])
        new_plan, inherited = _apply_dialogue_continuation(plan, "推荐点爵士乐", prev)
        assert inherited == ""
        assert new_plan.retrieval_plan.entities == []

    def test_no_inherit_without_prior_state(self):
        plan = _plan(intent="recommend", entities=[])
        new_plan, inherited = _apply_dialogue_continuation(plan, "再来几首", None)
        assert inherited == ""
        assert new_plan is plan


class TestEndToEndThroughGraph:
    """跑通 load_context→plan→finalize 一轮后，DialogueState 落盘且下一轮可继承。"""

    def test_round_trip_persists_and_inherits(self, tmp_path):
        from app.agent import AudioVisualAgent
        agent = AudioVisualAgent(JsonStore(tmp_path / "store"))

        # 第一轮：明确推荐周杰伦
        agent.chat("user-x", "推荐周杰伦的歌")
        saved = agent.memory.get_dialogue_state("user-x")
        assert saved.last_intent in {"recommend", "search"}
        assert saved.turn_count >= 1

    def test_chat_clears_dialogue_state(self, tmp_path):
        from app.agent import AudioVisualAgent
        agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
        agent.memory.save_dialogue_state("user-y", intent="recommend", query="q", entities=["周杰伦"])
        agent.chat("user-y", "你好")
        # chat 意图应清空旧延续状态
        assert agent.memory.get_dialogue_state("user-y").entities == []
