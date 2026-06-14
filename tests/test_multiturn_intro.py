"""第三步：多轮上下文 + LLM 个性化引言 的核心行为测试。

锁住两件事：
1. plan_with_llm 把对话历史拼进 LLM prompt（多轮指代能力的前提）。
2. compose_answer 的引言可由 LLM 生成，但歌名清单始终来自真实候选；
   LLM 不可用或输出异常时回退确定性模板。
"""
from __future__ import annotations

from app.graph.nodes import compose_answer, plan_with_llm
from app.models import AgentPlan, ExternalTrack


class _SpyLLM:
    """记录最后一次 generate 的 prompt，返回可控文本。"""
    def __init__(self, reply: str = ""):
        self.last_prompt = ""
        self.last_system = None
        self.reply = reply

    def generate(self, prompt, system=None, temperature=0.7):
        self.last_prompt = prompt
        self.last_system = system
        return self.reply


class _Agent:
    def __init__(self, llm):
        self.llm = llm


class TestMultiTurnContext:
    def test_history_threaded_into_prompt(self):
        spy = _SpyLLM(reply='{"intent":"recommend","entities":["周杰伦"],'
                            '"use_local":true,"use_vector":false,"use_web":true,'
                            '"target_count":null,"reasoning":"延续上一轮歌手"}')
        agent = _Agent(spy)
        plan = plan_with_llm(agent, "再来几首", history_text="user: 找周杰伦的歌\nassistant: 好的")
        assert plan is not None
        assert "周杰伦" in spy.last_prompt
        assert "再来几首" in spy.last_prompt
        assert plan.retrieval_plan.entities == ["周杰伦"]

    def test_no_history_keeps_plain_prompt(self):
        spy = _SpyLLM(reply='{"intent":"search","entities":[],"use_local":true,'
                            '"use_vector":false,"use_web":true,"target_count":null,'
                            '"reasoning":"x"}')
        agent = _Agent(spy)
        plan_with_llm(agent, "找点歌")
        assert "最近对话" not in spy.last_prompt


class TestComposeIntro:
    def _tracks(self):
        return [
            ExternalTrack(external_id="1", title="Blinding Lights", artist="The Weeknd", source="netease"),
            ExternalTrack(external_id="2", title="Save Your Tears", artist="The Weeknd", source="netease"),
        ]

    def test_llm_intro_used_when_clean(self):
        spy = _SpyLLM(reply="懂你想要的氛围，这几首很对味：")
        agent = _Agent(spy)
        plan = AgentPlan(intent="recommend", tools_needed=["recommend"])
        out = compose_answer("推荐 The Weeknd", [{"type": "web_music_search", "tracks": self._tracks()}],
                             plan, agent=agent, memory_query="喜欢 R&B")
        assert "懂你想要的氛围" in out
        # 歌名清单仍是确定性拼接，来自真实候选
        assert "《Blinding Lights》" in out
        assert "《Save Your Tears》" in out

    def test_llm_intro_with_bookmark_falls_back(self):
        """LLM 引言里出现书名号（疑似编造歌名）→ 回退安全模板。"""
        spy = _SpyLLM(reply="推荐《我编的歌》给你")
        agent = _Agent(spy)
        plan = AgentPlan(intent="recommend", tools_needed=["recommend"])
        out = compose_answer("推荐", [{"type": "web_music_search", "tracks": self._tracks()}],
                             plan, agent=agent, memory_query="")
        assert "我编的歌" not in out
        assert "可追溯候选" in out  # 回退模板特征

    def test_no_agent_uses_template(self):
        plan = AgentPlan(intent="recommend", tools_needed=["recommend"])
        out = compose_answer("推荐", [{"type": "web_music_search", "tracks": self._tracks()}], plan)
        assert "可追溯候选" in out
        assert "《Blinding Lights》" in out
