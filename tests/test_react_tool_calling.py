"""验证新的真 ReAct + tool calling 循环工作。"""

from app.agent import CineSonicAgent
from app.llm.protocol import LLMResponse, ToolCall
from app.llm.tools import AGENT_TOOLS, TOOL_RECOMMEND, TOOL_SEARCH
from app.react_loop import ReActLoop
from app.storage import JsonStore


def test_tool_calling_loop_runs_recommend(tmp_path):
    """用 MockLLM 跑 chat，确认 trace 显示 tool_calls 流程。"""
    agent = CineSonicAgent(JsonStore(tmp_path / "store"))
    result = agent.chat("u1", "推荐一些适合工作的歌")

    assert result.answer
    assert result.agent_trace
    assert any("recommend" in s for s in result.agent_trace)
    # 验证新流程的 meta token 行存在
    assert any("[meta] tokens=" in s for s in result.agent_trace)
    # 验证有 finalize 步（LLM 主动收尾）或 max_steps
    assert any(("finalize" in s) or ("max_steps" in s) for s in result.agent_trace)


def test_max_steps_bound(tmp_path):
    """构造一个永远返回 tool_calls 的 LLM，验证 MAX_REACT_STEPS 强制终止。"""
    agent = CineSonicAgent(JsonStore(tmp_path / "store"))

    class InfiniteToolLLM:
        def generate(self, prompt, system=None, temperature=0.7):
            return ""

        def chat(self, messages, temperature=0.7):
            return ""

        def chat_with_tools(self, messages, tools, temperature=0.3, tool_choice="auto"):
            return LLMResponse(
                tool_calls=[ToolCall(id="x", name=TOOL_RECOMMEND, arguments={"query": "x"})],
                finish_reason="tool_calls",
            )

    agent.llm = InfiniteToolLLM()
    result = agent.chat("u1", "推荐")
    # 必须在 MAX_REACT_STEPS 内终止，trace 末尾包含 max_steps_reached
    assert any("max_steps" in s for s in result.agent_trace)


def test_unknown_tool_rejected(tmp_path):
    """LLM 返回不存在的工具时拒绝执行，不崩。"""
    agent = CineSonicAgent(JsonStore(tmp_path / "store"))

    state = {"called": False}

    class WeirdToolLLM:
        def generate(self, p, system=None, temperature=0.7):
            return ""

        def chat(self, m, temperature=0.7):
            return ""

        def chat_with_tools(self, messages, tools, temperature=0.3, tool_choice="auto"):
            if not state["called"]:
                state["called"] = True
                return LLMResponse(
                    tool_calls=[ToolCall(id="x", name="evil_tool", arguments={})],
                    finish_reason="tool_calls",
                )
            return LLMResponse(content="ok", finish_reason="stop")

    agent.llm = WeirdToolLLM()
    result = agent.chat("u1", "hi")
    assert result.answer
    assert any("evil_tool" in s and ("拒绝" in s or "Unknown" in s or "未知" in s) for s in result.agent_trace)


def test_legacy_think_still_works(tmp_path):
    """向后兼容：旧的 _think API 必须仍可用（_keyword_think fallback）。"""
    agent = CineSonicAgent(JsonStore(tmp_path / "store"))
    loop = ReActLoop(agent)
    actions, reason = loop._think("推荐一些歌", None, history=None)
    assert actions
    assert reason
