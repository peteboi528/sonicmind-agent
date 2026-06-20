from __future__ import annotations

import asyncio

import app.memory as memory_mod
from app.agent import CineSonicAgent
from app.config import settings
from app.memory import MemoryManager
from app.models import EpisodicMemory, UserMemory
from app.storage import JsonStore


class StubLLM:
    """可编程的假 LLM：按调用记录返回预设 JSON，用于驱动 LLM 抽取/巩固路径。"""

    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    async def agenerate(self, prompt: str, system: str | None = None, temperature: float = 0.7) -> str:
        self.calls.append(f"{system or ''}\n{prompt}")
        sys = system or ""
        if "偏好抽取器" in sys:
            return self.responses.get("extract", '{"preferences": []}')
        if "画像总结器" in sys:
            return self.responses.get("consolidate", '{"profile": ""}')
        return "{}"


def _force_real_llm(monkeypatch, mgr: MemoryManager, llm) -> None:
    """让 _llm_ready() 为真：注入 llm + 关掉 mock_mode + 开 semantic。"""
    mgr.llm = llm
    monkeypatch.setattr(settings, "llm_api_key", "test-key", raising=False)
    monkeypatch.setattr(settings, "enable_semantic_memory", True, raising=False)


def _learn(manager: MemoryManager, user_id: str, query: str) -> bool:
    return asyncio.run(manager.auto_learn_from_turn_async(user_id, query, []))


def test_regex_path_unchanged_without_llm(tmp_path):
    """无 key（mock 模式）时：LLM 抽取/巩固不触发，正则路径照常工作。"""
    agent = CineSonicAgent(JsonStore(tmp_path / "store"))
    assert settings.mock_mode  # 测试环境默认无 key
    _learn(agent.memory, "u1", "我喜欢慵懒爵士")
    mem = agent.memory.get_memory("u1")
    assert any("慵懒爵士" in e.text for e in mem.structured_preferences)
    # 巩固画像在 mock 下不应被 LLM 填充
    assert mem.consolidated_profile == ""


def test_llm_extract_fallback_on_oblique_phrasing(tmp_path, monkeypatch):
    """正则抽不到的绕口表述，LLM 兜底抽出结构化偏好。"""
    mgr = MemoryManager(JsonStore(tmp_path / "store"))
    llm = StubLLM({"extract": '{"preferences": ["带电子元素的国摇"]}'})
    _force_real_llm(monkeypatch, mgr, llm)
    # 这句话不含"喜欢/偏好"等正则锚点
    _learn(mgr, "u2", "最近上头那种带点电子的国摇")
    mem = mgr.get_memory("u2")
    assert any("带电子元素的国摇" in e.text for e in mem.structured_preferences)
    assert any("偏好抽取器" in c for c in llm.calls)


def test_llm_extract_not_called_when_regex_hits(tmp_path, monkeypatch):
    """正则命中时不浪费 LLM 调用（仅在未命中时兜底）。"""
    mgr = MemoryManager(JsonStore(tmp_path / "store"))
    llm = StubLLM({"extract": '{"preferences": ["不该出现"]}'})
    _force_real_llm(monkeypatch, mgr, llm)
    _learn(mgr, "u3", "我喜欢慵懒爵士")
    mem = mgr.get_memory("u3")
    assert not any("不该出现" in e.text for e in mem.structured_preferences)


def test_consolidation_after_interval(tmp_path, monkeypatch):
    """达到巩固间隔后，LLM 把零散偏好巩固成一句话画像。"""
    mgr = MemoryManager(JsonStore(tmp_path / "store"))
    llm = StubLLM({"consolidate": '{"profile": "偏爱慵懒爵士与 city pop 的夜归听众"}'})
    _force_real_llm(monkeypatch, mgr, llm)
    monkeypatch.setattr(settings, "memory_consolidation_interval", 2, raising=False)
    # 先种一条偏好，避免巩固时无信号
    _learn(mgr, "u4", "我喜欢慵懒爵士")
    _learn(mgr, "u4", "随便放点歌")  # 第 2 轮触发巩固
    mem = mgr.get_memory("u4")
    assert "慵懒爵士" in mem.consolidated_profile
    assert mem.turns_since_consolidation == 0


def test_recall_falls_back_to_recent_without_embeddings(tmp_path, monkeypatch):
    """无 embeddings 时，召回退化为按时间取最近几条，不崩。"""
    monkeypatch.setattr(memory_mod.embeddings, "embeddings_available", lambda: False)
    mgr = MemoryManager(JsonStore(tmp_path / "store"))
    monkeypatch.setattr(settings, "enable_semantic_memory", True, raising=False)
    _learn(mgr, "u5", "想听慵懒爵士")
    _learn(mgr, "u5", "来点 city pop")
    recalled = mgr.recall_episodes("u5", "爵士", top_k=2)
    assert recalled  # 退化路径仍返回最近条目
    assert "city pop" in recalled[0]  # 最近的在前


def test_semantic_recall_ranks_by_cosine(tmp_path, monkeypatch):
    """有 embeddings 时，召回按 cosine 相似度排序，相关项命中。"""
    monkeypatch.setattr(memory_mod.embeddings, "embeddings_available", lambda: True)

    def fake_encode(texts):
        # 简单 1-hot 风向量：含"爵士"→[1,0]，含"摇滚"→[0,1]，否则混合
        out = []
        for t in texts:
            if "爵士" in t:
                out.append([1.0, 0.0])
            elif "摇滚" in t:
                out.append([0.0, 1.0])
            else:
                out.append([0.7, 0.7])
        return out

    monkeypatch.setattr(memory_mod.embeddings, "encode", fake_encode)
    mgr = MemoryManager(JsonStore(tmp_path / "store"))
    monkeypatch.setattr(settings, "enable_semantic_memory", True, raising=False)
    _learn(mgr, "u6", "想听慵懒爵士")
    _learn(mgr, "u6", "来点硬核摇滚")
    recalled = mgr.recall_episodes("u6", "爵士乐推荐", top_k=1)
    assert recalled == ["想听慵懒爵士"]


def test_episodic_memory_capped(tmp_path, monkeypatch):
    """情景记忆超出 cap 时丢弃最旧，防止无限增长。"""
    mgr = MemoryManager(JsonStore(tmp_path / "store"))
    monkeypatch.setattr(settings, "enable_semantic_memory", True, raising=False)
    monkeypatch.setattr(settings, "episodic_memory_cap", 10, raising=False)
    for i in range(20):
        _learn(mgr, "u7", f"第 {i} 次想听不同的歌曲")
    mem = mgr.get_memory("u7")
    assert len(mem.episodic_memory) <= 10


def test_episodic_model_roundtrip():
    """EpisodicMemory 序列化/反序列化稳定，UserMemory 含新字段默认值。"""
    ep = EpisodicMemory(text="慵懒爵士", embedding=[0.1, 0.2])
    restored = EpisodicMemory.model_validate(ep.model_dump())
    assert restored.text == "慵懒爵士"
    assert restored.kind == "episodic"
    mem = UserMemory(user_id="x")
    assert mem.episodic_memory == []
    assert mem.consolidated_profile == ""
    assert mem.turns_since_consolidation == 0
