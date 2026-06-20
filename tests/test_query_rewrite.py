"""查询改写（contextual query rewriting）核心行为测试。

锁住 P1 的关键修复：「不要中文歌曲」这类纯否定/带上下文约束，由 LLM 在 query_plan
阶段改写成自包含的正向 search_query（融合多轮历史 + 否定转正向），检索层优先用它，
而非把否定原样发给搜索 API 再事后删空。
"""
from __future__ import annotations

import asyncio

from app.graph.nodes import _apply_language_filter, _query_with_entities, plan_with_llm_async
from app.models import AgentPlan, ExternalTrack, RetrievalPlan
from app.tools.handlers import _filter_content_exclusions


class _StubLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_prompt = ""

    async def agenerate(self, prompt, system=None, temperature=0.7):
        self.last_prompt = prompt
        return self.reply


class _Agent:
    def __init__(self, llm):
        self.llm = llm


def test_negative_constraint_rewritten_to_positive_query():
    """「不要中文歌曲」+ 深夜历史 → search_query 转正向英文词 + language=en。"""
    stub = _StubLLM(
        '{"intent":"recommend","entities":[],"use_local":true,"use_vector":true,'
        '"use_web":true,"search_query":"深夜 英文歌 欧美 安静","language":"en",'
        '"target_count":null,"reasoning":"延续深夜场景转英文"}'
    )
    plan = asyncio.run(plan_with_llm_async(
        _Agent(stub), "不要中文歌曲",
        history_text="user: 推荐几首适合深夜的歌\nassistant: 推荐了7首深夜歌曲",
    ))
    assert plan is not None
    rp = plan.retrieval_plan
    # 关键：否定被转成正向检索词，且保留了上一轮"深夜"上下文
    assert "英文" in rp.search_query or "欧美" in rp.search_query
    assert "深夜" in rp.search_query
    assert rp.language_filter == "en"
    # 历史确实进了 prompt
    assert "深夜" in stub.last_prompt


def test_query_with_entities_prefers_search_query():
    """检索层以 LLM 的 search_query 为基底，而非原始否定句。"""
    plan = AgentPlan(
        intent="recommend",
        retrieval_plan=RetrievalPlan(search_query="深夜 英文歌 欧美", language_filter="en"),
    )
    out = _query_with_entities("不要中文歌曲", plan)
    assert "英文" in out
    assert "不要中文歌曲" not in out  # 原始否定句不再原样发给搜索层


def test_query_with_entities_falls_back_when_no_search_query():
    """search_query 为空时降级回原始 query（mock/无 key 路径不变）。"""
    plan = AgentPlan(intent="search", retrieval_plan=RetrievalPlan(entities=["Beyond"]))
    out = _query_with_entities("找点歌", plan)
    assert "找点歌" in out and "Beyond" in out


def _track(title, artist=""):
    return ExternalTrack(external_id=title, title=title, artist=artist, source="netease")


def test_language_filter_keeps_only_target_language():
    tracks = [_track("Blinding Lights", "The Weeknd"), _track("晴天", "周杰伦"),
              _track("Shape of You", "Ed Sheeran"), _track("七里香", "周杰伦")]
    kept = _apply_language_filter(tracks, "en", target=2)
    titles = {t.title for t in kept}
    assert "Blinding Lights" in titles and "Shape of You" in titles
    assert "晴天" not in titles and "七里香" not in titles


def test_language_filter_backs_off_when_too_few():
    """过滤后候选过少（<目标一半）→ 回退不过滤，避免删空。"""
    tracks = [_track("晴天", "周杰伦"), _track("七里香", "周杰伦"), _track("Blinding Lights", "The Weeknd")]
    kept = _apply_language_filter(tracks, "en", target=6)  # 只有 1 首英文，<3 → 回退
    assert len(kept) == 3


def test_language_filter_noop_for_unknown_language():
    tracks = [_track("晴天", "周杰伦")]
    assert _apply_language_filter(tracks, "", target=5) == tracks
    assert _apply_language_filter(tracks, "ja", target=5) == tracks  # ja 不可判 → 不过滤


def test_hard_content_exclusions_filter_language_aliases_and_scripts():
    tracks = [
        _track("Vietnamese Chill Mix", "Demo"),
        _track("đêm bình yên", "Ca sĩ"),
        _track("夜に駆ける", "YOASOBI"),
        _track("Late Night R&B", "Demo"),
    ]
    no_vietnamese = _filter_content_exclusions(tracks, ["越南语"])
    assert [track.title for track in no_vietnamese] == ["夜に駆ける", "Late Night R&B"]
    no_japanese = _filter_content_exclusions(tracks, ["日本语"])
    assert all(track.title != "夜に駆ける" for track in no_japanese)


def test_mock_llm_also_rewrites_negation():
    """零依赖 mock 路径也把"不要中文"转正向 + 标 language=en（demo 不搜空）。"""
    from app.llm.mock import MockLLM
    from app.prompts.query_plan import QUERY_PLAN_SYSTEM

    raw = MockLLM().generate(
        "user: 推荐几首适合深夜的歌\nassistant: ok\n不要中文歌曲",
        system=QUERY_PLAN_SYSTEM,
    )
    import json
    data = json.loads(raw)
    assert data["language"] == "en"
    assert "英文" in data["search_query"] or "欧美" in data["search_query"]
    assert "深夜" in data["search_query"]  # 保留了场景上下文
