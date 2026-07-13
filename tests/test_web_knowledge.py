"""web_knowledge provider 抽象层离线单测。

全程不打真网络/LLM：_deepseek_agenerate 与 asearch_web_info/_asearch_duckduckgo 全 mock。

设计要点（P1.5）：DeepSeek parametric 改为**直答**（产出 answer_summary 全文，不再 claims→再合成），
且在 auto 链里对知识意图**提为首选**——名盘模型知识扎实，直答质量胜过稀疏网页检索。
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.services import web_knowledge as wk


@pytest.fixture(autouse=True)
def _clear_cache():
    wk.clear_cache()
    yield
    wk.clear_cache()


# ── JSON 抠取 ──


class TestExtractJson:
    def test_claims_array(self):
        obj = wk._extract_json_object('{"answer":"x","style_tags":["a"]}')
        assert obj["answer"] == "x"

    def test_fenced(self):
        assert wk._extract_json_object('```json\n{"answer":"x"}\n```') == {"answer": "x"}

    def test_prose_with_stray_brace(self):
        obj = wk._extract_json_object('好的 {"answer":"x"} 见附录}')
        assert obj["answer"] == "x"


# ── DeepSeek 先验 provider（直答）──


_ANSWER_JSON = '{"answer":"《Blonde》是 Frank Ocean 2016 年的实验性 R&B 专辑，广受乐评推崇。","style_tags":["R&B","art pop","实验"]}'


class TestDeepSeekParametric:
    @pytest.mark.anyio
    async def test_direct_answer_with_style_tags(self, monkeypatch):
        async def fake_gen(prompt, system=None, temperature=0.4):  # noqa: ARG001
            return _ANSWER_JSON

        monkeypatch.setattr(wk, "_deepseek_agenerate", fake_gen)
        r = await wk.deepseek_parametric_search(
            query="Blonde", intent="album_deep_dive", entities=["Blonde", "Frank Ocean"]
        )
        assert r.provider == "deepseek_parametric"
        assert r.usable and r.answer_summary  # 直答全文
        assert "Blonde" in r.answer_summary
        assert r.style_tags == ["R&B", "art pop", "实验"]
        assert r.confidence <= settings.deepseek_parametric_confidence_cap  # 置信封顶
        assert "未联网核实" in (r.degraded_reason or "")

    @pytest.mark.anyio
    async def test_disallowed_intent_returns_empty(self, monkeypatch):
        called = {"n": 0}

        async def fake_gen(*a, **k):  # noqa: ARG001
            called["n"] += 1
            return "{}"

        monkeypatch.setattr(wk, "_deepseek_agenerate", fake_gen)
        # concert_events / music_fact_check 必须真来源，禁用先验
        r = await wk.deepseek_parametric_search(
            query="The Weeknd tour", intent="concert_events", entities=["The Weeknd"]
        )
        assert not r.usable
        assert called["n"] == 0  # 根本没调 LLM
        assert "不适用" in (r.degraded_reason or "")

    @pytest.mark.anyio
    async def test_disabled(self, monkeypatch):
        monkeypatch.setattr(settings, "deepseek_parametric_enabled", False)
        r = await wk.deepseek_parametric_search(query="x", intent="album_deep_dive", entities=[])
        assert not r.usable and "未启用" in (r.degraded_reason or "")

    @pytest.mark.anyio
    async def test_llm_failure_degrades(self, monkeypatch):
        async def boom(*a, **k):  # noqa: ARG001
            raise RuntimeError("net")

        monkeypatch.setattr(wk, "_deepseek_agenerate", boom)
        r = await wk.deepseek_parametric_search(query="x", intent="album_deep_dive", entities=[])
        assert not r.usable and "调用失败" in (r.degraded_reason or "")

    @pytest.mark.anyio
    async def test_empty_answer_degrades(self, monkeypatch):
        async def fake_gen(*a, **k):  # noqa: ARG001
            return '{"answer":""}'

        monkeypatch.setattr(wk, "_deepseek_agenerate", fake_gen)
        r = await wk.deepseek_parametric_search(query="x", intent="album_deep_dive", entities=[])
        assert not r.usable and "未产出" in (r.degraded_reason or "")


# ── DeepSeek 先验 provider（同步兜底入口）──
# deepseek_parametric_search_sync 供 dossier 在 web_knowledge_search 工具超时/空时直接补生成，
# 行为须与 async 版一致（同一 prompt/解析），仅改走同步 _deepseek_generate 接缝。


class TestDeepSeekParametricSync:
    def test_direct_answer_with_style_tags(self, monkeypatch):
        def fake_gen(prompt, system=None, temperature=0.4):  # noqa: ARG001
            return _ANSWER_JSON

        monkeypatch.setattr(wk, "_deepseek_generate", fake_gen)
        r = wk.deepseek_parametric_search_sync(
            query="Blonde", intent="album_deep_dive", entities=["Blonde", "Frank Ocean"]
        )
        assert r.provider == "deepseek_parametric"
        assert r.usable and r.answer_summary
        assert "Blonde" in r.answer_summary
        assert r.style_tags == ["R&B", "art pop", "实验"]
        assert r.confidence <= settings.deepseek_parametric_confidence_cap
        assert "未联网核实" in (r.degraded_reason or "")

    def test_disallowed_intent_returns_empty(self, monkeypatch):
        called = {"n": 0}

        def fake_gen(*a, **k):  # noqa: ARG001
            called["n"] += 1
            return "{}"

        monkeypatch.setattr(wk, "_deepseek_generate", fake_gen)
        r = wk.deepseek_parametric_search_sync(
            query="The Weeknd tour", intent="concert_events", entities=["The Weeknd"]
        )
        assert not r.usable
        assert called["n"] == 0  # intent 门控挡掉，根本没调 LLM
        assert "不适用" in (r.degraded_reason or "")

    def test_disabled(self, monkeypatch):
        monkeypatch.setattr(settings, "deepseek_parametric_enabled", False)
        r = wk.deepseek_parametric_search_sync(query="x", intent="album_deep_dive", entities=[])
        assert not r.usable and "未启用" in (r.degraded_reason or "")

    def test_llm_failure_degrades(self, monkeypatch):
        def boom(*a, **k):  # noqa: ARG001
            raise RuntimeError("net")

        monkeypatch.setattr(wk, "_deepseek_generate", boom)
        r = wk.deepseek_parametric_search_sync(query="x", intent="album_deep_dive", entities=[])
        assert not r.usable and "调用失败" in (r.degraded_reason or "")

    def test_shared_parse_with_async(self, monkeypatch):
        # 同步/异步走同一 _build_parametric_result，empty 文本产生同一降级文案
        monkeypatch.setattr(wk, "_deepseek_generate", lambda *a, **k: '{"answer":""}')  # noqa: ARG001
        r = wk.deepseek_parametric_search_sync(query="x", intent="album_deep_dive", entities=[])
        assert not r.usable and "未产出" in (r.degraded_reason or "")


# ── dossier 侧兜底：maybe_parametric_rescue ──


class TestParametricRescue:
    def _spy(self, monkeypatch, calls):
        def fake_sync(**k):
            calls.append(k)
            return wk.WebKnowledgeResult(
                provider="deepseek_parametric",
                query=k["query"],
                answer_summary="直答正文……",
                style_tags=["R&B"],
                confidence=0.45,
                degraded_reason="DeepSeek 模型先验知识，未联网核实",
            )

        monkeypatch.setattr(wk, "deepseek_parametric_search_sync", fake_sync)

    def test_disallowed_intent_skips(self, monkeypatch):
        calls: list = []
        self._spy(monkeypatch, calls)
        # concert_events/fact_check 必须真来源，不在兜底白名单 → 不兜底
        assert (
            wk.maybe_parametric_rescue(query="tour", intent="concert_events", entities=["The Weeknd"], remaining=40.0)
            is None
        )
        assert (
            wk.maybe_parametric_rescue(query="核验", intent="music_fact_check", entities=["Blonde"], remaining=40.0)
            is None
        )
        assert calls == []

    def test_compare_is_rescued(self, monkeypatch):
        calls: list = []
        self._spy(monkeypatch, calls)
        # music_compare 现在消费直答正文（build_dossier compare 分支不再回落静态模板），
        # 故 web_knowledge 工具失败时也兜底一次直答，与 album/artist 一致。
        rescued = wk.maybe_parametric_rescue(
            query="A 和 B", intent="music_compare", entities=["A", "B"], remaining=40.0
        )
        assert rescued is not None and rescued.answer_summary
        assert len(calls) == 1 and calls[0]["intent"] == "music_compare"

    def test_low_budget_skips(self, monkeypatch):
        calls: list = []
        self._spy(monkeypatch, calls)
        # 剩余预算不足以完成一次生成 → 不启动注定被工具墙杀的调用
        assert (
            wk.maybe_parametric_rescue(query="Blonde", intent="album_deep_dive", entities=["Blonde"], remaining=5.0)
            is None
        )
        assert calls == []

    def test_no_deadline_allows_attempt(self, monkeypatch):
        calls: list = []
        self._spy(monkeypatch, calls)
        # remaining=None（无 deadline，如离线）→ 放行
        r = wk.maybe_parametric_rescue(query="Blonde", intent="album_deep_dive", entities=["Blonde"], remaining=None)
        assert r is not None and r.answer_summary
        assert len(calls) == 1

    def test_allowed_with_budget_delegates(self, monkeypatch):
        calls: list = []
        self._spy(monkeypatch, calls)
        r = wk.maybe_parametric_rescue(query="Blonde", intent="album_deep_dive", entities=["Blonde"], remaining=40.0)
        assert r is not None and r.answer_summary
        assert r.provider == "deepseek_parametric"
        assert r.style_tags == ["R&B"]
        assert len(calls) == 1 and calls[0]["intent"] == "album_deep_dive"


# ── web / ddg provider 包装 ──


class TestWebProviders:
    @pytest.mark.anyio
    async def test_tavily_wraps_sources_to_citations(self, monkeypatch):
        async def fake_asearch(query, max_results=5, api_key=""):  # noqa: ANN001
            return [{"title": "Pitchfork Blonde review", "url": "https://pitchfork.com/x", "content": "landmark"}]

        monkeypatch.setattr(wk.web_search_source, "asearch_web_info", fake_asearch)
        r = await wk.tavily_web_search(query="Blonde review", intent="review_summary", entities=["Blonde"])
        assert r.usable and len(r.sources) == 1
        assert r.sources[0].tier == "B" and r.sources[0].provenance == "web"
        assert len(r.citations) == 1 and r.citations[0]["url"].endswith("/x")

    @pytest.mark.anyio
    async def test_duckduckgo_capped_confidence(self, monkeypatch):
        async def fake_ddg(query, max_results=5):  # noqa: ANN001
            return [{"title": "t", "url": "https://example.org/a", "content": "c"}]

        monkeypatch.setattr(wk.web_search_source, "_asearch_duckduckgo", fake_ddg)
        r = await wk.duckduckgo_search(query="x", intent="album_deep_dive", entities=[])
        assert r.usable and r.confidence <= 0.45 and r.sources[0].tier == "C"


# ── auto 编排：parametric 优先 / web 兜底 / 缓存 / intent 门控 ──


class TestAutoOrchestration:
    @pytest.mark.anyio
    async def test_auto_uses_parametric_first(self, monkeypatch):
        # 知识意图：DeepSeek 直答优先，web 不应被调用
        asearch_calls = {"n": 0}

        async def fake_asearch(*a, **k):  # noqa: ARG001
            asearch_calls["n"] += 1
            return [{"title": "x", "url": "https://y", "content": "c"}]

        async def fake_gen(*a, **k):  # noqa: ARG001
            return _ANSWER_JSON

        monkeypatch.setattr(wk.web_search_source, "asearch_web_info", fake_asearch)
        monkeypatch.setattr(wk, "_deepseek_agenerate", fake_gen)
        r = await wk.run_web_knowledge_search(
            query="Blonde", intent="album_deep_dive", entities=["Blonde", "Frank Ocean"]
        )
        assert r.provider == "deepseek_parametric" and r.usable and r.answer_summary
        assert asearch_calls["n"] == 0  # parametric 命中，web 没跑
        assert r.cached is False

    @pytest.mark.anyio
    async def test_web_used_when_parametric_empty(self, monkeypatch):
        # parametric 返空 → 落到 web 兜底
        async def fake_asearch(query, max_results=5, api_key=""):  # noqa: ANN001
            return [{"title": "real review", "url": "https://pitchfork.com/b", "content": "c"}]

        async def fake_gen(*a, **k):  # noqa: ARG001
            return '{"answer":""}'

        monkeypatch.setattr(wk.web_search_source, "asearch_web_info", fake_asearch)
        monkeypatch.setattr(wk, "_deepseek_agenerate", fake_gen)
        r = await wk.run_web_knowledge_search(query="Blonde", intent="album_deep_dive", entities=["Blonde"])
        assert r.provider == "tavily" and r.usable

    @pytest.mark.anyio
    async def test_cache_hit_skips_providers(self, monkeypatch):
        gen_calls = {"n": 0}

        async def fake_gen(*a, **k):  # noqa: ARG001
            gen_calls["n"] += 1
            return _ANSWER_JSON

        monkeypatch.setattr(wk, "_deepseek_agenerate", fake_gen)
        await wk.run_web_knowledge_search(query="Blonde", intent="album_deep_dive", entities=[])
        assert gen_calls["n"] == 1
        r2 = await wk.run_web_knowledge_search(query="Blonde", intent="album_deep_dive", entities=[])
        assert r2.cached is True
        assert gen_calls["n"] == 1  # 第二次命中缓存，没再打 provider

    @pytest.mark.anyio
    async def test_concert_intent_skips_parametric(self, monkeypatch):
        # concert_events：parametric 被 intent 门控挡掉 → web/ddg 兜底也空 → 诚实不 usable
        async def empty_asearch(query, max_results=5, api_key=""):  # noqa: ANN001
            return []

        async def empty_ddg(query, max_results=5):  # noqa: ANN001
            return []

        gen_calls = {"n": 0}

        async def fake_gen(*a, **k):  # noqa: ARG001
            gen_calls["n"] += 1
            return "{}"

        monkeypatch.setattr(wk.web_search_source, "asearch_web_info", empty_asearch)
        monkeypatch.setattr(wk.web_search_source, "_asearch_duckduckgo", empty_ddg)
        monkeypatch.setattr(wk, "_deepseek_agenerate", fake_gen)
        r = await wk.run_web_knowledge_search(query="The Weeknd tour", intent="concert_events", entities=["The Weeknd"])
        assert not r.usable
        assert gen_calls["n"] == 0  # 先验被 intent 门控挡掉，根本没调

    @pytest.mark.anyio
    async def test_provider_none_returns_unusable(self, monkeypatch):
        monkeypatch.setattr(settings, "knowledge_search_provider", "none")
        r = await wk.run_web_knowledge_search(query="x", intent="album_deep_dive", entities=[])
        assert not r.usable


# ── web_knowledge_search 工具 handler（接 graph 那层）──


class TestWebKnowledgeToolHandler:
    @pytest.mark.anyio
    async def test_handler_returns_structured_result(self, monkeypatch):
        from app.tools.contracts import ToolContext, ToolStatus
        from app.tools.handlers import _web_knowledge_search_async

        async def fake_run(*, query, intent, entities, mode):  # noqa: ARG001
            return wk.WebKnowledgeResult(
                provider="deepseek_parametric",
                query=query,
                answer_summary="《Blonde》是 Frank Ocean 2016 年专辑……",
                style_tags=["R&B", "art pop"],
                confidence=0.45,
                degraded_reason="DeepSeek 模型先验，未联网核实",
            )

        monkeypatch.setattr(wk, "run_web_knowledge_search", fake_run)
        result = await _web_knowledge_search_async(
            {"query": "Blonde", "intent": "album_deep_dive"},
            ToolContext(thread_id="t", user_id="u", query="Blonde"),
        )
        assert result.tool == "web_knowledge_search"
        assert result.status == ToolStatus.OK
        assert result.data["type"] == "web_knowledge"
        assert result.data["provider"] == "deepseek_parametric"
        assert "Blonde" in result.data["answer_summary"]
        assert result.data["style_tags"] == ["R&B", "art pop"]

    @pytest.mark.anyio
    async def test_handler_legacy_fallback_when_provider_empty(self, monkeypatch):
        """provider 链全空 → handler 内部回退 legacy review_search，citations 被填上。"""
        from app import knowledge
        from app.tools.contracts import ToolContext, ToolStatus
        from app.tools.handlers import _web_knowledge_search_async

        async def empty_run(*, query, intent, entities, mode):  # noqa: ARG001
            return wk.WebKnowledgeResult(provider="none", query=query, degraded_reason="无可用 provider")

        monkeypatch.setattr(wk, "run_web_knowledge_search", empty_run)
        monkeypatch.setattr(
            knowledge,
            "search_reviews",
            lambda *a, **k: {
                "citations": [
                    {
                        "source": "last.fm",
                        "title": "Blonde",
                        "url": "https://last.fm/x",
                        "kind": "review",
                        "confidence": 0.6,
                    }
                ],
                "opinions": [],
            },
        )

        result = await _web_knowledge_search_async(
            {"query": "Blonde", "intent": "album_deep_dive"},
            ToolContext(thread_id="t", user_id="u", query="Blonde"),
        )
        assert result.status == ToolStatus.OK  # legacy 兜底让它 usable
        assert result.data["citations"], "legacy fallback 应填入 citations"
        assert result.data["citations"][0]["url"] == "https://last.fm/x"
