"""专辑封面识别服务离线单测。

全程不打真实网络：视觉调用走模块级 ``_vision_chat_completion`` 接缝 mock，
OCR 走 ``_get_ocr_engine`` 接缝 mock，图片用 Pillow 现造的小图。确定、可重复。
"""
from __future__ import annotations

import io

import pytest

from app.config import settings
from app.services import cover_recognizer as cr
from app.services.cover_recognizer import CoverRecognition


def _tiny_png(size=(8, 8), color=(255, 128, 0)) -> bytes:
    """用 Pillow 现造一张小 PNG（确定、零网络）。Pillow 是已装依赖。"""
    from PIL import Image

    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _vision_returning(text: str):
    """造一个替换 _vision_chat_completion 的 async 函数，固定返回 text。"""

    async def _fake(payload, headers):  # noqa: ARG001
        return text

    return _fake


def _fake_ocr_engine(lines: list[str]):
    """造一个假的 rapidocr 引擎：返回 (items, elapsed)，item 形如 [box, text, score]。"""

    def _call(_image_bytes):  # noqa: ANN001
        return [[[None, t, 0.9] for t in lines], 0.01]

    return _call


# ── synthesize_query 各分支 ──


class TestSynthesizeQuery:
    def test_vision_with_artist(self):
        q = cr.synthesize_query(CoverRecognition(album="Blonde", artist="Frank Ocean", confidence=0.9, method="vision"))
        # 结构化多行：line0=album 钉类型、line1=名、line2=歌手
        assert q and q.startswith("album\n") and "Blonde" in q and "Frank Ocean" in q

    def test_vision_without_artist(self):
        q = cr.synthesize_query(CoverRecognition(album="Blonde", confidence=0.6, method="vision"))
        assert q and q.startswith("album\n") and "Blonde" in q

    def test_ocr(self):
        q = cr.synthesize_query(CoverRecognition(raw_text="BLONDE Frank Ocean", method="ocr", confidence=0.4))
        assert q and q.startswith("album\n") and "BLONDE" in q

    def test_none_returns_none(self):
        assert cr.synthesize_query(CoverRecognition(method="none")) is None

    def test_empty_returns_none(self):
        assert cr.synthesize_query(CoverRecognition()) is None


class TestSynthesizedQueryResolves:
    """回归铁律：synthesize_query 的产物必须 (1) 命中 album_deep_dive 意图，且
    (2) 被 resolve_music_entities 干净解析出专辑名/歌手——否则封面识别对了、
    知识档案却因实体名乱码而 MusicBrainz 消歧失败（曾经 end-to-end 失效的 critical bug）。"""

    def test_vision_with_artist_routes_and_resolves(self):
        from app.intents import match_intent_by_keywords
        from app.knowledge import resolve_music_entities

        q = cr.synthesize_query(CoverRecognition(album="Blonde", artist="Frank Ocean", confidence=0.9, method="vision"))
        assert match_intent_by_keywords(q) == "album_deep_dive"
        ent = resolve_music_entities(q, "album_deep_dive")[0]
        assert ent.type == "album" and ent.name == "Blonde" and ent.artist == "Frank Ocean"

    def test_vision_no_artist_resolves(self):
        from app.knowledge import resolve_music_entities

        q = cr.synthesize_query(CoverRecognition(album="OK Computer", confidence=0.6, method="vision"))
        assert resolve_music_entities(q, "album_deep_dive")[0].name == "OK Computer"

    def test_ocr_resolves_nonempty(self):
        from app.knowledge import resolve_music_entities

        q = cr.synthesize_query(CoverRecognition(raw_text="BLONDE Frank Ocean", method="ocr", confidence=0.4))
        ent = resolve_music_entities(q, "album_deep_dive")[0]
        assert ent.type == "album" and ent.name  # 非空，交给 MB 模糊匹配


# ── JSON 抠取鲁棒性 ──


class TestExtractJson:
    def test_plain_json(self):
        assert cr._extract_json_object('{"album":"X","artist":"Y","confidence":0.5}') == {"album": "X", "artist": "Y", "confidence": 0.5}

    def test_fenced_json(self):
        assert cr._extract_json_object('```json\n{"album":"X"}\n```') == {"album": "X"}

    def test_json_with_prose(self):
        assert cr._extract_json_object('好的，识别结果是 {"album":"X","artist":"Y"} 希望有帮助') == {"album": "X", "artist": "Y"}

    def test_empty_or_garbage(self):
        assert cr._extract_json_object("") == {}
        assert cr._extract_json_object("没有json这里") == {}

    def test_trailing_prose_with_stray_brace(self):
        """模型在 JSON 后写散文且散文含 } ：取第一个完整对象，不被尾部 } 干扰。"""
        text = '{"album":"Blonde","artist":"Frank Ocean","confidence":0.9} 置信度很高（见附录}）'
        assert cr._extract_json_object(text) == {"album": "Blonde", "artist": "Frank Ocean", "confidence": 0.9}

    def test_two_objects_takes_first_complete(self):
        assert cr._extract_json_object('{"album":"A"} 后续 {"album":"B"}') == {"album": "A"}

    def test_escaped_brace_inside_string(self):
        assert cr._extract_json_object('{"album":"{Not the end}","artist":"X"}') == {"album": "{Not the end}", "artist": "X"}


# ── 视觉识别 ──


@pytest.fixture
def vision_enabled(monkeypatch):
    monkeypatch.setattr(settings, "vision_llm_api_key", "test-key")


class TestRecognizeViaVision:
    @pytest.mark.anyio
    async def test_high_confidence(self, vision_enabled, monkeypatch):
        monkeypatch.setattr(
            cr, "_vision_chat_completion",
            _vision_returning('{"album":"Blonde","artist":"Frank Ocean","confidence":0.92}'),
        )
        rec = await cr.recognize_cover_via_vision(_tiny_png(), "image/png")
        assert rec is not None
        assert rec.method == "vision"
        assert rec.album == "Blonde"
        assert rec.artist == "Frank Ocean"
        assert rec.confidence == pytest.approx(0.92)

    @pytest.mark.anyio
    async def test_not_recognized_empty(self, vision_enabled, monkeypatch):
        monkeypatch.setattr(cr, "_vision_chat_completion", _vision_returning('{"album":"","artist":"","confidence":0}'))
        rec = await cr.recognize_cover_via_vision(_tiny_png(), "image/png")
        assert rec is not None
        assert rec.method == "vision"
        assert not rec.usable

    @pytest.mark.anyio
    async def test_disabled_returns_none(self, monkeypatch):
        monkeypatch.setattr(settings, "vision_llm_api_key", "")
        rec = await cr.recognize_cover_via_vision(_tiny_png(), "image/png")
        assert rec is None

    @pytest.mark.anyio
    async def test_http_error_degrades(self, vision_enabled, monkeypatch):
        """视觉调用抛错 → 返回带 note 的空结果（usable=False），不向上抛。"""
        async def _boom(payload, headers):  # noqa: ARG001
            raise RuntimeError("network down")

        monkeypatch.setattr(cr, "_vision_chat_completion", _boom)
        rec = await cr.recognize_cover_via_vision(_tiny_png(), "image/png")
        assert rec is not None
        assert not rec.usable


# ── 主入口编排：视觉优先 → OCR 兜底 → none ──


class TestRecognizeAlbumCover:
    @pytest.mark.anyio
    async def test_vision_path_used(self, vision_enabled, monkeypatch):
        monkeypatch.setattr(
            cr, "_vision_chat_completion",
            _vision_returning('{"album":"Blonde","artist":"Frank Ocean","confidence":0.9}'),
        )
        rec = await cr.recognize_album_cover(_tiny_png(), "image/png")
        assert rec.method == "vision" and rec.album == "Blonde"

    @pytest.mark.anyio
    async def test_ocr_fallback_when_no_vision(self, monkeypatch):
        # 无视觉 key + OCR 引擎返回文字 → ocr 结果
        monkeypatch.setattr(settings, "vision_llm_api_key", "")
        monkeypatch.setattr(cr, "_get_ocr_engine", lambda: _fake_ocr_engine(["BLONDE", "Frank Ocean"]))
        rec = await cr.recognize_album_cover(_tiny_png(), "image/png")
        assert rec.method == "ocr"
        assert "BLONDE" in rec.raw_text and rec.usable

    @pytest.mark.anyio
    async def test_none_when_vision_off_and_ocr_disabled(self, monkeypatch):
        monkeypatch.setattr(settings, "vision_llm_api_key", "")
        monkeypatch.setattr(settings, "cover_ocr_enabled", False)
        rec = await cr.recognize_album_cover(_tiny_png(), "image/png")
        assert rec.method == "none" and not rec.usable

    @pytest.mark.anyio
    async def test_none_when_ocr_engine_unavailable(self, monkeypatch):
        # 视觉关、OCR 开但引擎拿不到（rapidocr 未装）→ none
        monkeypatch.setattr(settings, "vision_llm_api_key", "")
        monkeypatch.setattr(cr, "_get_ocr_engine", lambda: None)
        rec = await cr.recognize_album_cover(_tiny_png(), "image/png")
        assert rec.method == "none"

    @pytest.mark.anyio
    async def test_low_confidence_vision_falls_through_to_ocr(self, vision_enabled, monkeypatch):
        """视觉给了名字但置信 0.2 < 阈值 0.5 → 落到 OCR，且 OCR 置信更高取胜。"""
        monkeypatch.setattr(
            cr, "_vision_chat_completion",
            _vision_returning('{"album":"Maybe Blonde","artist":"?","confidence":0.2}'),
        )
        monkeypatch.setattr(cr, "_get_ocr_engine", lambda: _fake_ocr_engine(["Real Title", "Real Artist"]))
        rec = await cr.recognize_album_cover(_tiny_png(), "image/png")
        assert rec.method == "ocr"
        assert "Real Title" in rec.raw_text

    @pytest.mark.anyio
    async def test_subthreshold_vision_still_beats_ocr_when_higher(self, vision_enabled, monkeypatch):
        """视觉 0.45 < 阈值，但仍 ≥ OCR 的 0.4 → 视觉胜出（comparison 分支）。"""
        monkeypatch.setattr(
            cr, "_vision_chat_completion",
            _vision_returning('{"album":"Blonde","artist":"Frank Ocean","confidence":0.45}'),
        )
        monkeypatch.setattr(cr, "_get_ocr_engine", lambda: _fake_ocr_engine(["Wrong"]))
        rec = await cr.recognize_album_cover(_tiny_png(), "image/png")
        assert rec.method == "vision" and rec.album == "Blonde"

    @pytest.mark.anyio
    async def test_low_conf_vision_survives_when_no_ocr(self, vision_enabled, monkeypatch):
        """视觉低置信 + OCR 不可用 → 仍返回低置信视觉（好过 none，交 MB 校验）。"""
        monkeypatch.setattr(
            cr, "_vision_chat_completion",
            _vision_returning('{"album":"Blonde","artist":"Frank Ocean","confidence":0.3}'),
        )
        monkeypatch.setattr(cr, "_get_ocr_engine", lambda: None)
        rec = await cr.recognize_album_cover(_tiny_png(), "image/png")
        assert rec.method == "vision" and rec.album == "Blonde"

    @pytest.mark.anyio
    async def test_vision_error_then_ocr_recovers(self, vision_enabled, monkeypatch):
        """视觉调用抛错 → 编排层降级到 OCR 并返回 raw_text（不向上抛）。"""
        async def _boom(payload, headers):  # noqa: ARG001
            raise RuntimeError("network down")

        monkeypatch.setattr(cr, "_vision_chat_completion", _boom)
        monkeypatch.setattr(cr, "_get_ocr_engine", lambda: _fake_ocr_engine(["BLONDE", "Frank Ocean"]))
        rec = await cr.recognize_album_cover(_tiny_png(), "image/png")
        assert rec.method == "ocr"
        assert "BLONDE" in rec.raw_text and rec.usable


# ── 缩略图 ──


class TestThumbnail:
    def test_build_thumbnail_data_url(self):
        url = cr.build_thumbnail_data_url(_tiny_png(size=(120, 120)))
        assert url.startswith("data:image/")
        assert "base64," in url

    def test_resize_keeps_under_max(self):
        # 大图缩到长边 ≤ 64
        url = cr.build_thumbnail_data_url(_tiny_png(size=(800, 400)), max_side=64)
        assert url.startswith("data:image/")

    def test_garbage_returns_empty(self):
        """非图/损坏输入：严格模式返回空串，绝不回退成原图 data URI（防 ~10MiB 膨胀）。"""
        assert cr.build_thumbnail_data_url(b"not an image at all") == ""
        assert cr.build_thumbnail_data_url(b"") == ""
