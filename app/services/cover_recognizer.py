"""专辑封面识别服务：上传封面图 → 识别专辑/歌手 → 产出可路由进知识链路的查询。

为什么独立于现有 LLM 客户端
----------------------------
DeepSeek API 没有视觉能力（仅文本模型），而现有 ``OpenAICompatibleLLM`` 会往请求体注入
DeepSeek 专有的 ``thinking`` 字段并对 ``content`` 做文本假设。视觉模型（默认阿里百炼
DashScope Qwen-VL）走标准 OpenAI 兼容 ``/chat/completions``，``content`` 要用 ``[{text},{image_url}]``
分段格式，且不需要 thinking 字段。故本模块用一条精简的 httpx 视觉调用，与 DeepSeek 客户端解耦。

识别三级降级
------------
1. 视觉模型（``VISION_LLM_API_KEY`` 配置即启用）：直接让 VLM 读封面，产出 {album, artist, confidence}。
2. 本地 OCR（``rapidocr-onnxruntime``，可选依赖）：视觉不可用 / 低置信时，读封面上的标题、歌手文字。
3. 都不行：返回 ``method="none"``，由上层提示用户直接输入专辑名/歌手。

识别结果再由 :func:`synthesize_query` 改写成多行结构化 query（``album\\n<名>\\n<歌手>\\n解读这张专辑``）：
前端把它喂给现有 ``/agent/stream``，于是整条 ``album_deep_dive`` 知识链路（消歧→元数据→乐评→档案）
原样复用，本模块不碰图、不碰 agent。多行格式让 ``resolve_music_entities`` 干净解析出专辑名/歌手，
末行的 ``这张专辑`` 保证关键词兜底命中 album_deep_dive 意图。
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import threading
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# 视觉模型系统/用户提示。要求严格 JSON 输出，便于解析；非专辑封面给空串 + 0 分。
_VISION_SYSTEM_PROMPT = (
    "你是音乐专辑封面识别专家。用户上传一张图片，你要判断它是不是音乐专辑/唱片封面，"
    "并识别出专辑名称和歌手/乐队名。只依据封面可见信息（标题、歌手名、厂牌 logo 等）判断，"
    "不确定就把置信度打低。不要编造图片上没有的信息。"
)
_VISION_USER_PROMPT = (
    "请识别这张图片对应的音乐专辑。严格只输出一个 JSON，不要 markdown、不要任何额外文字：\n"
    '{"album":"专辑名（没有则空字符串）","artist":"歌手或乐队名（没有则空字符串）",'
    '"confidence":0到1之间的置信度小数}。\n'
    "如果图片不是专辑封面或无法识别，album 与 artist 都给空字符串、confidence 给 0。"
)


@dataclass
class CoverRecognition:
    """封面识别结果。

    ``method`` 取值：
      - ``vision``：视觉模型直接给出 album/artist（置信达标）。
      - ``ocr``：只读到了封面文字（raw_text），album/artist 可能仍空，交由下游检索消歧。
      - ``none``：没识别出来，上层应提示用户输入。
    """

    album: str = ""
    artist: str = ""
    confidence: float = 0.0
    method: str = "none"
    raw_text: str = ""
    note: str = ""

    @property
    def usable(self) -> bool:
        """是否有可往下走的信号（拿到了专辑名，或至少读到了封面文字）。"""
        return bool(self.album or self.raw_text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "album": self.album,
            "artist": self.artist,
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "raw_text": self.raw_text,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# 图片预处理
# ---------------------------------------------------------------------------

def _encode_image_data_url(image_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"


def _resize_for_vision(image_bytes: bytes, max_side: int) -> tuple[bytes, str]:
    """把封面缩到长边 ≤ max_side，省视觉 token、避开尺寸上限。

    Pillow 是可选依赖（视觉功能才需要）：没装就直接返回原图。返回 (bytes, mime)。
    失败（损坏图 / 解码异常）也返回原图——视觉路径宁可原图直送也不阻断。
    """
    try:
        from PIL import Image, ImageOps  # type: ignore
    except Exception:  # Pillow 未装：原图直送
        return image_bytes, "image/jpeg"
    try:
        import io

        # 显式 decompression-bomb 上限（Pillow 默认 89M 像素略宽；专辑封面给 40M 足够且更稳）。
        Image.MAX_IMAGE_PIXELS = 40_000_000
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)  # 手机拍的封面按 EXIF 方向正向，避免横/倒着送 VLM/OCR
        img.load()
        # 统一转 RGB（PNG 透明通道 / 调色板）→ JPEG 体积小且兼容性好。
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        w, h = img.size
        scale = max_side / max(w, h)
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        logger.debug("封面缩图失败，原图直送视觉模型", exc_info=True)
        return image_bytes, "image/jpeg"


def build_thumbnail_data_url(image_bytes: bytes, max_side: int = 400) -> str:
    """生成供前端气泡显示的缩略图 data URI（无盘落地、无静态挂载）。

    严格模式：解码/缩放失败时返回空串，绝不回退成原图 data URI（原图可达 10MiB，
    base64 后膨胀进响应/聊天历史）。前端拿到空串就不渲染缩略图。
    """
    try:
        import io

        from PIL import Image, ImageOps  # type: ignore

        Image.MAX_IMAGE_PIXELS = 40_000_000
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        img.load()
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        w, h = img.size
        scale = max_side / max(w, h)
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        thumb = buf.getvalue()
    except Exception:
        logger.debug("缩略图生成失败，置空", exc_info=True)
        return ""
    # 兜底：缩略图若仍超 200KB（异常大图），不塞进响应。
    if len(thumb) > 200 * 1024:
        return ""
    return _encode_image_data_url(thumb, "image/jpeg")


# ---------------------------------------------------------------------------
# 视觉模型调用（模块级，便于测试 monkeypatch）
# ---------------------------------------------------------------------------

async def _vision_chat_completion(payload: dict[str, Any], headers: dict[str, str]) -> str:
    """POST {vision_base_url}/chat/completions，返回 assistant 文本。

    独立的精简调用：不注入 DeepSeek thinking 字段，content 用标准分段格式。
    失败抛异常，由上层捕获降级。此函数是测试 mock 的接缝。
    """
    timeout = httpx.Timeout(settings.vision_llm_timeout_seconds, connect=settings.vision_llm_connect_timeout)
    url = settings.vision_llm_base_url.rstrip("/") + "/chat/completions"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return (msg.get("content") or "").strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    """从模型输出里抠出第一个完整 JSON 对象（兼容 ```json``` 包裹 / 夹叙夹议 / 尾部带 ``}`` 的散文）。

    用花括号深度扫描而不是 ``rfind('}')``：模型可能写 ``{...} 置信度（90%）{备注}``，
    取最后一个 ``}`` 会把后面的非 JSON 文本包进来导致解析失败；从首个 ``{`` 起按字符串内
    转义计深度，到深度归零处截断，保证取到的是一个语法完整的对象。
    """
    if not text:
        return {}
    s = text.strip()
    # 去 markdown 代码围栏
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    start = s.find("{")
    if start == -1:
        return {}
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(s[start : i + 1])
                    return obj if isinstance(obj, dict) else {}
                except json.JSONDecodeError:
                    return {}
    return {}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, f))


async def recognize_cover_via_vision(image_bytes: bytes, mime: str) -> CoverRecognition | None:
    """视觉模型识别。未配置 key 返回 None（交由上层走 OCR）。

    返回 CoverRecognition（即便没认出来也返回 method=vision 的空结果，带 note）。
    """
    if not settings.vision_enabled:
        return None
    data, send_mime = _resize_for_vision(image_bytes, settings.vision_image_max_side)
    data_url = _encode_image_data_url(data, send_mime)
    payload = {
        "model": settings.vision_llm_model,
        "messages": [
            {"role": "system", "content": _VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _VISION_USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": settings.vision_llm_max_tokens,
    }
    headers = {"Authorization": f"Bearer {settings.vision_llm_api_key}", "Content-Type": "application/json"}
    try:
        text = await _vision_chat_completion(payload, headers)
    except Exception as exc:  # 网络 / 鉴权 / 解析错误：降级到 OCR
        logger.warning("视觉识别调用失败，降级 OCR：%s", exc)
        return CoverRecognition(method="vision", note=f"视觉调用失败：{exc}")
    parsed = _extract_json_object(text)
    album = str(parsed.get("album") or "").strip()
    artist = str(parsed.get("artist") or "").strip()
    confidence = _to_float(parsed.get("confidence"), 0.7 if (album or artist) else 0.0)
    if not album and not artist:
        return CoverRecognition(method="vision", confidence=0.0, note="视觉模型未识别出专辑")
    return CoverRecognition(album=album, artist=artist, confidence=confidence, method="vision")


# ---------------------------------------------------------------------------
# OCR 兜底（可选依赖 rapidocr-onnxruntime）
# ---------------------------------------------------------------------------

_OCR_ENGINE: Any = None
_OCR_INIT_LOCK = threading.Lock()


def _get_ocr_engine() -> Any:
    """懒加载并缓存 rapidocr 引擎（模型加载较重）。未装返回 None。"""
    global _OCR_ENGINE
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE
    with _OCR_INIT_LOCK:
        if _OCR_ENGINE is not None:
            return _OCR_ENGINE
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
        except Exception:
            logger.info("rapidocr-onnxruntime 未安装，OCR 兜底不可用（可 pip install rapidocr-onnxruntime 启用）")
            return None
        try:
            _OCR_ENGINE = RapidOCR()
        except Exception:
            logger.warning("rapidocr 引擎初始化失败", exc_info=True)
            _OCR_ENGINE = None
        return _OCR_ENGINE


def _ocr_run(image_bytes: bytes) -> CoverRecognition | None:
    """同步 OCR：rapidocr 在线程里跑。返回 raw_text 尽可能全的 CoverRecognition。"""
    engine = _get_ocr_engine()
    if engine is None:
        return None
    try:
        result, _elapsed = engine(image_bytes)
    except Exception:
        logger.warning("OCR 识别异常", exc_info=True)
        return None
    lines: list[str] = []
    for item in result or []:
        # rapidocr 每条: [box, text, score]
        try:
            txt = item[1]
        except Exception:
            continue
        if txt and str(txt).strip():
            lines.append(str(txt).strip())
    raw = " ".join(lines).strip()
    if not raw:
        return CoverRecognition(method="ocr", note="OCR 未读到文字")
    return CoverRecognition(raw_text=raw, method="ocr", confidence=0.4, note="OCR 读取封面文字")


async def recognize_cover_via_ocr(image_bytes: bytes) -> CoverRecognition | None:
    """OCR 兜底。关闭或未装 rapidocr 返回 None。"""
    if not settings.cover_ocr_enabled:
        return None
    return await asyncio.to_thread(_ocr_run, image_bytes)


# ---------------------------------------------------------------------------
# 主入口 + 查询改写
# ---------------------------------------------------------------------------

async def recognize_album_cover(image_bytes: bytes, mime: str = "image/jpeg") -> CoverRecognition:
    """封面识别主入口：视觉优先 → OCR 兜底 → none。永远返回 CoverRecognition，不抛错。"""
    vision = await recognize_cover_via_vision(image_bytes, mime)
    if vision is not None and vision.usable and vision.confidence >= settings.vision_confidence_threshold:
        return vision

    # 视觉不可用 / 低置信 / 空结果 → 试 OCR
    ocr = await recognize_cover_via_ocr(image_bytes)
    if ocr is not None and ocr.usable:
        # 视觉有低置信结果时仍优先于纯 OCR（结构化 album/artist 比裸文字更可信）。
        if vision is not None and vision.usable and vision.confidence >= ocr.confidence:
            return vision
        return ocr

    if vision is not None and vision.usable:
        return vision  # 低置信视觉仍是唯一线索，交下游 MusicBrainz 消歧校验
    return CoverRecognition(method="none", note="未能识别封面，请直接输入专辑名或歌手")


def synthesize_query(rec: CoverRecognition) -> str | None:
    """把识别结果改写成喂给现有 /agent/stream 的 query，命中 album_deep_dive 并可被实体解析。

    返回 None 表示没有可路由的线索，上层应提示用户输入。

    为什么用多行结构化格式而不是自然语言句子：resolve_music_entities 的自然语言分支
    （``_explicit_artist_entity_from_query`` 只认 的/-/:/'s 分隔符、``_strip_entity_noise``
    只去首尾噪声）解析不了「《Blonde》by Frank Ocean，介绍…」这类句子——整句会被当成专辑名，
    MusicBrainz 消歧失败、档案降级。改用 ``_structured_entity_from_query`` 专门解析的格式：
    第 1 行 ``album`` 钉类型、第 2 行专辑名、第 3 行歌手。末尾追加 ``解读这张专辑`` 一行：
    (a) ``_structured_entity_from_query`` 只读前 3 行，该行被忽略，不影响解析；
    (b) ``这张专辑`` 是 album_deep_dive 的关键词信号——plan_intent 对知识类意图以关键词兜底
        为准（nodes.py 165-174），保证稳定命中 album_deep_dive 而非 review_summary/artist_albums。
    """
    if rec.method == "vision" and rec.album:
        if rec.artist:
            return f"album\n{rec.album}\n{rec.artist}\n解读这张专辑"
        return f"album\n{rec.album}\n解读这张专辑"
    if rec.method == "ocr" and rec.raw_text:
        # OCR 只有裸文字：当专辑名交给 MusicBrainz 模糊匹配，比解析自然语言句子稳。
        # OCR 文本攻击者可控（恶意封面/印刷文字），剔除常见注入话术防越权流入下游 prompt。
        from app.prompts.untrusted_boundary import strip_directive_phrases
        return f"album\n{strip_directive_phrases(rec.raw_text)}\n解读这张专辑"
    return None
