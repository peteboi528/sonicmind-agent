"""Web 前端路由：Vue 单页应用服务 + 播放代理。"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response

from app import netease_auth
from app.api.main import _effective_user_id, agent
from app.config import settings
from app.services.cover_recognizer import (
    CoverRecognition,
    build_thumbnail_data_url,
    recognize_album_cover,
    synthesize_query,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["web"])

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"
_DIST_DIR = _WEB_DIR / "dist"


def _sniff_image_mime(data: bytes) -> str | None:
    """按魔数字节判真实图片类型，不信客户端 Content-Type（可伪造）。"""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


async def _read_capped(file: UploadFile, cap: int) -> bytes:
    """分块读取上传文件，累计超过 cap 立即 413 中止——不一次性把超大文件读进内存。"""
    buf = bytearray()
    while True:
        chunk = await file.read(64 * 1024)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > cap:
            raise HTTPException(status_code=413, detail="图片过大")
    return bytes(buf)


# ---- Vue SPA 服务 ----


@router.get("/web", response_class=HTMLResponse)
@router.get("/web/", response_class=HTMLResponse)
async def serve_index():
    """Vue 应用入口。构建产物缺失时给出友好提示。"""
    index = _DIST_DIR / "index.html"
    if not index.exists():
        return HTMLResponse(
            "<h2>前端尚未构建</h2><p>请在 frontend/ 下运行 <code>npm install &amp;&amp; npm run build</code>。</p>",
            status_code=503,
        )
    return index.read_text(encoding="utf-8")


@router.get("/web/assets/{file_path:path}")
async def serve_assets(file_path: str):
    """Vue 构建出的 JS/CSS 静态资源。"""
    target = (_DIST_DIR / "assets" / file_path).resolve()
    # 防目录穿越：必须落在 assets 目录内
    if not str(target).startswith(str((_DIST_DIR / "assets").resolve())) or not target.is_file():
        return Response(status_code=404)
    return FileResponse(target)


# ---- 播放代理 ----


@router.post("/api/playback/audio")
async def playback_audio(request: Request):
    """获取音频播放 URL，返回结构化失败原因。

    Body: { "track": { title, artist, source, source_id, ... }, "user_id": "..." }
    返回: { "url": str|None, "reason": "ok"|"vip_required"|"not_found"|"error" }
      - ok: 拿到可播 URL
      - vip_required: 网易云付费歌曲，未登录/非 VIP 无法取流（前端提示扫码登录）
      - not_found: 该来源无音频直链（B站/YouTube 只能看视频）
      - error: 取流异常
    """
    body = await request.json()
    track = body.get("track", {})
    user_id = _effective_user_id(
        request,
        body.get("user_id", "web_user"),
        bind_in_anonymous_mode=True,
    )
    source = (track.get("source") or "").lower()

    track_obj = _make_track_ns(track)
    cookie = _load_netease_cookie(user_id)
    try:
        url = agent.get_audio_url(track_obj, netease_cookie=cookie)
    except Exception:
        logger.exception("get_audio_url failed for %s", track.get("title", ""))
        return {"url": None, "reason": "error", "asset_id": track.get("asset_id", "")}

    if url:
        # ⚠️ 播放 ≠ 入库：不再 ensure_asset_from_track（那是 bug——每首播过的歌都进库且无标签）。
        # 只回一个逻辑 asset_id（与入库同算法）供前端收听采集 keying，库本身不动。
        logical_id = agent.library_svc.asset_id_for_track(track_obj)
        return {"url": url, "reason": "ok", "asset_id": logical_id or track.get("asset_id", "")}

    # 无 URL：区分原因，便于前端给出可操作提示
    if "netease" in source:
        # 网易云拿不到流，最常见是付费歌曲需登录+VIP
        reason = "vip_required" if not cookie else "not_found"
    elif source in {"bilibili", "youtube"} or "fallback" in source:
        reason = "not_found"  # 视频源无音频直链，应走 MV
    else:
        reason = "not_found"
    return {"url": None, "reason": reason, "asset_id": track.get("asset_id", "")}


@router.post("/api/playback/mv")
async def playback_mv(request: Request):
    """获取 MV 播放 URL。

    Body: { "track": { title, artist, source, source_id, ... }, "user_id": "..." }
    """
    body = await request.json()
    track = body.get("track", {})

    track_obj = _make_track_ns(track)
    url = None
    try:
        url = agent.get_mv_url(track_obj)
    except Exception:
        logger.exception("get_mv_url failed for %s", track.get("title", ""))

    return {"url": url}


# ---- 专辑封面识别 ----


@router.post("/api/identify-album")
async def identify_album(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Form("web_user"),
):
    """上传专辑封面 → 识别专辑/歌手 → 返回可路由进知识链路的 query。

    前端拿到 ``query`` 后，直接用现有 ``/agent/stream`` 发这一句（命中 album_deep_dive 意图），
    于是整条知识链路（消歧→元数据→乐评→档案）原样复用，本端点只负责「图片→文字」。

    返回::
        {
          "recognized": {album, artist, confidence, method, raw_text, note},
          "query": "album\\nX\\nY\\n解读这张专辑" | null,  # null → 前端提示用户输入
          "thumbnail_url": "data:image/jpeg;base64,…",      # 气泡里显示的缩略图（解码失败为空串）
          "user_id": "<effective uid>"
        }
    """
    uid = _effective_user_id(request, user_id)
    # 分块限速读取 + 魔数字节判真实类型（不信可伪造的 Content-Type 头）。
    raw = await _read_capped(file, settings.album_cover_max_bytes)
    if not raw:
        raise HTTPException(status_code=400, detail="空文件")
    mime = _sniff_image_mime(raw)
    if mime is None:
        raise HTTPException(status_code=415, detail="文件不是合法的 PNG/JPEG/WebP 图片")

    try:
        recognition = await recognize_album_cover(raw, mime)
    except Exception:
        logger.exception("封面识别失败")
        recognition = None
    if recognition is None:
        recognition = CoverRecognition()

    query = synthesize_query(recognition)
    try:
        thumbnail = build_thumbnail_data_url(raw)
    except Exception:
        thumbnail = ""

    return {
        "recognized": recognition.to_dict(),
        "query": query,
        "thumbnail_url": thumbnail,
        "user_id": uid,
    }


# ---- Helpers ----


def _make_track_ns(data: dict) -> SimpleNamespace:
    """从 dict 创建 SimpleNamespace，模拟 Asset/ExternalTrack 属性访问。

    注意：前端 SongCard 用 playback_url 字段（如 https://music.163.com/song?id=xxx），
    而后端 get_audio_url 读 source_url 做直接 ID 提取。这里把 playback_url 也映射进
    source_url，让有 ID 的网易云歌曲走秒级直链，而不是每首都触发搜索（易限流）。
    """
    return SimpleNamespace(
        title=data.get("title", ""),
        artist=data.get("artist", ""),
        source=data.get("source", "unknown"),
        external_id=data.get("source_id") or data.get("external_id", ""),
        source_url=data.get("source_url", "") or data.get("playback_url", ""),
        cover_url=data.get("cover_url"),
    )


def _load_netease_cookie(user_id: str) -> str:
    """尝试加载用户的网易云 cookie。

    走 ``netease_auth.load_cookie`` 而非自己读文件——确保与写入路径走同一套
    解密逻辑，避免「写入加密、此处读回明文」的漂移。
    """
    try:
        auth_data = netease_auth.load_cookie(user_id)
        if auth_data:
            return auth_data.get("cookie", "")
    except Exception:
        pass
    return ""
