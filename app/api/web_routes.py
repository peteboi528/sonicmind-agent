"""Web 前端路由：Vue 单页应用服务 + 播放代理。"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from app.api.main import agent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["web"])

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"
_DIST_DIR = _WEB_DIR / "dist"


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
    user_id = body.get("user_id", "web_user")
    source = (track.get("source") or "").lower()

    track_obj = _make_track_ns(track)
    cookie = _load_netease_cookie(user_id)
    try:
        url = agent.get_audio_url(track_obj, netease_cookie=cookie)
    except Exception:
        logger.exception("get_audio_url failed for %s", track.get("title", ""))
        return {"url": None, "reason": "error"}

    if url:
        return {"url": url, "reason": "ok"}

    # 无 URL：区分原因，便于前端给出可操作提示
    if "netease" in source:
        # 网易云拿不到流，最常见是付费歌曲需登录+VIP
        reason = "vip_required" if not cookie else "not_found"
    elif source in {"bilibili", "youtube"} or "fallback" in source:
        reason = "not_found"  # 视频源无音频直链，应走 MV
    else:
        reason = "not_found"
    return {"url": None, "reason": reason}


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


# ---- Helpers ----


def _make_track_ns(data: dict) -> SimpleNamespace:
    """从 dict 创建 SimpleNamespace，模拟 Asset/ExternalTrack 属性访问。"""
    return SimpleNamespace(
        title=data.get("title", ""),
        artist=data.get("artist", ""),
        source=data.get("source", "unknown"),
        external_id=data.get("source_id") or data.get("external_id", ""),
        source_url=data.get("source_url", ""),
        cover_url=data.get("cover_url"),
    )


def _load_netease_cookie(user_id: str) -> str:
    """尝试加载用户的网易云 cookie。"""
    try:
        auth_path = Path(agent.store.root) / "netease_auth" / f"{user_id}.json"
        if auth_path.exists():
            auth_data = json.loads(auth_path.read_text(encoding="utf-8"))
            return auth_data.get("cookie", "")
    except Exception:
        pass
    return ""
