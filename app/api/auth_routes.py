"""网易云认证 + 歌单导入路由（供 Web 前端使用）。

复用 app/netease_auth.py 的已有方法，把 Streamlit 内进程调用暴露成 HTTP 端点：
  GET  /auth/netease/qr/key       获取扫码 unikey
  GET  /auth/netease/qr/status    轮询登录状态（成功时落 cookie）
  GET  /auth/netease/account      读取已绑定账号信息
  POST /auth/netease/unbind       解绑
  POST /playlist/import/netease   导入网易云歌单
  GET  /playlist/netease/list     拉取「我的歌单」

扫码状态机由前端驱动（每 2s 轮询），后端只在 code==803 时持久化 cookie。
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Query, Request

from app import netease_auth
from app.api.main import agent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])


# ---- 网易云扫码登录 ----


@router.get("/auth/netease/qr/key")
def netease_qr_key():
    """获取扫码 unikey + 二维码 PNG（data URI），前端直接 <img> 渲染。"""
    try:
        unikey = netease_auth.get_qr_key()
    except Exception:
        logger.exception("netease get_qr_key failed")
        return {"unikey": "", "qr_img": "", "error": "获取二维码失败，请重试"}
    qr_url = f"https://music.163.com/login?codekey={unikey}"
    return {
        "unikey": unikey,
        "qr_url": qr_url,
        "qr_img": _qr_data_uri(qr_url),
    }


def _qr_data_uri(content: str) -> str:
    """把内容编码成二维码 PNG 的 data URI。qrcode 缺失时返回空串。"""
    try:
        import base64
        import io

        import qrcode

        img = qrcode.make(content)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"
    except Exception:
        logger.debug("qrcode generation failed", exc_info=True)
        return ""


@router.get("/auth/netease/qr/status")
def netease_qr_status(unikey: str = Query(...), user_id: str = Query("web_user")):
    """轮询扫码状态。code: 800过期/801待扫/802已扫待确认/803成功。

    成功时持久化 cookie 到 data/store/netease_auth/{user_id}.json，
    返回脱敏后的账号信息（不回传 cookie 明文给前端）。
    """
    try:
        result = netease_auth.check_qr_status(unikey)
    except Exception:
        logger.exception("netease check_qr_status failed")
        return {"code": 800, "error": "状态查询失败"}

    code = result.get("code", 800)
    payload = {"code": code}

    if code == 803 and result.get("cookie"):
        netease_auth.save_cookie(
            user_id,
            result["cookie"],
            nickname=result.get("nickname"),
            avatar=result.get("avatar"),
            vip_type=result.get("vip_type", 0),
            vip_label=result.get("vip_label", ""),
        )
        payload.update({
            "nickname": result.get("nickname"),
            "avatar": result.get("avatar"),
            "vip_type": result.get("vip_type", 0),
            "vip_label": result.get("vip_label", ""),
        })
    return payload


@router.get("/auth/netease/account")
def netease_account(user_id: str = Query("web_user")):
    """读取已绑定的网易云账号信息（不含 cookie 明文）。"""
    info = netease_auth.load_cookie(user_id)
    if not info:
        return {"bound": False}
    return {
        "bound": True,
        "nickname": info.get("nickname"),
        "avatar": info.get("avatar"),
        "vip_type": info.get("vip_type", 0),
        "vip_label": info.get("vip_label", ""),
    }


@router.post("/auth/netease/unbind")
async def netease_unbind(request: Request):
    """解除网易云绑定。"""
    body = await request.json()
    user_id = body.get("user_id", "web_user")
    netease_auth.clear_cookie(user_id)
    return {"unbound": True}


# ---- 网易云歌单导入 ----


@router.post("/playlist/import/netease")
async def import_netease(request: Request):
    """导入网易云歌单。body: {user_id, playlist_ref, limit?}。

    自动使用用户已绑定的 cookie（私密歌单/完整曲目需要）。
    """
    body = await request.json()
    user_id = body.get("user_id", "web_user")
    playlist_ref = (body.get("playlist_ref") or "").strip()
    limit = int(body.get("limit", 200))
    if not playlist_ref:
        return {"error": "请提供歌单链接或 ID"}

    cookie = ""
    info = netease_auth.load_cookie(user_id)
    if info:
        cookie = info.get("cookie", "")

    try:
        result = agent.import_netease_playlist(
            playlist_ref, cookie=cookie, user_id=user_id, limit=limit,
        )
    except ValueError as e:
        return {"error": str(e)}
    except Exception:
        logger.exception("import_netease_playlist failed")
        return {"error": "导入失败，请检查歌单链接"}

    # 去掉 tracks 详情，前端只需统计
    return {
        "name": result.get("name", ""),
        "total": result.get("total", 0),
        "imported": result.get("imported", 0),
        "skipped": result.get("skipped", 0),
    }


@router.get("/playlist/netease/list")
def netease_playlist_list(user_id: str = Query("web_user")):
    """拉取用户「我的歌单」列表（需先扫码登录）。"""
    info = netease_auth.load_cookie(user_id)
    if not info or not info.get("cookie"):
        return {"playlists": [], "error": "请先扫码登录网易云"}
    try:
        playlists = netease_auth.fetch_user_playlists(info["cookie"])
    except Exception:
        logger.exception("fetch_user_playlists failed")
        return {"playlists": [], "error": "拉取歌单失败"}
    return {"playlists": playlists}
