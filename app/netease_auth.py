"""NetEase Cloud Music authentication via QR code scan.

Uses the ``/api/`` endpoints for QR login flow.  The MUSIC_U cookie obtained
after a successful scan is persisted to ``data/store/netease_auth/`` so the
binding survives app restarts.
"""

from __future__ import annotations

import http.cookiejar
import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://music.163.com/",
    "Content-Type": "application/x-www-form-urlencoded",
    "Cookie": "os=pc; appver=2.10.11; osver=MacOS14;",
}

# 整个扫码流程（unikey → 轮询 → 确认）共用一个 CookieJar，
# 让创建二维码时拿到的匿名会话 cookie（NMTID 等）持续带到确认请求，
# 否则网易云可能因会话不一致而在 803 时不下发 MUSIC_U。
_JAR = http.cookiejar.CookieJar()
_OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_JAR))


def _post(endpoint: str, data: dict) -> dict:
    url = f"https://music.163.com/api/{endpoint}"
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=body, headers=_HEADERS)
    with _OPENER.open(req, timeout=10) as r:
        resp = json.loads(r.read().decode())
    # 从共享 jar 读取所有 cookie（已正确解析，含 MUSIC_U 及其完整长值）
    cookies = {c.name: c.value for c in _JAR}
    resp["_cookies"] = cookies
    return resp


def _full_cookie_string() -> str:
    """把 jar 里登录相关的 cookie 拼成完整字符串，供播放接口使用。"""
    keep = ("MUSIC_U", "__csrf", "NMTID", "MUSIC_A")
    parts = [f"{c.name}={c.value}" for c in _JAR if c.name in keep and c.value]
    return "; ".join(parts)


def _cookie_header(cookie: str) -> str:
    """Normalize a stored cookie value into a usable Cookie header.

    Accepts either a bare MUSIC_U value or a full ``MUSIC_U=...; ...`` string,
    and always prefixes ``os=pc`` which NetEase requires for full-quality /
    VIP-tier responses.
    """
    raw = (cookie or "").strip()
    if not raw:
        return ""
    if "MUSIC_U=" not in raw:
        raw = f"MUSIC_U={raw}"
    if "os=" not in raw:
        raw = f"os=pc; {raw}"
    return raw


def get_qr_key() -> str:
    """Request a QR-code login key.  Returns the *unikey* string."""
    resp = _post("login/qrcode/unikey", {"type": 1})
    return resp["unikey"]


def check_qr_status(unikey: str) -> dict:
    """Poll QR login status.

    Returns ``{"code": int, "cookie": str|None, "nickname": str|None,
    "avatar": str|None, "raw": dict}``.

    * 800 = expired  * 801 = waiting  * 802 = scanned  * 803 = confirmed
    """
    resp = _post("login/qrcode/client/login", {"key": unikey, "type": 1})
    code = resp.get("code", 800)
    result: dict = {
        "code": code,
        "cookie": None,
        "nickname": None,
        "avatar": None,
        "raw": resp,
    }

    # Try to extract MUSIC_U from Set-Cookie headers
    cookies = resp.get("_cookies", {})
    music_u = cookies.get("MUSIC_U", "")

    # Fallback: try JSON body cookie field
    if not music_u:
        raw_cookie = resp.get("cookie", "")
        if isinstance(raw_cookie, str) and raw_cookie:
            for part in raw_cookie.split(";"):
                part = part.strip()
                if part.startswith("MUSIC_U="):
                    music_u = part.split("=", 1)[1]
                    break
            if not music_u:
                music_u = raw_cookie

    if code == 803:
        # 优先用 jar 里的完整 cookie 串（MUSIC_U + __csrf 等），播放接口更稳；
        # 退回到单独的 MUSIC_U 值。
        full = _full_cookie_string()
        result["cookie"] = full if "MUSIC_U=" in full else (music_u or None)
        result["nickname"] = (
            resp.get("nickname")
            or resp.get("account", {}).get("userName")
            or resp.get("profile", {}).get("nickname")
        )
        result["avatar"] = (
            resp.get("avatarUrl")
            or resp.get("profile", {}).get("avatarUrl")
            or resp.get("avatar")
        )
        # 登录成功后立即拉取账号详情，补全昵称/头像并记录 VIP 等级
        if music_u:
            info = fetch_account_info(music_u)
            if info:
                result["nickname"] = info.get("nickname") or result["nickname"]
                result["avatar"] = info.get("avatar") or result["avatar"]
                result["vip_type"] = info.get("vip_type", 0)
                result["vip_label"] = info.get("vip_label", "")
    return result


def fetch_account_info(cookie: str) -> dict | None:
    """拉取当前登录账号的昵称/头像/VIP 等级。

    vipType: 0=非会员，10=普通黑胶 VIP，11=黑胶 SVIP（数值随活动变动，
    >0 即视为会员）。失败返回 None。
    """
    raw = (cookie or "").strip()
    if not raw:
        return None
    headers = {**_HEADERS, "Cookie": _cookie_header(raw)}
    try:
        url = "https://music.163.com/api/nuser/account/get"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
    except Exception:
        logger.debug("NetEase account info fetch failed", exc_info=True)
        return None
    profile = data.get("profile") or {}
    account = data.get("account") or {}
    vip_type = int(account.get("vipType", 0) or 0)
    if vip_type >= 11:
        vip_label = "黑胶 SVIP"
    elif vip_type > 0:
        vip_label = "黑胶 VIP"
    else:
        vip_label = "非会员"
    return {
        "nickname": profile.get("nickname"),
        "avatar": profile.get("avatarUrl"),
        "vip_type": vip_type,
        "vip_label": vip_label,
    }


# ── 歌单导入 ──────────────────────────────────────────────────────────────

def _api_get(url: str, cookie: str = "") -> dict | None:
    """GET 一个网易云 API，带可选登录 cookie。失败返回 None。"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://music.163.com/",
        "Cookie": _cookie_header(cookie) if cookie else "os=pc;",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=12) as r:
            return json.loads(r.read().decode())
    except Exception:
        logger.debug("NetEase API request failed: %s", url, exc_info=True)
        return None


def fetch_playlist_tracks(playlist_id: str, cookie: str = "", limit: int = 200) -> dict:
    """拉取歌单的全部曲目。

    返回 ``{"name": str, "tracks": [{"song_id","title","artist","album",
    "cover","duration"}], "total": int}``。失败时 tracks 为空列表。
    """
    out: dict = {"name": "", "tracks": [], "total": 0}
    detail = _api_get(
        f"https://music.163.com/api/v6/playlist/detail?id={playlist_id}&n=1000",
        cookie,
    )
    if not detail or detail.get("code") != 200:
        return out
    pl = detail.get("playlist") or {}
    out["name"] = pl.get("name", "")
    track_ids = [str(t.get("id")) for t in (pl.get("trackIds") or []) if t.get("id")]
    out["total"] = len(track_ids)
    track_ids = track_ids[:limit]
    if not track_ids:
        return out
    # 批量取歌曲详情（c 参数是 [{"id":x}] 列表）
    ids_param = urllib.parse.quote(json.dumps([{"id": int(i)} for i in track_ids]))
    songs_resp = _api_get(
        f"https://music.163.com/api/v3/song/detail?c={ids_param}",
        cookie,
    )
    songs = (songs_resp or {}).get("songs") or []
    for s in songs:
        artists = "、".join(a.get("name", "") for a in (s.get("ar") or []) if a.get("name"))
        album = s.get("al") or {}
        out["tracks"].append({
            "song_id": str(s.get("id")),
            "title": s.get("name", ""),
            "artist": artists,
            "album": album.get("name", ""),
            "cover": album.get("picUrl", ""),
            "duration": int((s.get("dt") or 0) // 1000) or 180,
        })
    return out


def fetch_user_playlists(cookie: str, uid: str | None = None) -> list[dict]:
    """登录后拉取"我的歌单"列表。返回 ``[{"id","name","cover","count"}]``。"""
    if not cookie:
        return []
    if not uid:
        info = _api_get("https://music.163.com/api/nuser/account/get", cookie)
        uid = str(((info or {}).get("account") or {}).get("id", "")) if info else ""
    if not uid:
        return []
    resp = _api_get(
        f"https://music.163.com/api/user/playlist?uid={uid}&limit=100&offset=0",
        cookie,
    )
    playlists = (resp or {}).get("playlist") or []
    return [{
        "id": str(p.get("id")),
        "name": p.get("name", ""),
        "cover": p.get("coverImgUrl", ""),
        "count": p.get("trackCount", 0),
    } for p in playlists]


# ── Persistence ─────────────────────────────────────────────────────────

_STORE_DIR = Path("data/store/netease_auth")


def _path(user_id: str) -> Path:
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    return _STORE_DIR / f"{user_id}.json"


def save_cookie(
    user_id: str,
    cookie: str,
    nickname: str | None = None,
    avatar: str | None = None,
    vip_type: int = 0,
    vip_label: str = "",
) -> None:
    data = {
        "cookie": cookie,
        "nickname": nickname,
        "avatar": avatar,
        "vip_type": vip_type,
        "vip_label": vip_label,
    }
    _path(user_id).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def load_cookie(user_id: str) -> dict | None:
    p = _path(user_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def clear_cookie(user_id: str) -> None:
    p = _path(user_id)
    if p.exists():
        p.unlink()
