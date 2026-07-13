"""入库 URL 治理（Issue 5）：scheme 校验 + SSRF/私网阻断 + 可选站点白名单。

``/assets/ingest(_full)`` 此前接受任意字符串 URL，不匹配已知站点的会落入 yt-dlp subprocess，
拿到原始用户输入做网络请求。本模块在 API 边界拦截：

- scheme ∈ {http, https}（拒 file://、ftp://、gopher:// 等）；
- host 非空；
- SSRF/私网阻断（**常开**，无视 ALLOW_ANY_URL）：解析 host 全部地址，任一为
  私网/环回/链路本地/保留/组播/未指定 → 拒（覆盖 127.0.0.0/8、10/8、172.16/12、
  192.168/16、169.254/16、::1、fc00::/7、localhost 等）；
- 站点白名单（仅 ``ALLOW_ANY_URL=false`` 时）：host 须后缀命中 ``ingest_allowed_hosts``。

局限：基础 getaddrinfo 校验挡绝大多数 SSRF 探测；DNS rebinding（首次解析公网、二次解析内网）
在本地自用场景不重点防御。解析失败（NXDOMAIN 等）不视为私网——交由后续 fetch 自然失败。
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

_MAX_URL_LENGTH = 2048
_ALLOWED_SCHEMES = {"http", "https"}


class IngestURLError(ValueError):
    """入库 URL 不合法或不被允许。"""


def _host_is_private(host: str) -> bool:
    """host 解析出的任一 IP 为私网/环回/链路本地/保留/组播/未指定 → True。

    字面 IP（如 ``127.0.0.1``）不经 DNS 直接判定；域名经 getaddrinfo 解析。
    解析失败（gaierror）→ False（不视为私网，交由后续 fetch 自然失败，避免误杀公网域名）。
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return True
    return False


def validate_ingest_url(url: str) -> None:
    """校验入库 URL；不合法 raise :class:`IngestURLError`。

    常开校验（无视 ALLOW_ANY_URL）：非空、长度 ≤ 2048、scheme ∈ {http,https}、host 非空、
    非 SSRF/私网。仅 ``settings.allow_any_url=False`` 时额外要求 host 命中白名单。
    """
    from app.config import settings

    if not isinstance(url, str) or not url.strip():
        raise IngestURLError("empty url")
    if len(url) > _MAX_URL_LENGTH:
        raise IngestURLError("url too long")
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise IngestURLError(f"scheme not allowed: {parsed.scheme or '(missing)'}")
    host = (parsed.hostname or "").lower()
    if not host:
        raise IngestURLError("missing host")
    if _host_is_private(host):
        raise IngestURLError(f"internal/private host blocked: {host}")
    if not settings.allow_any_url:
        allowed = [h.lower() for h in settings.ingest_allowed_hosts]
        if not any(host == h or host.endswith("." + h) for h in allowed):
            raise IngestURLError(f"host not in allowlist: {host}")
