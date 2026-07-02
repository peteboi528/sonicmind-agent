"""MusicBrainz 客户端：权威实体消歧 + 结构化元数据（别名/标签/发行）。

免费、无需 API Key（只需标识性的 User-Agent）。MusicBrainz 建议 ≤1 req/s；
本 client 在 run_parallel 里被调用，靠超时 + 异常降级兜住限流——失败返回空，
知识链路据此降级，不报错。search 返回的 score(0-100) 是天然消歧信号，
替代 knowledge.py 里靠正则猜测实体名的做法。

与 lastfm_client 同风格：同步 urllib + 异常返回空，零新依赖。
"""
from __future__ import annotations

import json
import logging
import re
import threading
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_BASE_URL = "https://musicbrainz.org/ws/2"
# MusicBrainz 强制要求带联系方式/项目的 User-Agent，否则可能被拒。
_HEADERS = {"User-Agent": "MusicAgent/1.0 (https://github.com/peteboi528/MusicAgent)"}
# 单次 MB 调用超时。从国内访问 musicbrainz.org 单次常 5–8s，6s 会刚好超时→实体 unresolved
# →档案降级。提到 12s 留余量；快网络随响应即返回，不会真等满。
_TIMEOUT = 12

# 进程级响应缓存：MB 建议 ≤1 req/s，而知识链路对同一实体会调两次——消歧阶段
# (canonicalize_entities) 和元数据阶段 (_metadata_for_entity) 各一次。缓存让第二次
# 命中内存、不再发起被限流的冗余请求，省下一整个网络 round-trip 给慢源腾预算。
_RESPONSE_CACHE: dict[str, dict] = {}
_RESPONSE_CACHE_LOCK = threading.Lock()
_RESPONSE_CACHE_MAX = 128


def _get(path: str, **params: Any) -> dict:
    """统一 GET MusicBrainz REST API(fmt=json)。失败返回 {}。命中进程缓存则跳过网络。"""
    cache_key = path + "?" + urllib.parse.urlencode(sorted({"fmt": "json", **params}.items()))
    with _RESPONSE_CACHE_LOCK:
        if cache_key in _RESPONSE_CACHE:
            return _RESPONSE_CACHE[cache_key]
    query = {"fmt": "json", **params}
    url = f"{_BASE_URL}/{path.lstrip('/')}?{urllib.parse.urlencode(query)}"
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        logger.debug("MusicBrainz API call failed: path=%s params=%s", path, params, exc_info=True)
        return {}
    with _RESPONSE_CACHE_LOCK:
        if len(_RESPONSE_CACHE) >= _RESPONSE_CACHE_MAX:
            _RESPONSE_CACHE.clear()
        _RESPONSE_CACHE[cache_key] = data
    return data


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _first_release_date(raw: Any) -> str:
    # MB 的 first-release-date 在 JSON 里是 {"date": "YYYY-MM-DD", ...} 或纯字符串。
    if isinstance(raw, dict):
        return str(raw.get("date") or "")
    return str(raw or "")


def _norm(value: str) -> str:
    """消歧用的归一化：小写 + 去标点/空白，比对实体名是否实质相同。"""
    return re.sub(r"[^a-z0-9一-鿿]+", "", (value or "").lower())


def _best_by_exact_name(hits: list[dict], query: str, key: str) -> dict | None:
    """精确名优先消歧：先在 hits[key] 与 query 归一化相等的子集里取 score 最高者，
    没有精确命中才回落全局 score 最高。专治裸名查询被模糊匹配带偏（Blonde→Blonde on Blonde）。"""
    if not hits:
        return None
    q = _norm(query)
    exact = [h for h in hits if _norm(h.get(key, "")) == q]
    pool = exact or hits
    return max(pool, key=lambda h: h.get("score", 0))


def _url_relations(raw: Any) -> list[dict[str, str]]:
    """Normalize MusicBrainz URL relations into compact source links.

    MB keeps valuable curated outbound links (BBC reviews/pages, AllMusic,
    Discogs, RateYourMusic, archived Pitchfork reviews, etc.) under
    ``relations``. Search endpoints do not always include them, so lookup
    methods use ``inc=url-rels`` and pass these through to the knowledge layer.
    """
    out: list[dict[str, str]] = []
    for rel in raw or []:
        if not isinstance(rel, dict) or rel.get("target-type") != "url":
            continue
        url = (rel.get("url") or {}).get("resource") or ""
        url = str(url).strip()
        if not url:
            continue
        rel_type = str(rel.get("type") or "").strip()
        out.append({
            "type": rel_type,
            "url": url,
            "ended": "true" if rel.get("ended") else "",
        })
    return out


def _artist_from_raw(a: dict[str, Any], score: int = 0) -> dict[str, Any]:
    aliases = [
        str(al.get("name", "")).strip()
        for al in (a.get("aliases") or [])
        if isinstance(al, dict) and al.get("name")
    ]
    tags = [
        str(tg.get("name", "")).strip()
        for tg in (a.get("tags") or [])
        if isinstance(tg, dict) and tg.get("name")
    ]
    return {
        "mbid": a.get("id", ""),
        "name": a.get("name", ""),
        "score": score or _as_int(a.get("score"), 0),
        "country": a.get("country", ""),
        "type": a.get("type", ""),
        "disambiguation": a.get("disambiguation", ""),
        "aliases": aliases,
        "tags": tags,
        "relations": _url_relations(a.get("relations")),
    }


def _release_group_from_raw(rg: dict[str, Any], score: int = 0) -> dict[str, Any]:
    credit = rg.get("artist-credit") or []
    artists = [
        str(ac.get("name", "")).strip()
        for ac in credit
        if isinstance(ac, dict) and ac.get("name")
    ]
    tags = [
        str(tg.get("name", "")).strip()
        for tg in (rg.get("tags") or [])
        if isinstance(tg, dict) and tg.get("name")
    ]
    return {
        "mbid": rg.get("id", ""),
        "title": rg.get("title", ""),
        "artist": " & ".join(artists),
        "score": score or _as_int(rg.get("score"), 0),
        "date": _first_release_date(rg.get("first-release-date")),
        "type": rg.get("primary-type", ""),
        "tags": tags,
        "relations": _url_relations(rg.get("relations")),
    }



class MusicBrainzClient:
    """同步 MusicBrainz 客户端。所有方法失败时返回空结构，调用方负责降级。"""

    def search_artist(self, name: str, limit: int = 3) -> list[dict]:
        """搜索艺人，返回按 score 降序的候选。

        每项: {mbid, name, score(0-100), country, type, disambiguation, aliases, tags}
        """
        name = (name or "").strip()
        if not name:
            return []
        data = _get("artist/", query=name, limit=limit)
        raw = data.get("artists") or []
        out: list[dict] = []
        for a in raw:
            if not isinstance(a, dict):
                continue
            out.append(_artist_from_raw(a))
        return out

    def resolve_artist(self, name: str) -> dict | None:
        """返回最佳艺人候选（消歧）。无结果返回 None。

        MusicBrainz 的返回顺序不保证按 score 降序，且裸名查询常把模糊匹配排在精确名
        之前（如查 "Blonde" 命中 "Blonde Redhead"）。这里先在「名字/别名精确匹配」的
        子集里取 score 最高者，没有精确命中才回落到全局 score 最高——精确名优先于模糊分。
        """
        hits = self.search_artist(name, limit=5)
        if not hits:
            return None
        return _best_by_exact_name(hits, name, key="name")

    def search_release_group(self, title: str, artist: str = "", limit: int = 3) -> list[dict]:
        """搜索专辑(release-group)。

        每项: {mbid, title, artist, score(0-100), date, type, tags}
        """
        title = (title or "").strip()
        if not title:
            return []
        artist = (artist or "").strip()
        # 用简单空格拼接而非 Lucene 的 release:"x" AND artist:"y" 语法——
        # 后者对引号/转义敏感、中文易出错；空格拼接更鲁棒，消歧交给 score。
        query = f"{title} {artist}".strip() if artist else title
        data = _get("release-group/", query=query, limit=limit)
        raw = data.get("release-groups") or []
        out: list[dict] = []
        for rg in raw:
            if not isinstance(rg, dict):
                continue
            out.append(_release_group_from_raw(rg))
        return out

    def resolve_release_group(self, title: str, artist: str = "") -> dict | None:
        """返回最佳专辑候选（消歧）。无结果返回 None。

        裸标题查询易被模糊匹配带偏（查 "Blonde" 命中 Bob Dylan 的 "Blonde on Blonde"），
        故先在「标题精确匹配」子集里取 score 最高者；若指定了 artist，进一步要求 artist
        也匹配，彻底排除同名异艺人作品。没有精确命中才回落全局 score 最高。
        """
        hits = self.search_release_group(title, artist, limit=5)
        if not hits:
            return None
        if artist:
            artist_exact = [h for h in hits if _norm(h.get("artist", "")) == _norm(artist) or _norm(artist) in _norm(h.get("artist", ""))]
            title_and_artist = [h for h in artist_exact if _norm(h.get("title", "")) == _norm(title)]
            if title_and_artist:
                return max(title_and_artist, key=lambda h: h.get("score", 0))
            if artist_exact:
                return max(artist_exact, key=lambda h: h.get("score", 0))
        return _best_by_exact_name(hits, title, key="title")

    def lookup_artist(self, mbid: str) -> dict | None:
        """Lookup one artist by MBID, including tags, aliases and URL relations."""
        mbid = (mbid or "").strip()
        if not mbid:
            return None
        data = _get(f"artist/{mbid}", inc="url-rels+tags+aliases")
        if not data or not data.get("id"):
            return None
        return _artist_from_raw(data, score=100)

    def lookup_release_group(self, mbid: str) -> dict | None:
        """Lookup one release-group by MBID, including tags, credits and URL relations."""
        mbid = (mbid or "").strip()
        if not mbid:
            return None
        data = _get(f"release-group/{mbid}", inc="url-rels+tags+artist-credits")
        if not data or not data.get("id"):
            return None
        return _release_group_from_raw(data, score=100)
