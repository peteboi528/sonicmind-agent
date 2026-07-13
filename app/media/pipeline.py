from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from app.media.analyzer import DemoAnalyzer, MediaAnalyzer
from app.models import Asset, AssetStatus, Segment, utc_now_iso
from app.storage import JsonStore

logger = logging.getLogger(__name__)


class MediaPipeline:
    """A media pipeline with pluggable analyzers."""

    def __init__(
        self, store: JsonStore, media_root: Path | str = "data/media", analyzer: MediaAnalyzer | None = None
    ) -> None:
        self.store = store
        self.media_root = Path(media_root)
        self.media_root.mkdir(parents=True, exist_ok=True)
        self.analyzer: MediaAnalyzer = analyzer or DemoAnalyzer()

    def ingest_video(self, url: str, force_refresh: bool = False) -> Asset:
        url = normalize_url(url)  # 剥掉 uct2 等跟踪参数
        asset_id = stable_id(url)
        if not force_refresh:
            existing = self.store.read_model("assets", asset_id, Asset)
            if existing:
                return existing  # 命中缓存，直接返回
        if force_refresh:
            self.store.delete_key("assets", asset_id)
            self.store.delete_key("segments", asset_id)
        title = title_from_url(url)
        asset = Asset(
            asset_id=asset_id,
            source_url=url,
            title=title,
            duration_seconds=180,
            local_path=str(self.media_root / asset_id),
        )
        self.store.write_model("assets", asset.asset_id, asset)
        return asset

    def analyze_media(self, asset_id: str, force_refresh: bool = False) -> tuple[Asset, list[Segment]]:
        asset = self.store.read_model("assets", asset_id, Asset)
        if asset is None:
            raise ValueError(f"Unknown asset_id: {asset_id}")

        cached_segments = self.get_segments(asset_id)
        if cached_segments and asset.status == AssetStatus.ANALYZED and not force_refresh:
            return asset, cached_segments

        if force_refresh:
            self.store.delete_key("segments", asset_id)

        media_path = Path(asset.local_path) if asset.local_path else None
        segments = self.analyzer.analyze(asset, media_path)

        all_tags: set[str] = set()
        for seg in segments:
            all_tags.update(seg.audio_tags)
            all_tags.update(seg.visual_tags)
        asset.tags_fingerprint = sorted(all_tags)

        # 诚实化（反幻觉）：DemoAnalyzer 不做真实音频/视觉分析，未识别出的属性保持诚实——
        # genre/mood 标「未分类」（与项目「失败标未分类不猜」惯例一致），tempo/energy 保持 None
        # （下游 score_track 用 or 默认值兜底，不会算术崩溃）。
        # 过去这里用 rng.sample(GENRE_POOL) 随机伪造具体曲风，会污染 taste_profile，已移除。
        if not asset.genre:
            asset.genre = ["未分类"]
        if not asset.mood:
            asset.mood = ["未分类"]

        asset.status = AssetStatus.ANALYZED
        asset.updated_at = utc_now_iso()
        self.store.write_model("assets", asset.asset_id, asset)
        self.store.write_models("segments", asset.asset_id, segments)
        return asset, segments

    def get_asset(self, asset_id: str) -> Asset:
        asset = self.store.read_model("assets", asset_id, Asset)
        if asset is None:
            raise ValueError(f"Unknown asset_id: {asset_id}")
        return asset

    def get_segments(self, asset_id: str) -> list[Segment]:
        return self.store.read_models("segments", asset_id, Segment)


def normalize_url(url: str) -> str:
    """\u89c4\u8303\u5316 URL\uff0c\u5265\u6389\u7eaf\u8ddf\u8e2a\u53c2\u6570\uff0c\u4f7f\u540c\u4e00\u9996\u6b4c\u7684\u4e0d\u540c\u5206\u4eab\u94fe\u63a5\u6620\u5c04\u5230\u540c\u4e00\u4e2a asset_id\u3002

    \u7f51\u6613\u4e91 music.163.com\uff1auct2 / userid / from \u7b49\u5747\u662f\u5206\u4eab token\uff0c\u4e0e\u6b4c\u66f2\u5185\u5bb9\u65e0\u5173\u3002
    \u53ea\u4fdd\u7559 id\uff08\u6b4c\u66f2 ID\uff09\u53c2\u6570\u4f5c\u4e3a\u89c4\u8303\u5f62\u5f0f\u3002
    """
    import urllib.parse as _parse

    _TRACKING_PARAMS = {"uct2", "userid", "from", "utm_source", "utm_medium", "utm_campaign"}
    url = url.strip()
    try:
        parsed = _parse.urlparse(url)
        if "163.com" in parsed.netloc or "163cn.tv" in parsed.netloc:
            song_id = netease_song_id(url)
            if song_id:
                return f"https://music.163.com/song?id={song_id}"
            qs = _parse.parse_qs(parsed.query, keep_blank_values=False)
            # \u53ea\u4fdd\u7559\u6709\u610f\u4e49\u7684\u53c2\u6570
            clean_qs = {k: v for k, v in qs.items() if k not in _TRACKING_PARAMS}
            clean_query = _parse.urlencode({k: v[0] for k, v in clean_qs.items()}, safe="")
            # \u7edf\u4e00\u53bb\u6389 fragment\uff08# \u540e\u7684\u5185\u5bb9\uff09
            normalized = parsed._replace(query=clean_query, fragment="").geturl()
            return normalized
    except Exception:
        logger.debug("URL normalization failed; using original URL", exc_info=True)
    return url


def netease_playlist_id(url: str) -> str | None:
    """从网易云歌单 URL 提取歌单 id。

    支持：
      https://music.163.com/playlist?id=19723756
      https://music.163.com/#/playlist?id=19723756
      https://music.163.com/my/m/music/playlist?id=19723756
    """
    import urllib.parse as _parse

    raw = url.strip()
    if "playlist" not in raw and not raw.isdigit():
        return None
    if raw.isdigit():
        return raw
    parsed = _parse.urlparse(raw)
    candidates = [parsed.query]
    if parsed.fragment:
        candidates.append(_parse.urlparse(parsed.fragment).query)
        if "?" in parsed.fragment:
            candidates.append(parsed.fragment.split("?", 1)[1])
    for qs in candidates:
        params = _parse.parse_qs(qs)
        if params.get("id"):
            return params["id"][0]
    return None


def netease_song_id(url: str) -> str | None:
    """Extract a NetEase Cloud Music song id from web, hash, and mobile URLs."""
    import urllib.parse as _parse

    parsed = _parse.urlparse(url.strip())
    query_candidates = [parsed.query]
    if parsed.fragment:
        fragment = parsed.fragment
        query_candidates.append(_parse.urlparse(fragment).query)
        if "?" in fragment:
            query_candidates.append(fragment.split("?", 1)[1])

    for qs in query_candidates:
        params = _parse.parse_qs(qs)
        if "id" in params and params["id"]:
            return params["id"][0]

    path_tail = parsed.path.rstrip("/").split("/")[-1]
    if path_tail.isdigit():
        return path_tail

    return None


def stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def title_from_url(url: str) -> str:
    if "163.com" in url or "163cn.tv" in url:
        song_id = netease_song_id(url)
        if song_id:
            return f"网易云歌曲 {song_id}"
    tail = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", " ", url.rstrip("/").split("/")[-1]).strip()
    return tail.title() if tail else "CineSonic Demo Asset"
