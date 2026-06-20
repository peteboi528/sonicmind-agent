from __future__ import annotations

import hashlib
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import settings
from app.library import ResourceLibrary
from app.llm.client import build_llm
from app.llm.protocol import LLMError, LLMProvider
from app.llm.structured import extract_json_dict, extract_json_list
from app.media.pipeline import MediaPipeline, netease_song_id
from app.memory import MemoryManager
from app.models import (
    AgentAnswer,
    Asset,
    AssetStatus,
    DailyRecommendation,
    DislikeRequest,
    EnrichResponse,
    ExternalTrack,
    FeedbackRequest,
    MemoryUpdateRequest,
    Playlist,
    RagEvidence,
    RecommendedTrack,
    ResourceTrack,
    SavedAlbum,
    SearchResponse,
    Segment,
    SimilarAssetResult,
    SimilarSegmentResult,
    TasteExperiment,
    TasteExperimentFeedbackRequest,
    TasteExperimentReport,
    TasteExperimentSegment,
    TasteExperimentTrack,
    TasteProfile,
    TrackRef,
    UserMemory,
    utc_now_iso,
)
from app.prompts import (
    AUTO_PLAYLIST_TEMPLATE,
    GENERATE_PLAYLIST_TEMPLATE,
    IDENTIFY_FROM_URL_TEMPLATE,
    LLM_SEARCH_TEMPLATE,
)
from app.recommend.daily import DailyRecommender
from app.recommend.engine import RecommendEngine
from app.retrieval.vector_store import HybridRetriever
from app.similarity import AssetSimilarity
from app.sources import bilibili as bilibili_source
from app.sources import netease as netease_source
from app.sources import web_search as web_search_source
from app.sources import youtube as youtube_source
from app.sources.mock_source import MockSource
from app.sources.netease_source import NeteaseSource
from app.sources.protocol import ExternalSource
from app.storage import JsonStore

logger = logging.getLogger(__name__)


def _graph_unavailable_answer() -> AgentAnswer:
    return AgentAnswer(
        answer="Agent 编排暂时不可用，请稍后重试。",
        evidences=[],
        recommended_tracks=[],
        agent_trace=["[graph_error] LangGraph unavailable; no secondary orchestrator was executed."],
        fallback_reason="langgraph_unavailable",
    )


def _build_source() -> ExternalSource:
    """选择外部源：默认真实网易云源；仅当 EXTERNAL_SOURCE=mock 时才用假目录。

    历史 bug：这里曾写死 MockSource()，导致推荐/歌单补位永远拉硬编码假歌单
    （晴天、Yellow、Let It Be…），真实网易云源形同虚设。现在默认走真实源。
    """
    if settings.external_source == "mock":
        logger.info("ExternalSource = MockSource（EXTERNAL_SOURCE=mock）")
        return MockSource()
    return NeteaseSource()


class AudioVisualAgent:
    def __init__(self, store: JsonStore | None = None) -> None:
        self.store = store or JsonStore(settings.store_root)
        self._assets_cache: list[Asset] | None = None
        self._assets_synced_dirty: bool = True
        self._caching_enabled: bool = False  # 构造期不缓存 list_assets，见该方法注释
        self.media = MediaPipeline(self.store)
        self.memory = MemoryManager(self.store)
        self.similarity = AssetSimilarity(self.store)
        # 注入临时 store（测试/隔离运行）时，资源库也必须同域，不能误写真实用户 SQLite。
        resource_path = Path(self.store.root).parent / "resource_library.sqlite" if store is not None else settings.resource_library_path
        self.library = ResourceLibrary(resource_path)
        self.llm: LLMProvider = build_llm()
        self._llm_default_ref: LLMProvider = self.llm
        self.llm_fast: LLMProvider = (
            self.llm if settings.llm_fast_model == settings.llm_model else build_llm("fast")
        )
        self.llm_strong: LLMProvider = (
            self.llm if settings.llm_strong_model == settings.llm_model else build_llm("strong")
        )
        # P1-G：把 LLM 注入记忆层，启用 LLM 偏好抽取兜底 + 巩固画像（mock 下自动跳过）。
        self.memory.llm = self.llm
        self.source: ExternalSource = _build_source()
        self.engine = RecommendEngine()
        self.daily = DailyRecommender(self.engine, self.source, self.llm)
        self.graph = None
        self.library.sync_assets(self.list_assets())
        try:
            from app.graph.builder import build_agent_graph
            self.graph = build_agent_graph(self)
        except Exception:
            logger.exception("LangGraph wrapper unavailable")
        # 启动时清一次候选池污染（历史 fallback 假候选 + 僵尸 local）。幂等、廉价。
        try:
            self.cleanup_resource_library()
        except Exception:
            logger.debug("启动清理候选池失败，跳过", exc_info=True)
        # 构造完成，开启 list_assets 缓存。此前所有读都不缓存，不会污染。
        self._caching_enabled = True

    def ingest_video(self, url: str, force_refresh: bool = False) -> Asset:
        asset = self.media.ingest_video(url, force_refresh=force_refresh)
        self._invalidate_assets_cache()
        return asset

    def enrich_asset(self, asset_id: str, use_network: bool = False) -> EnrichResponse:
        asset = self.store.read_model("assets", asset_id, Asset)
        if asset is None:
            raise ValueError(f"Unknown asset_id: {asset_id}")

        if not use_network and not settings.enable_online_enrich:
            return EnrichResponse(
                asset=asset,
                enriched=False,
                mode="offline",
                note="Offline-first mode keeps enrich optional. Enable network explicitly to fetch title metadata.",
            )

        before = asset.model_dump(mode="json")
        # 网易云：API 直接给出精确 title + artist，只用 LLM 补 genre/mood
        if "163.com" in asset.source_url or "163cn.tv" in asset.source_url:
            song_id = _netease_song_id(asset.source_url)
            if song_id and self._enrich_from_netease(asset, song_id):
                self.store.write_model("assets", asset.asset_id, asset)
                self._invalidate_assets_cache()
                after = asset.model_dump(mode="json")
                enriched = before != after
                return EnrichResponse(
                    asset=asset, enriched=enriched, mode="online",
                    note="Metadata enrichment completed." if enriched else "No new metadata was identified.",
                )
        video_title = self._fetch_video_title(asset.source_url)
        self._apply_title_artist_hint(asset, video_title)
        self._identify_from_url(asset, video_title)
        after = asset.model_dump(mode="json")
        enriched = before != after
        return EnrichResponse(
            asset=asset,
            enriched=enriched,
            mode="online",
            note="Metadata enrichment completed." if enriched else "No new metadata was identified.",
        )

    def _fetch_video_title(self, url: str) -> str | None:
        if "youtube.com" in url or "youtu.be" in url:
            title = youtube_source.fetch_youtube_title(url)
            if title:
                return title

        if "163.com" in url or "163cn.tv" in url:
            song_id = _netease_song_id(url)
            title = netease_source.fetch_netease_title(url, song_id)
            if title:
                return title

        if "bilibili.com" in url:
            title = bilibili_source.fetch_bilibili_title(url)
            if title:
                return title

        # 通用兜底：yt-dlp（不强依赖 Chrome cookies）
        try:
            import subprocess
            result = subprocess.run(
                ["yt-dlp", "--get-title", "--no-download", "--no-warnings", url],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            logger.debug("yt-dlp title fallback failed for url=%s", url, exc_info=True)
        return None

    def _enrich_from_netease(self, asset: Asset, song_id: str) -> bool:
        """从网易云 API 直接获取 title/artist，不走 LLM 猜测，再用 LLM 补 genre/mood。"""
        try:
            detail = netease_source.fetch_netease_song_detail(song_id)
            if not detail:
                return False
            song = detail.get("raw") or {}
            name = (detail.get("title") or "").strip()
            if not name:
                return False

            # 拼完整歌名：主名 + alias（副标题/英文名）+ tns（翻译名）
            # 网易云 UI 显示格式：晴天（Sunny Day）
            extras: list[str] = []
            for alias in (song.get("alias") or []):
                a = alias.strip()
                if a and a != name:
                    extras.append(a)
            for t in (song.get("tns") or []):
                t = t.strip()
                if t and t != name and t not in extras:
                    extras.append(t)
            full_title = f"{name}（{'、'.join(extras)}）" if extras else name

            artists = [a.strip() for a in (detail.get("artist") or "").split("、") if a.strip()]

            # 直接写入，不经过 LLM（保证准确性）
            asset.title = full_title
            if artists:
                asset.artist = "、".join(artists)
            if detail.get("album"):
                asset.album = detail["album"]
            if detail.get("cover"):
                asset.cover_url = detail["cover"]
            duration_ms = song.get("duration") or song.get("dt")
            if duration_ms:
                asset.duration_seconds = max(1, int(duration_ms) // 1000)
            # 只用 LLM 推断 genre 和 mood（这两个 API 不返回）
            if not asset.genre or not asset.mood:
                artist_str = asset.artist or "未知"
                prompt = (
                    f"歌曲：《{full_title}》，歌手：{artist_str}\n"
                    f"请判断这首歌的风格和情绪，严格按格式回复：\n"
                    f"风格: xxx（如：流行、摇滚、电子、古典、R&B、说唱、爵士、民谣）\n"
                    f"情绪: xxx（如：欢快、治愈、励志、伤感、放松、激昂、浪漫）"
                )
                try:
                    result = self.llm.generate(prompt)
                    for line in result.strip().split("\n"):
                        if "风格" in line and ":" in line:
                            genre = line.split(":", 1)[1].strip()
                            if genre and genre != "未知":
                                asset.genre = [g.strip() for g in genre.replace("、", ",").split(",") if g.strip()]
                        if "情绪" in line and ":" in line:
                            mood = line.split(":", 1)[1].strip()
                            if mood and mood != "未知":
                                asset.mood = [m.strip() for m in mood.replace("、", ",").split(",") if m.strip()]
                except Exception:
                    logger.debug("LLM genre/mood inference failed for song_id=%s", song_id, exc_info=True)
            return True
        except Exception:
            logger.debug("NetEase enrichment failed for song_id=%s", song_id, exc_info=True)
            return False

    def _apply_title_artist_hint(self, asset: Asset, video_title: str | None) -> None:
        """Use deterministic title hints before asking the LLM to guess."""
        if not video_title:
            return
        title = video_title.strip()
        for suffix in [" - 单曲 - 网易云音乐", " - 网易云音乐"]:
            title = title.removesuffix(suffix).strip()
        for separator in (" - ", " — ", " – "):
            if separator not in title:
                continue
            name, artist = [part.strip() for part in title.split(separator, 1)]
            if name and (not asset.title or asset.title.startswith("网易云歌曲") or asset.title == "CineSonic Demo Asset"):
                asset.title = name
            if artist and not asset.artist:
                asset.artist = artist
            return
        if title and (not asset.title or asset.title.startswith("网易云歌曲") or asset.title == "CineSonic Demo Asset"):
            asset.title = title

    def _identify_from_url(self, asset: Asset, video_title: str | None = None) -> None:
        prompt = IDENTIFY_FROM_URL_TEMPLATE(
            url=asset.source_url,
            parsed_title=asset.title,
            video_title=video_title,
        )
        try:
            result = self.llm.generate(prompt)
            lines = result.strip().split("\n")
            for line in lines:
                if "歌名" in line and ":" in line:
                    name = line.split(":", 1)[1].strip().strip("《》\"'")
                    if name and name != "未知":
                        asset.title = name
                if "歌手" in line and ":" in line:
                    artist = line.split(":", 1)[1].strip()
                    if artist and artist != "未知":
                        asset.artist = artist
                if "风格" in line and ":" in line:
                    genre = line.split(":", 1)[1].strip()
                    if genre and genre != "未知":
                        asset.genre = [g.strip() for g in genre.replace("、", ",").split(",") if g.strip()]
                if "情绪" in line and ":" in line:
                    mood = line.split(":", 1)[1].strip()
                    if mood and mood != "未知":
                        asset.mood = [m.strip() for m in mood.replace("、", ",").split(",") if m.strip()]
            self.store.write_model("assets", asset.asset_id, asset)
            self._invalidate_assets_cache()
        except Exception:
            logger.debug("URL identity inference failed for asset_id=%s", asset.asset_id, exc_info=True)

    def analyze_media(self, asset_id: str, force_refresh: bool = False) -> tuple[Asset, list[Segment]]:
        asset, segments = self.media.analyze_media(asset_id, force_refresh=force_refresh)
        self._invalidate_assets_cache()
        self.library.upsert_asset(asset)
        return asset, segments

    _VALID_GENRES = {"流行", "摇滚", "电子", "古典", "R&B", "说唱", "爵士", "民谣", "国风", "金属"}

    # 网易云歌单 tags → 本系统曲风词表的映射（歌单级 tags 是导入时唯一可靠的曲风线索）
    _NETEASE_TAG_TO_GENRE = {
        "R&B/Soul": "R&B", "R&B": "R&B", "Soul": "R&B", "蓝调": "R&B",
        "摇滚": "摇滚", "Rock": "摇滚", "金属": "金属", "Metal": "金属", "朋克": "摇滚",
        "电子": "电子", "Electronic": "电子", "House": "电子", "Techno": "电子", "EDM": "电子",
        "说唱": "说唱", "Rap": "说唱", "Hip-Hop": "说唱", "嘻哈": "说唱",
        "爵士": "爵士", "Jazz": "爵士", "布鲁斯": "爵士",
        "古典": "古典", "Classical": "古典", "纯音乐": "古典",
        "民谣": "民谣", "Folk": "民谣", "乡村": "民谣",
        "流行": "流行", "Pop": "流行",
        "国风": "国风", "古风": "国风", "中国风": "国风",
    }

    def _playlist_tags_to_genres(self, tags: list[str]) -> list[str]:
        """把网易云歌单 tags 映射成本系统曲风（用作整单兜底）。无映射则返回空。"""
        genres: list[str] = []
        for tag in tags:
            g = self._NETEASE_TAG_TO_GENRE.get(tag)
            if g and g not in genres:
                genres.append(g)
        return genres

    def _batch_classify_tracks(self, pairs: list[tuple[str, str]]) -> list[dict[str, list[str]]]:
        """批量让 LLM 判断一组 (歌名, 歌手) 的风格和情绪，一次调用处理多首。

        返回与输入等长的列表，每项 {"genre": [...], "mood": [...]}；失败则该项为空。
        会做一次重试：首次解析后仍为空的项，重新发一个只含这些歌的小批再问一次，
        减少落到「中性默认」兜底的数量（提升 R&B 等英文歌名的分类命中率）。
        """
        if not pairs:
            return []
        out = self._classify_once(pairs)
        # 重试：收集首轮没拿到 genre 的项，单独再问一次
        missing = [i for i, r in enumerate(out) if not r.get("genre")]
        if missing:
            retry_pairs = [pairs[i] for i in missing]
            retried = self._classify_once(retry_pairs)
            for slot, r in zip(missing, retried, strict=False):
                if r.get("genre"):
                    out[slot] = r
        return out

    def _classify_once(self, pairs: list[tuple[str, str]]) -> list[dict[str, list[str]]]:
        lines = "\n".join(f"{i}. 《{t}》- {a or '未知'}" for i, (t, a) in enumerate(pairs))
        prompt = (
            f"判断下面每首歌的风格（genre）和情绪（mood）。\n"
            f"歌手名是判断风格的重要线索：看歌手名是否包含或暗示特定风格。\n\n"
            f"{lines}\n\n"
            f"严格输出 JSON 数组，每项对应一首（按序号），格式：\n"
            f'[{{"genre":"说唱","mood":"激昂"}}]\n\n'
            f"风格可选：流行、摇滚、电子、古典、R&B、说唱、爵士、民谣、国风、金属。\n"
            f"情绪可选：欢快、治愈、励志、伤感、放松、激昂、浪漫、孤独、律动、慵懒、热血、暗黑。\n\n"
            f"判断指南：\n"
            f"- 歌手名含 Ft./feat./× 或多位歌手 → 可能是说唱/R&B 合作曲\n"
            f"- 英文歌名 + 中文歌手 → 可能是 R&B/说唱/独立\n"
            f"- 歌手名含 Gem/Trap/Lil/K/制作人代号 → 倾向说唱\n"
            f"- 歌手名含 keshi/Dean/Crush/Zion.T/SZA/The Weeknd → R&B\n"
            f"- 摇滚/金属/朋克相关关键词 → 摇滚\n"
            f"- DJ/Remix/电音/Beat → 电子\n"
            f"- 不确定的宁可标「流行」也不要瞎猜小众风格\n"
        )
        out: list[dict[str, list[str]]] = [{"genre": [], "mood": []} for _ in pairs]
        try:
            raw = extract_json_list(self.llm.generate(prompt)) or []
            for i, item in enumerate(raw[: len(pairs)]):
                if not isinstance(item, dict):
                    continue
                g = str(item.get("genre", "")).strip()
                m = str(item.get("mood", "")).strip()
                genres = [x.strip() for x in g.replace("、", ",").split(",") if x.strip()]
                # 只保留在合法集合内的风格，过滤 LLM 偶发的自由发挥
                genres = [x for x in genres if x in self._VALID_GENRES]
                out[i] = {
                    "genre": genres,
                    "mood": [x.strip() for x in m.replace("、", ",").split(",") if x.strip()],
                }
        except Exception:
            logger.debug("Track classification failed for %s tracks", len(pairs), exc_info=True)
        return out

    # 确定性兜底词表（与 tag_rules 的可选值保持一致）
    _FALLBACK_GENRES = ["流行", "摇滚", "电子", "古典", "R&B", "说唱", "爵士", "民谣", "国风", "金属"]
    _FALLBACK_MOODS = ["欢快", "治愈", "励志", "伤感", "放松", "激昂", "浪漫", "宁静", "律动", "梦幻", "暗黑", "性感"]

    def _ensure_track_tags(
        self,
        title: str,
        artist: str,
        genre: list[str],
        mood: list[str],
        playlist_genres: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """按可靠性逐层补全 genre/mood：
        1) LLM 分类结果（传入的 genre/mood）——最准；
        2) 关键词规则从歌名+歌手推断（tag_rules extract_genre/mood）；
        3) 歌手名→风格映射表（tag_rules extract_genre_from_artist）——覆盖已知艺人；
        4) 歌单级 tags 映射的曲风（playlist_genres）——整单线索；
        5) 仍为空 → genre 标「未分类」，绝不用 hash 随机或假「流行」污染品味画像。
        """
        from app.graph.tag_rules import extract_genre, extract_genre_from_artist, extract_mood

        text = f"{title} {artist}"
        if not genre:
            genre = extract_genre(text)
        if not mood:
            mood = extract_mood(text)
        # 歌手名映射兜底：比歌单 tags 更精准，覆盖 keshi/Drake/The Weeknd 等知名艺人
        if not genre and artist:
            genre = extract_genre_from_artist(artist)
        # 用歌单整单曲风兜底（网易云 tags 映射结果）
        if not genre and playlist_genres:
            genre = list(playlist_genres)
        # 仍为空 → 如实标「未分类」，不猜
        if not genre:
            genre = ["未分类"]
        if not mood:
            mood = ["放松"]
        return genre, mood

    def import_netease_playlist(
        self,
        playlist_ref: str,
        cookie: str = "",
        user_id: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        """把一个网易云歌单批量导入音乐库。

        playlist_ref 可以是歌单链接或纯 id。逐首转成 Asset 入库，
        title/artist/album/cover/duration 直接用网易云 API 的真实数据。
        返回 {"name","imported","skipped","total","tracks":[...]}。
        """
        from app.media.pipeline import netease_playlist_id
        from app.netease_auth import fetch_playlist_tracks

        pid = netease_playlist_id(playlist_ref)
        if not pid:
            raise ValueError("无法识别歌单链接，请确认是网易云歌单地址或 id。")

        data = fetch_playlist_tracks(pid, cookie=cookie, limit=limit)
        result: dict[str, Any] = {
            "name": data.get("name", ""),
            "imported": 0,
            "skipped": 0,
            "total": data.get("total", 0),
            "tracks": [],
        }
        tracks = data.get("tracks", [])
        # 歌单级 tags 映射成曲风，作为整单兜底（网易云歌曲级无曲风，歌单 tags 是唯一可靠线索）
        playlist_genres = self._playlist_tags_to_genres(data.get("tags", []))
        # 批量让 LLM 判断 genre/mood（每块 8 首：20 首会让 DeepSeek 超时整批失败）
        classifications: list[dict[str, list[str]]] = []
        for start in range(0, len(tracks), 8):
            chunk = tracks[start:start + 8]
            classifications.extend(
                self._batch_classify_tracks([(t.get("title", ""), t.get("artist", "")) for t in chunk])
            )

        existing_ids = {a.asset_id for a in self.list_assets()}
        for idx, t in enumerate(tracks):
            song_id = t.get("song_id")
            if not song_id:
                continue
            song_url = f"https://music.163.com/song?id={song_id}"
            asset = self.media.ingest_video(song_url)
            # 用歌单 API 的真实元数据覆盖占位标题
            asset.title = t.get("title") or asset.title
            if t.get("artist"):
                asset.artist = t["artist"]
            if t.get("album"):
                asset.album = t["album"]
            if t.get("cover"):
                asset.cover_url = t["cover"]
            if t.get("duration"):
                asset.duration_seconds = t["duration"]
            # 补全风格/情绪：LLM → 关键词规则 → 确定性兜底，三层保证永不为空
            cls = classifications[idx] if idx < len(classifications) else {}
            genre, mood = self._ensure_track_tags(
                asset.title, asset.artist or "", cls.get("genre") or [], cls.get("mood") or [],
                playlist_genres=playlist_genres,
            )
            asset.genre = genre
            asset.mood = mood
            # 诚实化：tempo/energy 无真实测量时保持 None（下游 score_track 用默认值兜底），
            # 不再用 rng 随机伪造具体数值（与 pipeline.analyze_media 一致）。
            # genre/mood 已由上方 _ensure_track_tags 基于真实曲名/歌手推断，这里不重复伪造。
            # 关键：标记为已分析，否则推荐/歌单/品味会过滤掉这些歌
            asset.status = AssetStatus.ANALYZED
            asset.updated_at = utc_now_iso()
            self.store.write_model("assets", asset.asset_id, asset)
            self._invalidate_assets_cache()
            if asset.asset_id in existing_ids:
                result["skipped"] += 1
            else:
                result["imported"] += 1
                existing_ids.add(asset.asset_id)
            result["tracks"].append(asset)

        # 导入后刷新品味档案，让推荐立即用上新歌
        if user_id and result["imported"]:
            library = [a for a in self.list_assets() if a.status == "analyzed"]
            self.memory.refresh_taste_profile(user_id, library)
        return result

    def list_assets(self) -> list[Asset]:
        # 进程内缓存：list_assets 是 O(库大小) 的逐文件磁盘读+反序列化，且一次请求内
        # 被多处反复调用（search/summarize_taste/list_resource_tracks/rerank…）。
        # 资产只在 ingest/enrich/analyze/delete/clear 时变动——这些点显式失效缓存。
        # 库大时这一项是超时主因之一，缓存把"每请求 ×N 次全量读"压成一次。
        # 构造期（_caching_enabled=False）不缓存：那时 store 常为空，缓存空快照会污染
        # 后续请求。__init__ 末尾开启缓存，第一个真实请求才填充。
        cached = self._assets_cache
        if cached is not None:
            return list(cached)
        keys = self.store.list_keys("assets")
        assets: list[Asset] = []
        for key in keys:
            asset = self.store.read_model("assets", key, Asset)
            if asset:
                assets.append(asset)
        if self._caching_enabled:
            self._assets_cache = assets
        return list(assets)

    def _invalidate_assets_cache(self) -> None:
        """资产写入/删除/清空后调用，确保下次 list_assets 读到最新，并标记需重新同步到 SQLite。"""
        self._assets_cache = None
        self._assets_synced_dirty = True

    def delete_asset(self, asset_id: str, user_id: str | None = None) -> bool:
        deleted_asset = self.store.delete_key("assets", asset_id)
        deleted_segments = self.store.delete_key("segments", asset_id)
        deleted = deleted_asset or deleted_segments
        if deleted:
            self._invalidate_assets_cache()
            self.memory.remove_asset_references(asset_id, user_id=user_id)
            if user_id:
                library = [a for a in self.list_assets() if a.status == "analyzed"]
                self.memory.refresh_taste_profile(user_id, library)
        return deleted

    def clear_cache(self, preserve_memory: bool = True) -> dict[str, int]:
        cleared = {
            "assets": self.store.clear_collection("assets"),
            "segments": self.store.clear_collection("segments"),
        }
        self._invalidate_assets_cache()
        if not preserve_memory:
            cleared["memory"] = self.store.clear_collection("memory")
        # 专辑详情是纯性能缓存（非用户数据），主动清缓存时一并清掉，确保下次点击重新取最新。
        try:
            from app.sources.netease import clear_album_detail_cache
            cleared["album_detail"] = clear_album_detail_cache()
        except Exception:
            logger.debug("clear_album_detail_cache failed", exc_info=True)
        return cleared

    def cleanup_resource_library(self) -> dict[str, int]:
        """清理候选池污染：删历史 fallback/mock 假候选 + 指向已删 asset 的僵尸 local 行。

        新代码已在入库口拦截 fallback，本方法清存量；可由启动钩子或 /cache 主动触发。
        """
        live_ids = {asset.asset_id for asset in self.list_assets()}
        removed_fallback = self.library.purge_fallback_sources()
        removed_orphan = self.library.purge_orphan_local(live_ids)
        if removed_fallback or removed_orphan:
            logger.info(
                "候选池清理：删除 fallback 假候选 %d 行、僵尸 local %d 行",
                removed_fallback, removed_orphan,
            )
        return {"fallback": removed_fallback, "orphan_local": removed_orphan}

    # --- 推荐功能 ---

    def daily_recommend(
        self,
        user_id: str,
        time_of_day: str | None = None,
        count: int | None = None,
    ) -> DailyRecommendation:
        count = count or settings.daily_rec_count
        memory = self.memory.get_memory(user_id)
        library = [a for a in self.list_assets() if a.status == "analyzed"]
        if not memory.taste_profile:
            memory = self.memory.refresh_taste_profile(user_id, library)

        # ── 构造品味驱动的推荐目标 ──
        # 关键：目标句只含时间/风格/情绪词，不含歌手名。
        # 歌手名通过 taste_summary + library_artists 传给 recommend_for_query 的 LLM 候选生成。
        # 如果目标句含歌手名 → _query_has_entity=True → 走 exact 搜索 → 返回垃圾。
        taste = memory.taste_profile or TasteProfile()
        taste_genres = [g for g, _ in taste.top_genres[:3]]
        taste_moods = [m for m, _ in taste.top_moods[:2]]
        time_hint = time_of_day or get_time_bucket_name()

        goal_parts = []
        if time_hint:
            goal_parts.append(time_hint)
        if taste_genres:
            goal_parts.append(" ".join(taste_genres))
        if taste_moods:
            goal_parts.append(" ".join(taste_moods))
        goal = " ".join(goal_parts) if goal_parts else "推荐好听的音乐"

        return self.recommend_for_query(user_id, goal, top_k=count)

    def find_similar_assets(self, asset_id: str, top_k: int = 5) -> list[SimilarAssetResult]:
        return self.similarity.find_similar_assets(asset_id, top_k)

    def find_similar_segments(self, asset_id: str, segment_id: str, top_k: int = 5) -> list[SimilarSegmentResult]:
        return self.similarity.find_similar_segments(asset_id, segment_id, top_k)

    # --- 搜索 ---

    def search(self, user_id: str, query: str, include_external: bool = True, top_k: int = 20, offset: int = 0) -> SearchResponse:
        memory = self.memory.get_memory(user_id)
        memory_query = self.memory.weighted_query(memory, include_artists=False)
        expanded_query = f"{query} {memory_query}".strip()
        search_goal = _extract_search_query(query)
        classification = self.classify_discover_query(query)
        query_kind = classification.get("kind")
        artist_query = query_kind == "artist"
        resolved_artist_query = classification.get("normalized_query") or search_goal or query
        # 本地搜索：只用核心搜索词（search_goal），不用 memory 扩展词。
        # 之前用 expanded_query.split() 做 any() 匹配，memory 的风格标签如
        # "说唱""R&B""chill" 会匹配库里几乎所有歌，淹没真正相关的结果。
        local_terms = search_goal.lower().split() if search_goal else query.lower().split()
        local_results: list[Asset] = []
        for asset in self.list_assets():
            if artist_query:
                if _artist_query_matches(resolved_artist_query, asset.artist or "", allow_fuzzy=True):
                    local_results.append(asset)
                continue
            searchable = f"{asset.title} {asset.artist or ''} {' '.join(asset.genre)} {' '.join(asset.mood)}".lower()
            if any(term in searchable for term in local_terms):
                local_results.append(asset)

        # 歌手查询不使用语义证据扩充，否则会把“氛围相似但歌手无关”的歌曲混进结果。
        evidences = [] if artist_query else self.retrieve_library_evidence(expanded_query, top_k=min(top_k, 6))
        local_by_id = {asset.asset_id: asset for asset in local_results}
        for evidence in evidences:
            asset_id = str(evidence.metadata.get("asset_id", ""))
            asset = next((item for item in self.list_assets() if item.asset_id == asset_id), None)
            if asset is not None:
                local_by_id.setdefault(asset.asset_id, asset)

        external_results: list[ExternalTrack] = []
        if include_external:
            # 用 expanded_query 搜索拿更广结果，但相关性过滤用核心词 search_goal。
            # offset 用于延续指令翻页取新歌（去重时由调用层传入已展示数）。
            external_results = self.search_web_music(
                resolved_artist_query if artist_query else expanded_query,
                top_k=top_k, relevance_query=search_goal, offset=offset,
            )

        # 抽象词（"痛苦""孤独""emo"）的字面歌曲搜索常归零——没有歌名叫"痛苦"，
        # netease type=1 又只按标题/歌词做相关性过滤。此时回退到歌单搜索：复用
        # search_and_extract 从真人策划歌单里捞相关曲目，让情绪/概念词也能返回结果
        # 而非 0。仅在外部候选不足时触发，不影响正常歌手/歌名搜索。
        if include_external and not artist_query and len(external_results) < 3:
            try:
                from app.search.netease_playlist import search_and_extract
                playlist_hits = search_and_extract(
                    f"{search_goal or query}音乐", max_playlists=3, tracks_per_playlist=top_k,
                )
                existing_keys = {_track_key(t) for t in external_results}
                for t in playlist_hits:
                    if _is_verified_online_track(t) and _track_key(t) not in existing_keys:
                        external_results.append(t)
                        existing_keys.add(_track_key(t))
            except Exception:
                logger.debug("search playlist fallback failed for %s", query, exc_info=True)

        summary = _format_search_summary(
            query=query,
            local=list(local_by_id.values())[:top_k],
            external=external_results[:top_k],
            memory_query="" if artist_query else memory_query,
        )
        return SearchResponse(
            local=list(local_by_id.values())[:top_k],
            external=external_results[:top_k],
            summary=summary,
            evidences=evidences,
            agent_trace=[
                f"query={query}",
                f"memory_query={memory_query or 'none'}",
                f"local_hits={len(local_by_id)}",
                f"external_hits={len(external_results)}",
                f"online_verified={sum(1 for track in external_results if _is_verified_online_track(track))}",
                f"fallback_hits={sum(1 for track in external_results if _is_fallback_track(track))}",
            ],
        )

    def search_web_music(
        self,
        query: str,
        top_k: int = 5,
        relevance_query: str = "",
        include_video_sources: bool = False,
        offset: int = 0,
        variants: list[str] | None = None,
    ) -> list[ExternalTrack]:
        """Agent tool wrapper for explicit online search.

        The default product flow remains offline-first. This method is only
        called when the LangGraph plan needs real platform data.
        每个候选都必须回查到真实曲目元数据；回查失败的候选直接丢弃，
        绝不把搜索词 query 当成歌名返回（这是幻觉的主要来源之一）。

        Args:
            query: 传给搜索 API 的完整查询词（可含 memory 扩展词，获取更广结果）。
            top_k: 目标候选数量。
            relevance_query: 相关性过滤用的核心查询词。为空时默认等于 query。
            include_video_sources: 是否包含 B站/YouTube 视频源。默认 False，
                只返回网易云歌曲。用户明确要 MV/视频时才传 True。
            offset: 网易云搜索翻页偏移。延续指令去重时传"已展示数"，
                跳过已给用户看过的那批最热结果，取更深位次的新歌。
            variants: query_plan 生成的同义/纠错查询。多路召回后统一去重、过滤。
        """
        query_list = _merge_search_queries(query, variants)
        if len(query_list) > 1:
            from app.concurrency import run_parallel

            tasks = [
                (
                    f"search_variant:{idx}:{q}",
                    lambda q=q, idx=idx: self.search_web_music(
                        q,
                        top_k=max(top_k, 3),
                        relevance_query=relevance_query or query,
                        include_video_sources=include_video_sources,
                        offset=offset if idx == 0 else 0,
                        variants=None,
                    ),
                )
                for idx, q in enumerate(query_list)
            ]
            merged: list[ExternalTrack] = []
            for batch in run_parallel(tasks, timeout=8.0, default=[]):
                merged.extend(batch or [])
            selected = _dedupe_tracks(merged)[:top_k]
            for track in selected:
                self.library.upsert_external(track)
            return selected

        tracks: list[ExternalTrack] = []

        # 网易云为主候选源：用多结果搜索拿真实歌曲（之前只取 1 首，导致大量缺口
        # 被 B站/YouTube 的合集视频/SEO 垃圾填补，搜索质量差）。
        try:
            from app.sources.netease import search_netease_many
            for meta in search_netease_many(query, limit=top_k, offset=offset):
                if not meta.get("title"):
                    continue
                tracks.append(ExternalTrack(
                    external_id=meta["song_id"],
                    title=meta["title"],
                    artist=meta.get("artist", ""),
                    album=meta.get("album"),
                    cover_url=meta.get("cover"),
                    source="netease",
                    candidate_kind=_classify_candidate_kind(meta["title"], "netease"),
                    playback_url=f"https://music.163.com/song?id={meta['song_id']}",
                ))
        except Exception:
            logger.debug("NetEase web music search failed for query=%s", query, exc_info=True)

        # B站/YouTube 仅在用户明确要视频内容时才补充，纯歌曲推荐不包含视频源。
        if include_video_sources:
            if len(tracks) < top_k:
                try:
                    bili = self._search_bilibili_detail(query)
                    if bili and bili.get("title"):
                        tracks.append(ExternalTrack(
                            external_id=bili["bvid"],
                            title=bili["title"],
                            artist=bili.get("author", ""),
                            source="bilibili",
                            candidate_kind=_classify_candidate_kind(bili["title"], "bilibili"),
                            playback_url=f"https://player.bilibili.com/player.html?bvid={bili['bvid']}&autoplay=0&high_quality=1&danmaku=0",
                        ))
                except Exception:
                    logger.debug("Bilibili web music search failed for query=%s", query, exc_info=True)

            if len(tracks) < top_k:
                try:
                    video_id = self._search_youtube_video(query)
                    if video_id:
                        url = f"https://www.youtube.com/watch?v={video_id}"
                        title = youtube_source.fetch_youtube_title(url)
                        if title:
                            tracks.append(ExternalTrack(
                                external_id=video_id,
                                title=title,
                                artist="",
                                source="youtube",
                                candidate_kind=_classify_candidate_kind(title, "youtube"),
                                playback_url=f"https://www.youtube.com/embed/{video_id}?autoplay=1&rel=0",
                            ))
                except Exception:
                    logger.debug("YouTube web music search failed for query=%s", query, exc_info=True)

        # 相关性过滤：用 relevance_query（核心词）而非完整 query（含 memory 扩展词）。
        rel_q = relevance_query or query
        tracks = [track for track in tracks if _valid_external_track(track, rel_q)]

        # mock 只作为联网不足时的降级候选，必须带 fallback 标记。
        if len(tracks) < top_k:
            tracks.extend(self._dense_library_fallback(
                query=relevance_query or query,
                existing=tracks,
                limit=top_k - len(tracks),
            ))

        # mock 只作为联网不足时的降级候选，必须带 fallback 标记。
        if len(tracks) < top_k:
            for candidate in self.source.search(query, limit=top_k - len(tracks)):
                fallback = candidate.model_copy(update={"source": f"{candidate.source}-fallback"})
                if _valid_external_track(fallback, rel_q):
                    tracks.append(fallback)

        seen: set[tuple[str, str]] = set()
        unique: list[ExternalTrack] = []
        for track in tracks:
            key = (track.source, track.external_id)
            if key in seen:
                continue
            seen.add(key)
            unique.append(track)
        selected = unique[:top_k]
        for track in selected:
            self.library.upsert_external(track)
        return selected

    async def search_web_music_async(
        self,
        query: str,
        top_k: int = 5,
        relevance_query: str = "",
        include_video_sources: bool = False,
        offset: int = 0,
        variants: list[str] | None = None,
    ) -> list[ExternalTrack]:
        """Native async music-source path used by Tool Runtime."""
        import asyncio

        query_list = _merge_search_queries(query, variants)
        if len(query_list) > 1:
            batches = await asyncio.gather(*(
                self.search_web_music_async(
                    item, top_k=max(top_k, 3), relevance_query=relevance_query or query,
                    include_video_sources=include_video_sources,
                    offset=offset if index == 0 else 0,
                    variants=None,
                )
                for index, item in enumerate(query_list)
            ))
            selected = _dedupe_tracks([track for batch in batches for track in batch])[:top_k]
            await asyncio.gather(*(asyncio.to_thread(self.library.upsert_external, track) for track in selected))
            return selected

        from app.sources.netease import asearch_netease_many

        metadata = await asearch_netease_many(query, limit=top_k, offset=offset)
        tracks = [
            ExternalTrack(
                external_id=item["song_id"],
                title=item["title"],
                artist=item.get("artist", ""),
                album=item.get("album"),
                cover_url=item.get("cover"),
                source="netease",
                candidate_kind=_classify_candidate_kind(item["title"], "netease"),
                playback_url=f"https://music.163.com/song?id={item['song_id']}",
            )
            for item in metadata if item.get("title")
        ]
        if include_video_sources and len(tracks) < top_k:
            video_tracks = await self.search_videos_async(query, top_k=top_k - len(tracks))
            tracks.extend(video_tracks)
        rel_q = relevance_query or query
        tracks = [track for track in tracks if _valid_external_track(track, rel_q)]
        if len(tracks) < top_k:
            tracks.extend(await asyncio.to_thread(
                self._dense_library_fallback, rel_q, tracks, top_k - len(tracks),
            ))
        if len(tracks) < top_k and isinstance(self.source, MockSource):
            for candidate in self.source.search(query, limit=top_k - len(tracks)):
                fallback = candidate.model_copy(update={"source": f"{candidate.source}-fallback"})
                if _valid_external_track(fallback, rel_q):
                    tracks.append(fallback)
        selected = _dedupe_tracks(tracks)[:top_k]
        await asyncio.gather(*(asyncio.to_thread(self.library.upsert_external, track) for track in selected))
        return selected

    def _dense_library_fallback(self, query: str, existing: list[ExternalTrack], limit: int = 5) -> list[ExternalTrack]:
        if limit <= 0:
            return []
        try:
            existing_keys = {_track_key(track) for track in existing}
            hits = self.library.semantic_search(
                query,
                limit=max(limit * 2, limit),
                min_score=settings.dense_recall_min_score,
            )
            if not hits:
                hits = self._lexical_resource_fallback(query, limit=max(limit * 2, limit))
            out: list[ExternalTrack] = []
            for item in hits:
                track = ExternalTrack(
                    external_id=item.source_id or f"library:{item.title}:{item.artist}",
                    title=item.title,
                    artist=item.artist,
                    genre=item.genre,
                    mood=item.mood,
                    playback_url=item.playback_url,
                    source=item.source,
                    candidate_kind="track",
                )
                if _track_key(track) in existing_keys:
                    continue
                out.append(track)
                existing_keys.add(_track_key(track))
                if len(out) >= limit:
                    break
            return out
        except Exception:
            logger.debug("dense library fallback failed for query=%s", query, exc_info=True)
            return []

    def _lexical_resource_fallback(self, query: str, limit: int = 10) -> list[ResourceTrack]:
        """Zero-network fallback over verified resource metadata when embeddings are unavailable."""
        from app.graph.tag_rules import extract_tags

        tags = extract_tags(query)
        wanted_genres = {item.lower() for item in tags["genre"]}
        wanted_moods = {item.lower() for item in tags["mood"]}
        wanted_scenarios = {item.lower() for item in tags["scenario"]}
        terms = {
            item.lower() for item in re.findall(r"[A-Za-z0-9&'-]+|[一-鿿㐀-䶿]{2,}", query or "")
            if item.lower() not in _QUERY_NOISE
        }
        scenario_moods = {
            "深夜": {"放松", "宁静", "孤独", "慵懒", "治愈"},
            "睡眠": {"放松", "宁静", "舒缓"},
            "学习": {"专注", "宁静", "放松"},
            "工作": {"专注", "放松"},
        }
        for scenario in wanted_scenarios:
            wanted_moods.update(item.lower() for item in scenario_moods.get(scenario, set()))

        ranked: list[tuple[float, ResourceTrack]] = []
        for track in self.library.list_tracks(1500, verified_only=True):
            genres = {item.lower() for item in track.genre}
            moods = {item.lower() for item in track.mood}
            searchable = " ".join([track.title, track.artist, *track.genre, *track.mood]).lower()
            score = len(wanted_genres & genres) * 4.0 + len(wanted_moods & moods) * 3.0
            score += sum(1.0 for term in terms if term in searchable)
            if score > 0:
                ranked.append((score, track))
        ranked.sort(key=lambda item: (-item[0], item[1].exposure_count, item[1].title.lower()))
        return [track for _, track in ranked[:limit]]

    def search_videos(self, query: str, top_k: int = 5) -> list[ExternalTrack]:
        """搜索 MV/现场/演唱会视频，B站优先、YouTube 补位。不走网易云。

        用于 video 意图：用户明确要 MV/现场/Live 视频时调用。
        P2-H：B站/YouTube 是独立 IO，用 run_parallel 并发发起再按固定顺序合并，
        降低串行等待延迟；任一源超时/失败安静降级，输出顺序确定（B站在前）。
        """
        from app.concurrency import run_parallel

        def _fetch_bili() -> list[ExternalTrack]:
            out: list[ExternalTrack] = []
            for item in bilibili_source.search_bilibili_many(query, limit=min(top_k, 5)):
                out.append(ExternalTrack(
                    external_id=item["bvid"],
                    title=item["title"],
                    artist=item.get("author", ""),
                    source="bilibili",
                    candidate_kind=_classify_candidate_kind(item["title"], "bilibili"),
                    playback_url=f"https://player.bilibili.com/player.html?bvid={item['bvid']}&autoplay=0&high_quality=1&danmaku=0",
                ))
            return out

        def _fetch_youtube() -> list[ExternalTrack]:
            out: list[ExternalTrack] = []
            for item in youtube_source.search_youtube_many(query, limit=min(top_k, 3)):
                vid = item["video_id"]
                title = item.get("title") or youtube_source.fetch_youtube_title(
                    f"https://www.youtube.com/watch?v={vid}"
                ) or ""
                out.append(ExternalTrack(
                    external_id=vid,
                    title=title,
                    artist="",
                    source="youtube",
                    candidate_kind=_classify_candidate_kind(title, "youtube"),
                    playback_url=f"https://www.youtube.com/embed/{vid}?autoplay=1&rel=0",
                ))
            return out

        # 并发发起两源，结果按固定顺序合并（B站在前、YouTube 补位），保证确定性。
        bili_tracks, yt_tracks = run_parallel(
            [("bilibili", _fetch_bili), ("youtube", _fetch_youtube)], default=[]
        )
        tracks: list[ExternalTrack] = [*(bili_tracks or []), *(yt_tracks or [])]

        # 去重
        seen: set[tuple[str, str]] = set()
        unique: list[ExternalTrack] = []
        for track in tracks:
            key = (track.source, track.external_id)
            if key in seen:
                continue
            seen.add(key)
            unique.append(track)
        return unique[:top_k]

    async def search_videos_async(self, query: str, top_k: int = 5) -> list[ExternalTrack]:
        import asyncio

        bili_items, youtube_items = await asyncio.gather(
            bilibili_source.asearch_bilibili_many(query, limit=min(top_k, 5)),
            youtube_source.asearch_youtube_many(query, limit=min(top_k, 3)),
        )
        tracks = [
            ExternalTrack(
                external_id=item["bvid"], title=item["title"], artist=item.get("author", ""),
                source="bilibili", candidate_kind=_classify_candidate_kind(item["title"], "bilibili"),
                playback_url=f"https://player.bilibili.com/player.html?bvid={item['bvid']}&autoplay=0&high_quality=1&danmaku=0",
            )
            for item in bili_items
        ]
        for item in youtube_items:
            title = item.get("title") or await youtube_source.afetch_youtube_title(item["video_id"])
            tracks.append(ExternalTrack(
                external_id=item["video_id"], title=title, artist="", source="youtube",
                candidate_kind=_classify_candidate_kind(title, "youtube"),
                playback_url=f"https://www.youtube.com/embed/{item['video_id']}?autoplay=1&rel=0",
            ))
        return _dedupe_tracks(tracks)[:top_k]

    def search_artist_info(self, query: str) -> list[dict[str, str]]:
        """用 Tavily/DuckDuckGo 搜索歌手/乐队百科信息。

        用于 artist_info 意图：用户要了解歌手背景时调用。
        返回 [{"title": ..., "content": ..., "url": ...}] 搜索摘要列表。
        """
        return web_search_source.search_web_info(
            query, max_results=5, api_key=settings.tavily_api_key,
        )

    async def search_artist_info_async(self, query: str) -> list[dict[str, str]]:
        return await web_search_source.asearch_web_info(
            query, max_results=5, api_key=settings.tavily_api_key,
        )

    def classify_discover_query(self, query: str) -> dict[str, Any]:
        """Classify Discover input before choosing category, artist, or track UI.

        Artist cards are intentionally conservative: bare names must exactly match an
        artist already present in the user's library; unknown artists need an explicit
        cue such as “歌手 Adele”. This prevents moods and activities from becoming
        fuzzy artist pages while keeping song-title search broad.
        """
        from app.graph.tag_rules import extract_tags

        raw = (query or "").strip()
        normalized = _extract_search_query(raw).strip() or raw
        tags = extract_tags(raw)
        artist_cues = ("歌手", "艺人", "乐队", "组合", "artist", "band")
        explicit_artist = any(cue in raw.lower() for cue in artist_cues)
        normalized_key = _normalize_match_text(normalized)
        artist_catalog: dict[str, set[str]] = {}
        for asset in self.list_assets():
            if not asset.artist:
                continue
            for artist_name in _artist_credit_parts(asset.artist):
                artist_catalog.setdefault(artist_name, set()).update(_artist_alias_keys(artist_name))

        exact_artist = next(
            (name for name, aliases in artist_catalog.items() if normalized_key and normalized_key in aliases),
            "",
        )

        if explicit_artist or exact_artist:
            canonical = exact_artist or normalized
            return {
                "kind": "artist", "normalized_query": canonical,
                "label": "歌手档案", "tags": tags,
                "confidence": 0.98 if exact_artist else 0.92,
                "matched_artist": exact_artist,
                "reason": "explicit_artist" if explicit_artist and not exact_artist else "library_artist_exact",
            }

        category_order = (("scenario", "scene", "场景电台"), ("mood", "mood", "情绪电台"), ("genre", "genre", "曲风探索"))
        for tag_key, browse_category, label in category_order:
            if tags.get(tag_key):
                values = [*tags.get("genre", []), *tags.get("mood", []), *tags.get("scenario", [])]
                browse_value = " ".join(dict.fromkeys(values))
                return {
                    "kind": "category", "normalized_query": normalized,
                    "label": label, "browse_category": browse_category,
                    "browse_value": browse_value or tags[tag_key][0],
                    "tags": tags, "confidence": 0.96, "reason": f"tag:{tag_key}",
                }

        # 拼写纠错只在“没有任何情绪/场景/曲风标签”的实体形输入上启用。
        # 要求高分且第一、第二候选拉开差距，避免把 focus/workout 等活动词误认成歌手。
        if len(normalized_key) >= 6 and re.search(r"[a-z]", normalized_key) and artist_catalog:
            scored: list[tuple[float, str]] = []
            for artist_name, aliases in artist_catalog.items():
                score = max((_string_similarity(normalized_key, alias) for alias in aliases), default=0.0)
                scored.append((score, artist_name))
            scored.sort(key=lambda item: (-item[0], item[1].lower()))
            best_score, best_artist = scored[0]
            second_score = scored[1][0] if len(scored) > 1 else 0.0
            if best_score >= 88.0 and best_score - second_score >= 4.0:
                return {
                    "kind": "artist", "normalized_query": best_artist,
                    "label": "歌手档案", "tags": tags,
                    "confidence": round(best_score / 100.0, 3),
                    "matched_artist": best_artist,
                    "reason": "library_artist_fuzzy",
                }

        return {
            "kind": "track", "normalized_query": normalized,
            "label": "歌曲搜索", "tags": tags, "confidence": 0.55,
            "reason": "default_track_search",
        }

    @staticmethod
    def artist_name_matches(query: str, artist: str) -> bool:
        """Match full artist names, credited collaborators, and safe Latin aliases."""
        return _artist_query_matches(query, artist, allow_fuzzy=True)

    def fetch_track_metadata(
        self,
        asset_id: str | None = None,
        url: str | None = None,
        use_network: bool = True,
    ) -> dict[str, Any]:
        """Agent tool wrapper for metadata fetch/enrich with graceful fallback."""
        if asset_id:
            asset = self.store.read_model("assets", asset_id, Asset)
            if asset is None:
                return {"found": False, "asset_id": asset_id, "error": "unknown asset"}
            if use_network:
                try:
                    enriched = self.enrich_asset(asset_id, use_network=True)
                    asset = enriched.asset
                except Exception as exc:
                    return {
                        "found": _has_reliable_metadata(asset),
                        "asset_id": asset_id,
                        "title": asset.title,
                        "artist": asset.artist,
                        "source_url": asset.source_url,
                        "error": str(exc),
                    }
            found = _has_reliable_metadata(asset)
            return {
                "found": found,
                "asset_id": asset.asset_id,
                "title": asset.title,
                "artist": asset.artist,
                "album": asset.album,
                "genre": asset.genre,
                "mood": asset.mood,
                "source_url": asset.source_url,
            }

        if url:
            title = self._fetch_video_title(url) if use_network else None
            return {
                "found": bool(title and not _generic_metadata_title(title)),
                "url": url,
                "title": title,
                "mode": "online" if use_network else "offline",
            }

        return {"found": False, "error": "asset_id or url is required"}

    def _llm_search(self, query: str, limit: int) -> list[ExternalTrack]:
        prompt = LLM_SEARCH_TEMPLATE(query=query, limit=limit)
        try:
            result = self.llm.generate(prompt)
            raw = extract_json_list(result)
            if not raw:
                return []
            tracks: list[ExternalTrack] = []
            for i, item in enumerate(raw[:limit]):
                if not isinstance(item, dict):
                    continue
                tracks.append(ExternalTrack(
                    external_id=f"llm-search-{i:03d}",
                    title=item.get("title", ""),
                    artist=item.get("artist", ""),
                    genre=[item.get("genre", "")] if item.get("genre") else [],
                    mood=[item.get("mood", "")] if item.get("mood") else [],
                    # source="llm" 是"未核实"标记：这些曲目由 LLM 生成、未经真实回查，
                    # Answer Guard 不会把它们放进面向用户答案的白名单（除非显式标注未核实）。
                    source="llm",
                ))
            return tracks
        except Exception:
            logger.debug("LLM search failed; returning no unverified candidates", exc_info=True)
            return []

    # --- 收听记录 ---

    def record_listen(self, user_id: str, asset_id: str, duration: int, completed: bool, context: str | None = None) -> UserMemory:
        memory = self.memory.record_listen(user_id, asset_id, duration, completed, context)
        # Thompson 在线学习反馈环：听完 → 正反馈(α+1)，秒跳 → 负反馈(β+0.5)。
        asset = self.store.read_model("assets", asset_id, Asset)
        if asset is not None:
            if completed:
                self.library.update_ts_feedback(asset, positive=True, weight=1.0)
            elif duration and asset.duration_seconds and duration < asset.duration_seconds * 0.3:
                self.library.update_ts_feedback(asset, positive=False, weight=0.5)
        return memory

    # --- 品味档案 ---

    # --- 评分 ---

    def rate_asset(self, user_id: str, asset_id: str, score: float) -> UserMemory:
        asset = self.store.read_model("assets", asset_id, Asset)
        if asset is None:
            raise ValueError(f"Unknown asset_id: {asset_id}")
        memory = self.memory.record_rating(user_id, asset, score)
        # 高分 → Thompson 正反馈，低分 → 负反馈。
        if score >= 7.0:
            self.library.update_ts_feedback(asset, positive=True, weight=(score - 6.0) / 4.0)
        elif score <= 3.0:
            self.library.update_ts_feedback(asset, positive=False, weight=(4.0 - score) / 4.0)
        # 评分后立即刷新品味档案
        library = [a for a in self.list_assets() if a.status == "analyzed"]
        memory = self.memory.refresh_taste_profile(user_id, library)
        return memory

    # --- 品味档案 ---

    def get_taste_profile(self, user_id: str) -> TasteProfile:
        memory = self.memory.get_memory(user_id)
        if not memory.taste_profile:
            library = [a for a in self.list_assets() if a.status == "analyzed"]
            memory = self.memory.refresh_taste_profile(user_id, library)
        return memory.taste_profile or TasteProfile()

    # --- 对话 ---

    async def chat_async(
        self,
        user_id: str,
        message: str,
        history: list[dict[str, Any]] | None = None,
        thread_id: str | None = None,
        run_id: str | None = None,
    ) -> AgentAnswer:
        asset_id = self._resolve_asset_context(user_id, message)
        if self.graph is not None:
            try:
                return await self.graph.ainvoke(
                    user_id=user_id, asset_id=asset_id, query=message, history=history, top_k=5,
                    thread_id=thread_id, run_id=run_id,
                )
            except Exception:
                logger.exception("LangGraph invoke failed")
        return _graph_unavailable_answer()

    async def stream_chat_async(
        self,
        user_id: str,
        message: str,
        history: list[dict[str, Any]] | None = None,
        thread_id: str | None = None,
        run_id: str | None = None,
    ):
        asset_id = self._resolve_asset_context(user_id, message)
        if self.graph is not None:
            async for event in self.graph.astream(
                user_id=user_id,
                asset_id=asset_id,
                query=message,
                history=history,
                top_k=5,
                thread_id=thread_id,
                run_id=run_id,
            ):
                yield event
            return
        from app.models import StreamEvent
        answer = _graph_unavailable_answer()
        yield StreamEvent(type="final", content=answer.answer, payload=answer.model_dump(mode="json"))

    def generate_greeting(self, user_id: str) -> str:
        memory = self.memory.get_memory(user_id)
        assets = self.list_assets()
        goal = self.memory.get_active_goal(user_id)
        parts = ["嘿，我先看了一眼你的音乐状态。"]

        if memory.taste_profile and memory.taste_profile.top_genres:
            top_genre = memory.taste_profile.top_genres[0][0]
            parts.append(f"你最近的品味更偏 {top_genre}。")
        elif memory.preferences:
            parts.append(f"我记得你提过：{memory.preferences[-1]}。")

        hour = datetime.now().hour
        if 6 <= hour < 11:
            parts.append("现在适合先找一些轻快但不吵的真实曲目。")
        elif 22 <= hour or hour < 2:
            parts.append("夜深了，我会优先找更松弛、耐听的版本。")

        if goal:
            parts.append(f"上次的目标还在：{goal.goal}")

        if memory.listening_history:
            recent = memory.listening_history[-3:]
            completed = sum(1 for item in recent if item.completed)
            if completed >= 2:
                parts.append("最近你完整听完的歌比较多，我会延续这个方向。")
            elif len(recent) >= 2 and completed == 0:
                parts.append("最近跳过比较多，我会少依赖本地库，多去线上找新候选。")

        if len(assets) < 3:
            parts.append("曲库还不多，我可以先联网找真实候选，或者导入网易云歌单再推荐。")
        else:
            parts.append("我会把真实线上候选放前面，本地库只当作你的口味参考。")

        return " ".join(parts)

    # --- 记忆 ---

    def update_memory(self, request: MemoryUpdateRequest) -> tuple[UserMemory, bool]:
        return self.memory.update_memory(request)

    def record_feedback(self, request: FeedbackRequest) -> UserMemory:
        segments = []
        for key in self.store.list_keys("segments"):
            segments.extend(self.store.read_models("segments", key, Segment))
        target = next((s for s in segments if s.segment_id == request.segment_id), None)
        if target is None:
            raise ValueError(f"Unknown segment_id: {request.segment_id}")
        return self.memory.record_feedback(request.user_id, target, request.accepted)

    def record_dislike(self, request: DislikeRequest) -> UserMemory:
        self.library.add_dislike(request)
        # 负反馈也推给 Thompson：明确不喜欢 → ts_beta 大幅上调，后续探索几乎不再选中。
        from types import SimpleNamespace
        self.library.update_ts_feedback(
            SimpleNamespace(
                title=request.title, artist=request.artist,
                source=request.source, external_id=request.source_id, asset_id=request.source_id,
            ),
            positive=False, weight=3.0,
        )
        memory = self.memory.get_memory(request.user_id)
        key = " - ".join(part for part in [request.title, request.artist] if part) or request.source_id or request.source
        if key and key not in memory.dislikes:
            memory.dislikes.append(key)
            memory.updated_at = utc_now_iso()
            self.store.write_model("memory", request.user_id, memory)
        return memory

    def list_resource_tracks(self, limit: int = 100):
        # sync_assets 把 JSON 资产同步进 SQLite，是 O(库大小) 的写。资产没变时重复同步
        # 纯属浪费（similar_artists 每次拉 2500 就触发一次全量 re-upsert）。只在资产
        # 实际变动后同步一次。
        if self._assets_synced_dirty:
            self.library.sync_assets(self.list_assets())
            self._assets_synced_dirty = False
        return self.library.list_tracks(limit)

    def generate_music_journey(self, user_id: str, instruction: str) -> dict[str, Any]:
        from app.concurrency import run_parallel
        from app.search.netease_playlist import search_and_extract

        memory = self.memory.get_memory(user_id)
        phases = _journey_phases(instruction, memory.taste_profile)
        out = {"user_id": user_id, "instruction": instruction, "phases": []}
        per_phase = 4
        recent = set(memory.journey_history[-160:])
        rotation = len(memory.journey_history) // max(1, per_phase)

        # 阶段互相独立，并行从高可信真人歌单召回。不能把完整旅程句子直接作为歌名
        # 搜索条件，否则“清晨/深夜/旅程”等场景词会让所有正式歌曲被相关性过滤掉。
        tasks = [
            (
                f"journey:{phase['name']}",
                lambda phase=phase: search_and_extract(
                    phase["queries"][rotation % len(phase["queries"])],
                    max_playlists=4,
                    tracks_per_playlist=per_phase * 4,
                ),
            )
            for phase in phases
        ]
        batches = run_parallel(tasks, timeout=15.0, default=[])
        seen: set[str] = set()
        local_fallback = [
            track for track in self.list_assets()
            if _is_recommendation_quality_track(track)
            and not self.library.is_disliked(user_id, track)
        ]
        journey_tracks: list[Asset | ExternalTrack] = []

        for phase, batch in zip(phases, batches, strict=False):
            pool: list[Asset | ExternalTrack] = []
            for track in _dedupe_tracks([*(batch or []), *local_fallback]):
                key = _track_key(track)
                if key in seen or key in recent or self.library.is_disliked(user_id, track):
                    continue
                if not _is_recommendation_quality_track(track):
                    continue
                pool.append(track)
            ranked = self._rerank_tracks(
                user_id, phase["query"], pool, top_k=max(per_phase * 2, per_phase),
            )
            candidates = [track for track, _ in ranked[:per_phase]]
            # 历史排除导致候选不足时，允许旧歌回到尾部，但仍按本阶段目标重新排序。
            if len(candidates) < per_phase:
                refill_pool = [
                    track for track in _dedupe_tracks([*(batch or []), *local_fallback])
                    if _track_key(track) not in seen
                    and _is_recommendation_quality_track(track)
                    and not self.library.is_disliked(user_id, track)
                ]
                refill = self._rerank_tracks(user_id, phase["query"], refill_pool, top_k=per_phase)
                for track, _ in refill:
                    if _track_key(track) not in {_track_key(item) for item in candidates}:
                        candidates.append(track)
                    if len(candidates) >= per_phase:
                        break
            for track in candidates:
                seen.add(_track_key(track))
            journey_tracks.extend(candidates)
            self.library.record_exposure(candidates)
            serialized_tracks: list[dict[str, Any]] = []
            for track in candidates:
                if isinstance(track, ExternalTrack):
                    serialized_tracks.append(track.model_dump(mode="json"))
                else:
                    serialized_tracks.append(ExternalTrack(
                        external_id=track.external_id or track.asset_id,
                        title=track.title,
                        artist=track.artist or "未知",
                        album=track.album,
                        genre=track.genre,
                        mood=track.mood,
                        tempo_bpm=track.tempo_bpm,
                        energy_level=track.energy_level,
                        cover_url=track.cover_url,
                        playback_url=track.source_url,
                        source="local",
                    ).model_dump(mode="json"))
            out["phases"].append({
                "name": phase["name"],
                "goal": phase["goal"],
                "transition": phase["transition"],
                "energy": phase["energy"],
                "tracks": serialized_tracks,
            })
        self._record_journey_history(user_id, journey_tracks)
        return out

    def _record_journey_history(self, user_id: str, tracks: list[Asset | ExternalTrack]) -> None:
        keys = [_track_key(track) for track in tracks]
        if not keys:
            return
        with self.store.lock("memory", user_id):
            memory = self.memory.get_memory(user_id)
            memory.journey_history = [*memory.journey_history, *keys][-240:]
            memory.updated_at = utc_now_iso()
            self.store.write_model("memory", user_id, memory)

    # --- 播放 ---

    def get_playback_url(self, track: Asset | ExternalTrack, netease_cookie: str = "") -> str | None:
        if isinstance(track, Asset) and track.source_url:
            video_id = self._extract_youtube_id(track.source_url)
            if video_id:
                return f"https://www.youtube.com/embed/{video_id}?autoplay=1&rel=0"
            bilibili = self._extract_bilibili_id(track.source_url)
            if bilibili:
                param, val = bilibili
                return f"https://player.bilibili.com/player.html?{param}={val}&autoplay=0&high_quality=1&danmaku=0"
            netease_id = _netease_song_id(track.source_url)
            if netease_id:
                audio = self._get_netease_audio_url(netease_id, netease_cookie)
                if audio:
                    return audio
            # NetEase source but no MP3 → try YouTube search as fallback
            video_id = self._search_youtube_video(
                f"{track.title} {track.artist or ''} official"
            )
            if video_id:
                return f"https://www.youtube.com/embed/{video_id}?autoplay=1&rel=0"
            return track.source_url
        if isinstance(track, ExternalTrack):
            if track.playback_url and "listType=search" not in track.playback_url:
                return track.playback_url
            # Try NetEase MP3 first
            netease_id = self._search_netease(f"{track.title} {track.artist}")
            if netease_id:
                audio = self._get_netease_audio_url(netease_id, netease_cookie)
                if audio:
                    return audio
            # NetEase failed → YouTube fallback
            video_id = self._search_youtube_video(f"{track.title} {track.artist}")
            if video_id:
                return f"https://www.youtube.com/embed/{video_id}?autoplay=1&rel=0"
            return None
        return None

    def get_external_url(self, track: Asset | ExternalTrack) -> str | None:
        import urllib.parse
        if isinstance(track, Asset) and track.source_url:
            return track.source_url
        if isinstance(track, ExternalTrack):
            query = urllib.parse.quote_plus(f"{track.title} {track.artist}")
            return f"https://music.163.com/#/search/m/?s={query}&type=1"
        return None

    # --- 两种播放模式：只听歌（音频） / 看 MV（视频） ---

    def get_audio_url(self, track: Asset | ExternalTrack, netease_cookie: str = "") -> str | None:
        """只听歌：只返回纯音频直链（网易云 MP3），不回退到 YouTube，避开机器人验证墙。

        用鸭子类型读属性，兼容 Asset / ExternalTrack / Web 前端传来的 SimpleNamespace
        （历史 bug：原来用 isinstance 严格判类型，前端的 SimpleNamespace 两者都不匹配，
        导致 VIP 登录后仍永远返回 None，看似"无法播放"）。
        """
        source_url = getattr(track, "source_url", "") or ""
        title = getattr(track, "title", "") or ""
        artist = getattr(track, "artist", "") or ""

        # 1) 来源链接里能直接提取网易云 song id 的，直接取流
        if source_url:
            netease_id = _netease_song_id(source_url)
            if netease_id:
                return self._get_netease_audio_url(netease_id, netease_cookie)
        # 2) 否则按 标题+歌手 搜网易云拿音频
        if title:
            netease_id = self._search_netease(f"{title} {artist}".strip())
            if netease_id:
                return self._get_netease_audio_url(netease_id, netease_cookie)
        return None

    def get_mv_url(self, track: Asset | ExternalTrack) -> str | None:
        """看 MV：B 站优先（华语命中率高、嵌入不弹机器人验证），YouTube 仅作兜底。"""
        # 1) 已有来源链接里能直接提取 ID 的，直接用
        if isinstance(track, Asset) and track.source_url:
            bilibili = self._extract_bilibili_id(track.source_url)
            if bilibili:
                param, val = bilibili
                return f"https://player.bilibili.com/player.html?{param}={val}&autoplay=0&high_quality=1&danmaku=0"
            video_id = self._extract_youtube_id(track.source_url)
            if video_id:
                return f"https://www.youtube-nocookie.com/embed/{video_id}?autoplay=1&rel=0"

        title = track.title
        artist = getattr(track, "artist", "") or ""
        # 2) 主路径：搜 B 站
        bvid = self._search_bilibili_video(f"{title} {artist} MV".strip())
        if bvid:
            return f"https://player.bilibili.com/player.html?bvid={bvid}&autoplay=0&high_quality=1&danmaku=0"
        # 3) 兜底：搜 YouTube（nocookie 域）
        video_id = self._search_youtube_video(f"{title} {artist} MV official".strip())
        if video_id:
            return f"https://www.youtube-nocookie.com/embed/{video_id}?autoplay=1&rel=0"
        return None

    def _extract_youtube_id(self, url: str) -> str | None:
        return youtube_source.extract_youtube_id(url)

    def _extract_bilibili_id(self, url: str) -> tuple[str, str] | None:
        return bilibili_source.extract_bilibili_id(url)

    def _search_youtube_video(self, query: str) -> str | None:
        return youtube_source.search_youtube_video(query)

    def _search_bilibili_video(self, query: str) -> str | None:
        """搜 B 站视频，返回 bvid。华语 MV 命中率高，嵌入不弹机器人验证。"""
        return bilibili_source.search_bilibili_video(query)

    def _search_netease(self, query: str) -> str | None:
        return netease_source.search_netease(query)

    def _search_netease_detail(self, query: str) -> dict[str, Any] | None:
        """搜网易云并回查真实曲目元数据。

        返回 {"song_id","title","artist","album","cover"}，
        拿不到真实歌名则返回 None（绝不用 query 当歌名兜底）。
        """
        return netease_source.search_netease_detail(query)

    def _search_bilibili_detail(self, query: str) -> dict[str, Any] | None:
        """搜 B 站并回查真实视频标题/作者。

        返回 {"bvid","title","author"}，拿不到真实标题则返回 None。
        """
        return bilibili_source.search_bilibili_detail(query)

    def _get_netease_audio_url(self, song_id: str, cookie: str = "") -> str | None:
        return netease_source.get_netease_audio_url(song_id, cookie)

    # --- 歌单 ---

    def generate_playlist(
        self,
        user_id: str,
        instruction: str,
        seed_tracks: list[Asset | ExternalTrack] | None = None,
        target_count: int | None = None,
    ) -> Playlist:
        import hashlib
        target_count = target_count or _infer_playlist_count(instruction) or 12
        target_count = max(1, min(target_count, 100))
        seed_tracks = seed_tracks or []
        library = self.list_assets()
        candidates = self._playlist_candidates(instruction, library, seed_tracks, target_count)
        lib_desc = "\n".join([f"- {a.asset_id}: {a.title} - {a.artist or '?'} ({', '.join(a.genre)}, {', '.join(a.mood)}, energy={a.energy_level})" for a in library[:120]])
        candidate_desc = "\n".join(
            f"- {track.title} - {getattr(track, 'artist', '') or '?'} ({getattr(track, 'source', 'local')})"
            for track in candidates[:120]
        )
        # 用户画像：品味摘要 + 排除规则，让 LLM 做个性化选曲
        memory = self.memory.get_memory(user_id)
        explicit_artist = self._query_has_entity(_extract_search_query(instruction))
        taste_summary = self.summarize_taste(user_id, include_artists=explicit_artist) if memory.taste_profile else ""
        exclusion_rules = memory.exclusion_rules or None

        prompt = GENERATE_PLAYLIST_TEMPLATE(
            instruction=instruction,
            library_size=len(library),
            lib_desc=lib_desc,
            target_count=target_count,
            candidate_desc=candidate_desc,
            taste_summary=taste_summary,
            exclusion_rules=exclusion_rules,
        )
        try:
            result = self.llm.generate(prompt)
            data = extract_json_dict(result)
        except LLMError:
            logger.debug("Playlist generation LLM call failed; using fallback", exc_info=True)
            data = None
        if not data:
            return self._fallback_playlist(user_id, instruction, library, target_count, candidates)
        asset_map = {a.asset_id: a for a in library}
        candidate_map = {
            _track_key(track): track
            for track in candidates
        }
        tracks: list[Asset | ExternalTrack] = []
        for item in data.get("tracks", []):
            aid = item.get("asset_id")
            if aid and aid in asset_map:
                tracks.append(asset_map[aid])
            elif _track_key(item) in candidate_map:
                tracks.append(candidate_map[_track_key(item)])
            # LLM 输出但不在本地库/候选池中的曲目不进入歌单。
            # 这些曲目未经回查，不能被后续 Answer Guard 当作白名单证据。
            else:
                continue
        tracks = _dedupe_tracks(tracks)
        allow_variants = _query_requests_variant_content(instruction)
        tracks = [
            track for track in tracks
            if _is_recommendation_quality_track(track, allow_variants=allow_variants)
        ]
        clean_candidates = [
            track for track in candidates
            if _is_recommendation_quality_track(track, allow_variants=allow_variants)
        ]
        tracks = _fill_tracks(tracks, clean_candidates, target_count)

        playlist = Playlist(
            playlist_id=hashlib.sha1(f"{user_id}-{instruction}".encode()).hexdigest()[:8],
            user_id=user_id, name=data.get("name", instruction),
            description=data.get("description", ""), tracks=tracks[:target_count], generated_by="llm",
        )
        self.save_playlist(user_id, playlist)
        return playlist

    def auto_playlists(self, user_id: str) -> list[Playlist]:
        import hashlib
        library = self.list_assets()
        if not library:
            return []
        lib_desc = "\n".join([f"- {a.asset_id}: {a.title} - {a.artist or '?'} ({', '.join(a.genre)}, {', '.join(a.mood)})" for a in library])

        prompt = AUTO_PLAYLIST_TEMPLATE(library_size=len(library), lib_desc=lib_desc)
        try:
            result = self.llm.generate(prompt)
            raw = extract_json_list(result)
        except LLMError:
            logger.debug("Auto playlist LLM call failed; using fallback", exc_info=True)
            raw = None
        if not raw:
            return self._fallback_auto_playlists(user_id, library)
        asset_map = {a.asset_id: a for a in library}
        playlists: list[Playlist] = []
        for item in raw:
            tracks = [asset_map[tid] for tid in item.get("track_ids", []) if tid in asset_map]
            pl = Playlist(
                playlist_id=hashlib.sha1(f"{user_id}-{item.get('name','')}".encode()).hexdigest()[:8],
                user_id=user_id, name=item.get("name", ""),
                description=item.get("description", ""), tracks=tracks, generated_by="auto",
            )
            self.save_playlist(user_id, pl)
            playlists.append(pl)
        return playlists

    def save_playlist(self, user_id: str, playlist: Playlist) -> None:
        key = f"{user_id}_{playlist.playlist_id}"
        self.store.write_model("playlists", key, playlist)

    def list_playlists(self, user_id: str) -> list[Playlist]:
        keys = self.store.list_keys("playlists")
        playlists: list[Playlist] = []
        for key in keys:
            if key.startswith(f"{user_id}_"):
                # 单个歌单文件可能是旧 schema 写的（字段已变更），解析失败时
                # 跳过它而不是让整个列表 500——否则前端只会看到「加载失败」。
                try:
                    pl = self.store.read_model("playlists", key, Playlist)
                except Exception:
                    logger.warning("Skipping unreadable playlist %s (stale schema?)", key, exc_info=True)
                    continue
                if pl:
                    playlists.append(pl)
        return playlists

    def delete_playlist(self, user_id: str, playlist_id: str) -> bool:
        return self.store.delete_key("playlists", f"{user_id}_{playlist_id}")

    # ── 收藏专辑（与歌单同构：collection=saved_albums，key=f"{user_id}_{album_id}"） ──

    def save_album(self, user_id: str, album: SavedAlbum) -> SavedAlbum:
        self.store.write_model("saved_albums", f"{user_id}_{album.album_id}", album)
        return album

    def list_saved_albums(self, user_id: str) -> list[SavedAlbum]:
        albums: list[SavedAlbum] = []
        for key in self.store.list_keys("saved_albums"):
            if not key.startswith(f"{user_id}_"):
                continue
            try:
                a = self.store.read_model("saved_albums", key, SavedAlbum)
            except Exception:
                logger.warning("Skipping unreadable saved album %s (stale schema?)", key, exc_info=True)
                continue
            if a:
                albums.append(a)
        return albums

    def delete_saved_album(self, user_id: str, album_id: str) -> bool:
        return self.store.delete_key("saved_albums", f"{user_id}_{album_id}")

    def is_album_saved(self, user_id: str, album_id: str) -> bool:
        return self.store.read_model("saved_albums", f"{user_id}_{album_id}", SavedAlbum) is not None

    def _playlist_candidates(
        self,
        instruction: str,
        library: list[Asset],
        seed_tracks: list[Asset | ExternalTrack],
        target_count: int,
    ) -> list[Asset | ExternalTrack]:
        search_terms = _playlist_search_terms(instruction)
        # 从 instruction 中提取核心搜索词，用于相关性过滤（不含扩展的风格词）
        relevance_core = _extract_search_query(instruction) or instruction
        allow_variants = _query_requests_variant_content(instruction)
        clean_seed_tracks = [
            track for track in seed_tracks
            if _is_recommendation_quality_track(track, allow_variants=allow_variants)
        ]
        external: list[ExternalTrack] = []

        # 场景词不是歌名。先从高可信真人歌单抽曲目，避免“跑步/节奏”直接搜歌时
        # Type Beat、伴奏和关键词标题反而获得最高相关性。
        if _is_scenario_playlist_instruction(instruction):
            try:
                from app.search.netease_playlist import search_and_extract

                curated = search_and_extract(
                    _curated_playlist_query(instruction),
                    max_playlists=3,
                    tracks_per_playlist=max(target_count, 12),
                )
                external.extend(
                    track for track in curated
                    if _is_recommendation_quality_track(track, allow_variants=allow_variants)
                )
            except Exception:
                logger.debug("curated playlist recall failed for %s", instruction, exc_info=True)

        for online_query in _playlist_online_queries(search_terms):
            if len(_dedupe_tracks([*clean_seed_tracks, *external])) >= target_count:
                break
            # 关键修复：用核心词做相关性过滤，扩展词只用于引导搜索API
            batch = self.search_web_music(
                online_query, top_k=min(max(target_count, 8), 25),
                relevance_query=relevance_core,
            )
            external.extend(
                track for track in batch
                if _is_recommendation_quality_track(track, allow_variants=allow_variants)
            )

        if len(_dedupe_tracks([*clean_seed_tracks, *external])) < target_count:
            source_tracks = self.source.get_recommendations(
                    seed_genres=["流行", "民谣", "R&B", "说唱", "电子"],
                    seed_moods=["放松", "治愈", "浪漫", "伤感"],
                    limit=max(target_count * 2, 40),
                )
            external.extend(
                track for track in source_tracks
                if _is_recommendation_quality_track(track, allow_variants=allow_variants)
            )

        library_ranked = sorted(
            [
                track for track in library
                if _is_recommendation_quality_track(track, allow_variants=allow_variants)
            ],
            key=lambda asset: _playlist_match_score(asset, search_terms),
            reverse=True,
        )
        ordered: list[Asset | ExternalTrack] = [*clean_seed_tracks, *external, *library_ranked]
        return _dedupe_tracks(ordered)

    def _fallback_playlist(
        self,
        user_id: str,
        instruction: str,
        library: list[Asset],
        target_count: int | None = None,
        candidates: list[Asset | ExternalTrack] | None = None,
    ) -> Playlist:
        target_count = target_count or _infer_playlist_count(instruction) or 12
        keywords = instruction.lower().split()
        matched = [
            asset
            for asset in library
            if any(
                term in f"{asset.title} {asset.artist or ''} {' '.join(asset.genre)} {' '.join(asset.mood)}".lower()
                for term in keywords
            )
        ]
        if not matched:
            matched = sorted(
                library,
                key=lambda asset: (asset.energy_level or 0.0, asset.updated_at),
                reverse=True,
            )
        allow_variants = _query_requests_variant_content(instruction)
        matched = [
            track for track in matched
            if _is_recommendation_quality_track(track, allow_variants=allow_variants)
        ]
        clean_candidates = [
            track for track in (candidates or [])
            if _is_recommendation_quality_track(track, allow_variants=allow_variants)
        ]
        tracks = _fill_tracks(matched, clean_candidates, target_count)
        playlist = Playlist(
            playlist_id=hashlib.sha1(f"{user_id}-{instruction}".encode()).hexdigest()[:8],
            user_id=user_id,
            name=instruction or "Agent 歌单",
            description="离线回退歌单：根据你的音乐库和指令自动整理。",
            tracks=tracks[:target_count],
            generated_by="fallback",
        )
        self.save_playlist(user_id, playlist)
        return playlist

    def _fallback_auto_playlists(self, user_id: str, library: list[Asset]) -> list[Playlist]:
        grouped: dict[str, list[Asset]] = {}
        for asset in library:
            bucket = asset.genre[0] if asset.genre else "未分类"
            grouped.setdefault(bucket, []).append(asset)
        playlists: list[Playlist] = []
        for genre, tracks in list(grouped.items())[:4]:
            playlist = Playlist(
                playlist_id=hashlib.sha1(f"{user_id}-{genre}".encode()).hexdigest()[:8],
                user_id=user_id,
                name=f"{genre}精选",
                description=f"按 {genre} 风格自动整理的离线歌单。",
                tracks=tracks[:10],
                generated_by="fallback-auto",
            )
            self.save_playlist(user_id, playlist)
            playlists.append(playlist)
        return playlists

    # --- RAG（保留兼容） ---

    def retrieve_evidence(self, asset_id: str, query: str, top_k: int = 5) -> list[RagEvidence]:
        segments = self._require_segments(asset_id)
        evidences = HybridRetriever(segments).search(query=query, top_k=top_k)
        asset = self.store.read_model("assets", asset_id, Asset)
        title = asset.title if asset else asset_id
        for evidence in evidences:
            evidence.metadata["asset_id"] = asset_id
            evidence.metadata["asset_title"] = title
        return evidences

    def retrieve_library_evidence(self, query: str, top_k: int = 5) -> list[RagEvidence]:
        ranked: list[RagEvidence] = []
        for asset in self.list_assets():
            if asset.status != "analyzed":
                continue
            # 全库搜索必须是只读操作。过去这里调用 retrieve_evidence，后者会在
            # segments 缺失时自动 analyze_media，导致一次普通歌曲/歌手搜索悄悄
            # 改写曲库指纹和 updated_at。未分析片段只是不参与 RAG，不应在查询时补写。
            segments = self.media.get_segments(asset.asset_id)
            if not segments:
                continue
            evidences = HybridRetriever(segments).search(query=query, top_k=min(3, top_k))
            for evidence in evidences:
                evidence.metadata["asset_id"] = asset.asset_id
                evidence.metadata["asset_title"] = asset.title
            ranked.extend(evidences)
        ranked.sort(key=lambda evidence: evidence.similarity, reverse=True)
        return ranked[:top_k]

    def recommend_with_memory(self, asset_id: str, user_id: str, goal: str, top_k: int = 3) -> AgentAnswer:
        memory = self.memory.get_memory(user_id)
        memory_query = self.memory.weighted_query(memory, include_artists=False)
        evidences = self.retrieve_evidence(asset_id, f"{goal} {memory_query}".strip(), top_k=max(top_k * 2, top_k))
        segment_map = {segment.segment_id: segment for segment in self._require_segments(asset_id)}
        segments: list[Segment] = []
        seen: set[str] = set()
        for evidence in evidences:
            if evidence.segment_id in seen:
                continue
            segment = segment_map.get(evidence.segment_id)
            if segment is not None:
                seen.add(evidence.segment_id)
                segments.append(segment)
        lines = [
            f"{index}. {segment.timestamp} - {segment.scene_summary}"
            for index, segment in enumerate(segments[:top_k], start=1)
        ]
        answer = "基于你的记忆和当前素材，我优先推荐这些片段：\n" + "\n".join(lines) if lines else "当前素材里没有足够明显的高匹配片段。"
        return AgentAnswer(
            answer=answer,
            evidences=evidences[:top_k],
            recommended_segments=segments[:top_k],
            agent_trace=[
                f"goal={goal}",
                f"memory_query={memory_query or 'none'}",
                f"evidence_chunks={len(evidences)}",
            ],
        )

    def generate_report(self, asset_id: str) -> dict[str, Any]:
        asset = self.store.read_model("assets", asset_id, Asset)
        if asset is None:
            raise ValueError(f"Unknown asset_id: {asset_id}")
        segments = self._require_segments(asset_id)
        evidences = self.retrieve_evidence(asset_id, "high energy climax mood genre summary", top_k=4)
        return {
            "asset": asset.model_dump(mode="json"),
            "summary": f"{asset.title} 已拆分为 {len(segments)} 个片段，可用于风格检索、推荐解释和相似内容分析。",
            "top_evidences": [evidence.model_dump(mode="json") for evidence in evidences],
            "fingerprint": {
                "genre": asset.genre,
                "mood": asset.mood,
                "tempo_bpm": asset.tempo_bpm,
                "energy_level": asset.energy_level,
            },
        }

    def summarize_taste(self, user_id: str, *, include_artists: bool = True) -> str:
        memory = self.memory.get_memory(user_id)
        if not memory.taste_profile:
            library = [asset for asset in self.list_assets() if asset.status == "analyzed"]
            memory = self.memory.refresh_taste_profile(user_id, library)
        taste = memory.taste_profile or TasteProfile()
        genres = [genre for genre, _ in taste.top_genres[:4]]
        moods = [mood for mood, _ in taste.top_moods[:4]]
        artists = [artist for artist, _ in taste.top_artists[:5]]
        prefs = memory.preferences[-3:]
        genre_text = "、".join(genres) if genres else "未形成稳定风格"
        mood_text = "、".join(moods) if moods else "暂无明显偏好"
        pref_text = "；".join(prefs) if prefs else "暂无"
        artist_text = "、".join(artists) if artists else ""
        parts = [
            f"你的品味目前更偏向 {genre_text}，"
            f"情绪上常出现 {mood_text}，"
        ]
        if include_artists and artist_text:
            parts.append(f"偏好的艺人有 {artist_text}，")
        parts.append(f"显式表达过的偏好包括 {pref_text}。")
        return "".join(parts)

    def recommend_artist_albums(self, user_id: str, artist: str, limit: int = 6) -> list[dict[str, Any]]:
        """推荐某歌手的真实专辑清单：走网易云专辑搜索（type=10，与单曲搜索不同端点），
        返回带真实 album_id 的专辑，前端可整张播放。

        关键：专辑端点不与单曲搜索共享限流，且 search_netease_artist_albums 有进程缓存——
        单曲搜索被限流掉到 netease-fallback 假候选时，专辑端点通常仍可用，故「推荐专辑」
        既更贴合用户意图，又比单曲推荐更稳。失败返回空列表（不造假）。
        """
        artist = (artist or "").strip()
        if not artist:
            return []
        try:
            return netease_source.search_netease_artist_albums(artist, limit)
        except Exception:
            logger.debug("recommend_artist_albums failed for %s", artist, exc_info=True)
            return []

    async def recommend_artist_albums_async(
        self, user_id: str, artist: str, limit: int = 6,
    ) -> list[dict[str, Any]]:
        import asyncio

        artist = (artist or "").strip()
        if not artist:
            return []
        try:
            return await netease_source.asearch_netease_artist_albums(artist, limit)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("async recommend_artist_albums failed for %s", artist, exc_info=True)
            return []

    def generate_taste_experiment(self, user_id: str, prompt: str, total: int = 12) -> TasteExperiment:
        """生成 safe/stretch/bold 三档品味实验。

        MVP 不造歌名：候选来自现有推荐/搜索，再用 rerank components 分桶。
        """
        total = max(3, min(total or 12, 30))
        per_bucket = max(1, total // 3)
        memory = self.memory.get_memory(user_id)
        hypothesis = self._taste_experiment_hypothesis(memory)
        seeds = self._taste_experiment_search_seeds(memory, prompt)
        candidates = self._collect_taste_candidates(user_id, seeds, total)
        prompt_rules = self._taste_prompt_exclusions(prompt)
        candidates = self._filter_taste_experiment_candidates(
            user_id, candidates, [*memory.exclusion_rules, *prompt_rules],
        )
        buckets = self._bucket_taste_experiment_candidates(candidates, per_bucket)
        confidence_ok = bool(buckets["safe"] and buckets["bold"])
        segments = [
            TasteExperimentSegment(
                name="safe",
                label="安全区",
                description=(
                    "命中你的核心风格或艺人，用来验证稳定偏好。"
                    if confidence_ok else "本轮证据不足，暂不强行标记安全区。"
                ),
                tracks=[
                    self._taste_experiment_track(track, "safe", components, reason, score)
                    for track, components, reason, score in buckets["safe"]
                ],
            ),
            TasteExperimentSegment(
                name="stretch",
                label="轻微越界",
                description=(
                    "和你的画像相邻，但至少在一个维度上有所变化。"
                    if confidence_ok else "候选区分度不足，先放在待验证区收集真实反馈。"
                ),
                tracks=[
                    self._taste_experiment_track(track, "stretch", components, reason, score)
                    for track, components, reason, score in buckets["stretch"]
                ],
            ),
            TasteExperimentSegment(
                name="bold",
                label="大胆探索",
                description=(
                    "有可解释连接点，同时明显超出你的主画像。"
                    if confidence_ok else "本轮证据不足，暂不强行标记大胆探索。"
                ),
                tracks=[
                    self._taste_experiment_track(track, "bold", components, reason, score)
                    for track, components, reason, score in buckets["bold"]
                ],
            ),
        ]
        actual_total = sum(len(segment.tracks) for segment in segments)
        if not confidence_ok and actual_total:
            shortfall = "本轮候选的熟悉度差异不足，已停止强行分档；请先试听待验证候选。"
        else:
            shortfall = "" if actual_total >= total else f"候选不足，本次先生成 {actual_total}/{total} 首。"
        experiment = TasteExperiment(
            experiment_id=self._new_taste_experiment_id(user_id, prompt),
            user_id=user_id,
            prompt=prompt,
            hypothesis=hypothesis,
            segments=segments,
            result_summary=shortfall,
        )
        self._save_taste_experiment(experiment)
        return experiment

    def _collect_taste_candidates(
        self,
        user_id: str,
        seeds: list[str],
        total: int,
    ) -> list[tuple[Any, dict[str, float], str, float]]:
        """汇总多路候选，再在同一查询和同一批次中统一评分。

        旧实现把每个搜索批次内部归一化后的 components 直接混排；这些分数没有
        可比性，是实验分档错乱的主因。
        """
        raw_tracks: list[Asset | ExternalTrack] = []
        if seeds:
            try:
                rec = self.recommend_for_query(user_id, seeds[0], top_k=max(total * 3, 18))
                for item in rec.tracks:
                    raw_tracks.append(item.asset)
            except Exception:
                logger.debug("taste_experiment recommend failed for %s", seeds[0], exc_info=True)
        for search_goal in seeds[:16]:
            try:
                tracks = self.search_web_music(search_goal, top_k=6, relevance_query=search_goal)
            except Exception:
                logger.debug("taste_experiment seed search failed for %s", search_goal, exc_info=True)
                continue
            raw_tracks.extend(tracks)
        raw_tracks = [
            track for track in _dedupe_tracks(raw_tracks)
            if _is_recommendation_quality_track(track)
        ]
        if not raw_tracks:
            return []
        unified_query = " ".join(seeds[:6])
        ranked = self._rerank_tracks(user_id, unified_query, raw_tracks, top_k=len(raw_tracks))
        return [
            (track, breakdown.components, breakdown.reason, breakdown.score)
            for track, breakdown in ranked
        ]

    @staticmethod
    def _taste_prompt_exclusions(prompt: str) -> list[str]:
        """Extract hard negative constraints from short experiment prompts."""
        text = (prompt or "").lower()
        rules: list[str] = []
        if any(token in text for token in ("别太吵", "不要太吵", "不吵", "低能量")):
            rules.extend(["激昂", "金属", "hard rock", "heavy metal"])
        if "type beat" in text or "不要beat" in text or "不要 beat" in text:
            rules.append("type beat")
        for match in re.finditer(r"(?:不要|别推|不想听)\s*([^，。,.；;]{1,16})", text):
            value = match.group(1).strip()
            if value:
                rules.append(value)
        return list(dict.fromkeys(rules))

    def regenerate_taste_experiment_bucket(self, user_id: str, experiment_id: str, bucket: str) -> TasteExperiment:
        """按上一轮反馈重做某一档（safe/stretch/bold）。

        取该档意图的种子重新搜候选，按 familiarity 取对应切片替换该 segment，
        并避开当前实验其它档已有的曲。这是"反馈驱动下一轮"的入口：报告判定 bold 太远
        → 用户点重做 bold → 换一批探索梯度更合适的候选。
        """
        if bucket not in {"safe", "stretch", "bold"}:
            raise ValueError(f"unknown bucket: {bucket}")
        with self.store.lock("taste_experiments", user_id):
            experiments = self.store.read_models("taste_experiments", user_id, TasteExperiment)
            exp = next((e for e in experiments if e.experiment_id == experiment_id), None)
            if exp is None:
                raise ValueError("Experiment not found")
            memory = self.memory.get_memory(user_id)
            segment = next((s for s in exp.segments if s.name == bucket), None)
            existing = sum(len(s.tracks) for s in exp.segments)
            per_bucket = len(segment.tracks) if segment and segment.tracks else max(1, (existing // 3) or 1)

            seeds = self._taste_experiment_seeds_for_bucket(memory, exp.prompt, bucket)
            candidates = self._collect_taste_candidates(user_id, seeds, per_bucket * 6)
            candidates = self._filter_taste_experiment_candidates(user_id, candidates, memory.exclusion_rules)
            # 避开其它档已有的曲，重做不撞车
            other_keys = {
                self._taste_experiment_track_key(item)
                for seg in exp.segments if seg.name != bucket
                for item in seg.tracks
            }
            candidates = [c for c in candidates if self._candidate_key(c) not in other_keys]
            ranked = sorted(candidates, key=self._taste_familiarity, reverse=True)
            band = self._slice_for_bucket(ranked, bucket, per_bucket)
            new_tracks = [
                self._taste_experiment_track(track, bucket, components, reason, score)
                for track, components, reason, score in band
            ]
            # 档内按 key 再去一次重
            seen: set[str] = set()
            deduped: list[TasteExperimentTrack] = []
            for it in new_tracks:
                k = self._taste_experiment_track_key(it)
                if k in seen:
                    continue
                seen.add(k)
                deduped.append(it)
            if segment is not None:
                segment.tracks = deduped
            exp.updated_at = utc_now_iso()
            self.store.write_models("taste_experiments", user_id, experiments[-20:])
            return exp

    def _taste_experiment_seeds_for_bucket(self, memory: UserMemory, prompt: str, bucket: str) -> list[str]:
        """按档位意图取搜索种子：safe=主打风格/歌手，stretch=相邻衍生，bold=跨风格探索。"""
        taste = memory.taste_profile or TasteProfile()
        genres = [g for g, _ in taste.top_genres[:3] if g]
        artists = [a for a, _ in taste.top_artists[:6] if a]
        if bucket == "safe":
            seeds: list[str] = [f"{a} {genres[0]}" for a in artists[:6] if genres]
            if genres:
                seeds.append(" ".join(genres[:3]))
            return self._dedupe_seeds(seeds) or ["热门 推荐"]
        if bucket == "stretch":
            seeds = []
            if genres:
                seeds.append(" ".join([genres[0], "相邻", "新风格"]))
            seeds += ["neo soul", "另类 R&B", "独立流行", "律动 R&B", "氛围 说唱"]
            return self._dedupe_seeds(seeds)
        # bold：跨风格、世界音乐、实验性
        return self._dedupe_seeds([
            "探索 新风格", "小众 世界音乐", "实验 电子", "融合 爵士", "独立 民谣", "另类 摇滚",
        ])

    @staticmethod
    def _dedupe_seeds(seeds: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for s in seeds:
            s = s.strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
        return out

    def list_taste_experiments(self, user_id: str) -> list[TasteExperiment]:
        return self.store.read_models("taste_experiments", user_id, TasteExperiment)

    def get_taste_experiment(self, user_id: str, experiment_id: str) -> TasteExperiment | None:
        return next((exp for exp in self.list_taste_experiments(user_id) if exp.experiment_id == experiment_id), None)

    def delete_taste_experiment(self, user_id: str, experiment_id: str) -> bool:
        with self.store.lock("taste_experiments", user_id):
            experiments = self.store.read_models("taste_experiments", user_id, TasteExperiment)
            remaining = [exp for exp in experiments if exp.experiment_id != experiment_id]
            if len(remaining) == len(experiments):
                return False
            self.store.write_models("taste_experiments", user_id, remaining)
            return True

    def record_taste_experiment_feedback(self, request: TasteExperimentFeedbackRequest) -> TasteExperiment:
        with self.store.lock("taste_experiments", request.user_id):
            experiments = self.store.read_models("taste_experiments", request.user_id, TasteExperiment)
            for exp in experiments:
                if exp.experiment_id != request.experiment_id:
                    continue
                item = self._find_taste_experiment_track(exp, request.track_key)
                if item is None:
                    raise ValueError("Track not found in experiment")
                feedback = item.feedback
                current = getattr(feedback, request.signal)
                setattr(feedback, request.signal, current + 1)
                feedback.last_signal = request.signal
                if request.signal == "rated" and request.score is not None:
                    feedback.scores.append(float(request.score))
                self._apply_taste_experiment_ts_feedback(item, request.signal, request.score)
                self._record_taste_experiment_listen(request.user_id, item, request.signal, request.score)
                if self._taste_experiment_feedback_count(exp) >= 6 and exp.status == "collecting":
                    exp.status = "ready"
                exp.updated_at = utc_now_iso()
                self.store.write_models("taste_experiments", request.user_id, experiments[-20:])
                return exp
        raise ValueError("Experiment not found")

    def summarize_taste_experiment(self, user_id: str, experiment_id: str) -> TasteExperimentReport:
        with self.store.lock("taste_experiments", user_id):
            experiments = self.store.read_models("taste_experiments", user_id, TasteExperiment)
            for exp in experiments:
                if exp.experiment_id != experiment_id:
                    continue
                stats = self._taste_experiment_bucket_stats(exp)
                feedback_total = int(sum(bucket.get("feedback_count", 0) for bucket in stats.values()))
                if feedback_total < 6:
                    summary = f"目前只有 {feedback_total} 条实验反馈，再听/跳过几首后报告会更可靠。"
                    hypothesis_result = "继续收集"
                    next_strategy = "先保持三档结构，优先补足 stretch 和 bold 的反馈。"
                else:
                    best_bucket = max(stats, key=lambda name: (stats[name].get("liked_rate", 0), stats[name].get("completed_rate", 0)))
                    too_far = stats.get("bold", {}).get("too_far", 0)
                    summary = f"已收集 {feedback_total} 条反馈，{self._bucket_label(best_bucket)} 的正反馈最强。"
                    hypothesis_result = (
                        "大胆探索边界偏远，需要收窄。"
                        if too_far >= 2 else
                        f"假设部分成立：{self._bucket_label(best_bucket)} 当前最能解释你的反应。"
                    )
                    next_strategy = (
                        "下一轮降低 bold 能量跨度，多做相邻风格实验。"
                        if too_far >= 2 else
                        "下一轮保留 safe 锚点，把 stretch 的比例提高一点。"
                    )
                report = TasteExperimentReport(
                    summary=summary,
                    bucket_stats=stats,
                    hypothesis_result=hypothesis_result,
                    next_recommendation_strategy=next_strategy,
                )
                exp.report = report
                exp.result_summary = summary
                exp.status = "reported" if feedback_total >= 6 else exp.status
                exp.updated_at = utc_now_iso()
                self.store.write_models("taste_experiments", user_id, experiments[-20:])
                return report
        raise ValueError("Experiment not found")

    def _save_taste_experiment(self, experiment: TasteExperiment) -> None:
        with self.store.lock("taste_experiments", experiment.user_id):
            experiments = self.store.read_models("taste_experiments", experiment.user_id, TasteExperiment)
            experiments = [exp for exp in experiments if exp.experiment_id != experiment.experiment_id]
            experiments.append(experiment)
            self.store.write_models("taste_experiments", experiment.user_id, experiments[-20:])

    @staticmethod
    def _new_taste_experiment_id(user_id: str, prompt: str) -> str:
        raw = f"{user_id}|{prompt}|{datetime.now(UTC).isoformat()}".encode()
        return "taste_" + hashlib.sha1(raw).hexdigest()[:12]

    def _taste_experiment_hypothesis(self, memory: UserMemory) -> str:
        taste = memory.taste_profile or TasteProfile()
        genres = "、".join(name for name, _ in taste.top_genres[:2]) or "你最近反复命中的风格"
        moods = "、".join(name for name, _ in taste.top_moods[:2]) or "稳定情绪锚点"
        return f"我猜你会稳定接受 {genres}/{moods}，但探索边界可能藏在相邻风格和不同能量密度里。"

    def _taste_experiment_search_seeds(self, memory: UserMemory, prompt: str) -> list[str]:
        """把“做个品味实验”这类功能句改写成音乐平台能搜到的候选种子。"""
        taste = memory.taste_profile or TasteProfile()
        genres = [name for name, _ in taste.top_genres[:3] if name]
        moods = [name for name, _ in taste.top_moods[:3] if name]
        artists = [name for name, _ in taste.top_artists[:5] if name]
        prompt = prompt or ""
        seeds: list[str] = []
        for artist in artists[:6]:
            primary_genre = genres[1] if len(genres) > 1 else (genres[0] if genres else "")
            if primary_genre:
                seeds.append(f"{artist} {primary_genre}")
            if moods:
                seeds.append(f"{artist} {moods[0]}")
        if genres:
            seeds.append(" ".join([*genres[:2], "新风格"]))
            seeds.append(" ".join([genres[0], "小众", "相邻风格"]))
        if moods:
            seeds.append(" ".join([moods[0], "氛围", "新歌"]))
        if any(token in prompt for token in ["不一样", "听腻", "新风格", "探索", "实验"]):
            seeds.extend(["探索 新风格", "小众 R&B", "另类 R&B", "neo soul", "氛围 说唱", "新灵魂"])
        seeds.extend(["独立流行", "律动 R&B", "另类流行", "chill R&B"])
        seen: set[str] = set()
        uniq: list[str] = []
        for seed in seeds:
            seed = seed.strip()
            if seed and seed.lower() not in seen:
                seen.add(seed.lower())
                uniq.append(seed)
        return uniq or ["探索 新风格"]

    def _filter_taste_experiment_candidates(
        self,
        user_id: str,
        candidates: list[tuple[Any, dict[str, float], str, float]],
        exclusion_rules: list[str],
    ) -> list[tuple[Any, dict[str, float], str, float]]:
        seen: set[str] = set()
        artist_counts: dict[str, int] = {}
        filtered: list[tuple[Any, dict[str, float], str, float]] = []
        rules = [rule.lower().strip() for rule in exclusion_rules if rule.strip()]
        for track, components, reason, score in candidates:
            if self.library.is_disliked(user_id, track):
                continue
            if not self._is_taste_experiment_quality_track(track):
                continue
            searchable = " ".join([
                getattr(track, "title", "") or "",
                getattr(track, "artist", "") or "",
                " ".join(getattr(track, "genre", []) or []),
                " ".join(getattr(track, "mood", []) or []),
            ]).lower()
            if any(rule and rule in searchable for rule in rules):
                continue
            title = (getattr(track, "title", "") or "").strip().lower()
            artist = (getattr(track, "artist", "") or "").strip().lower()
            title_artist_key = f"title:{title}:{artist}"
            base_title = re.sub(r"\s*[\[(（].*?[\])）]", "", title).strip()
            base_title_artist_key = f"title:{base_title}:{artist}"
            external_id = getattr(track, "external_id", "") or getattr(track, "source_id", "") or ""
            external_key = f"external:{external_id}" if external_id else ""
            if title_artist_key in seen or base_title_artist_key in seen or (external_key and external_key in seen):
                continue
            primary_artist = re.split(r"[、,/&]| feat\\.? | ft\\.? ", artist, maxsplit=1)[0].strip() or artist
            if primary_artist and artist_counts.get(primary_artist, 0) >= 4:
                continue
            seen.add(title_artist_key)
            seen.add(base_title_artist_key)
            if external_key:
                seen.add(external_key)
            if primary_artist:
                artist_counts[primary_artist] = artist_counts.get(primary_artist, 0) + 1
            filtered.append((track, components or {}, reason, float(score or 0.0)))
        return filtered

    @staticmethod
    def _is_taste_experiment_quality_track(track: Any) -> bool:
        """Taste Lab 候选质量门槛：挡掉明显不像正式歌曲的搜索噪声。"""
        return _is_recommendation_quality_track(track)

    def _bucket_taste_experiment_candidates(
        self,
        candidates: list[tuple[Any, dict[str, float], str, float]],
        per_bucket: int,
    ) -> dict[str, list[tuple[Any, dict[str, float], str, float]]]:
        """按 familiarity 排名切片成 safe/stretch/bold 三等分。

        旧实现用 personalize/behavior 的绝对阈值分桶，但行为锚长期为 0、个性化在在线
        候选上常偏低，导致三档全部塌向 bold。改为按 familiarity=0.6 口味 + 0.3 语义 +
        0.1 行为 的相对排名切片：最像口味→safe，中间→stretch，最不像→bold，保证三档
        均衡且探索梯度真实存在。
        """
        buckets: dict[str, list[tuple[Any, dict[str, float], str, float]]] = {"safe": [], "stretch": [], "bold": []}
        if not candidates:
            return buckets
        ranked = sorted(candidates, key=self._taste_familiarity, reverse=True)
        familiarities = [self._taste_familiarity(item) for item in ranked]
        # 绝对区分度不足时不制造虚假的 safe/bold 确定性：统一放进待验证档。
        if len(ranked) < per_bucket * 3 or max(familiarities) - min(familiarities) < 0.08:
            buckets["stretch"] = ranked[:per_bucket * 3]
            return buckets
        buckets["safe"] = self._slice_for_bucket(ranked, "safe", per_bucket)
        buckets["stretch"] = self._slice_for_bucket(ranked, "stretch", per_bucket)
        buckets["bold"] = self._slice_for_bucket(ranked, "bold", per_bucket)
        return buckets

    @staticmethod
    def _taste_familiarity(item: tuple[Any, dict[str, float], str, float]) -> float:
        """单候选的「熟悉度」：口味契合为主，语义为辅，行为微调（行为只对听过的歌有信号）。"""
        _, components, _, _ = item
        per = components.get("personalize", 0.0)
        sem = components.get("semantic", 0.0)
        beh = components.get("behavior", 0.0)
        return per * 0.6 + sem * 0.3 + beh * 0.1

    @staticmethod
    def _slice_for_bucket(
        ranked: list[tuple[Any, dict[str, float], str, float]],
        bucket: str,
        per_bucket: int,
    ) -> list[tuple[Any, dict[str, float], str, float]]:
        if bucket == "safe":
            return ranked[0:per_bucket]
        if bucket == "stretch":
            return ranked[per_bucket:2 * per_bucket]
        return ranked[2 * per_bucket:3 * per_bucket]

    @staticmethod
    def _candidate_key(item: tuple[Any, dict[str, float], str, float]) -> str:
        track, _, _, _ = item
        source = getattr(track, "source", "netease") or "netease"
        external_id = getattr(track, "external_id", "") or getattr(track, "source_id", "") or ""
        if external_id:
            return f"{source}:{external_id}"
        title = (getattr(track, "title", "") or "").strip().lower()
        artist = (getattr(track, "artist", "") or "").strip().lower()
        return f"title:{title}:{artist}"

    @staticmethod
    def _taste_experiment_track(
        track: Any,
        bucket: str,
        components: dict[str, float],
        reason: str,
        score: float,
    ) -> TasteExperimentTrack:
        source = getattr(track, "source", "local") or "local"
        source_id = getattr(track, "external_id", "") or getattr(track, "asset_id", "") or ""
        ref = TrackRef(
            title=getattr(track, "title", "") or "",
            artist=getattr(track, "artist", "") or "",
            source=source,
            source_id=source_id,
            genre=getattr(track, "genre", []) or [],
            mood=getattr(track, "mood", []) or [],
            score=score,
            components=components or {},
        )
        expected = {
            "safe": "如果你听完或收藏，说明稳定画像可信。",
            "stretch": "如果你喜欢，说明相邻风格可以扩大。",
            "bold": "如果你没跳过，说明探索边界比画像更宽。",
        }[bucket]
        return TasteExperimentTrack(
            track=ref,
            bucket=bucket,  # type: ignore[arg-type]
            reason=reason or f"{bucket} bucket candidate",
            expected_signal=expected,
            components=components or {},
        )

    @staticmethod
    def _taste_experiment_track_key(item: TasteExperimentTrack) -> str:
        source_id = item.track.source_id.strip()
        if source_id:
            return f"{item.track.source}:{source_id}"
        return f"title:{item.track.title.lower()}:{item.track.artist.lower()}"

    def _find_taste_experiment_track(self, experiment: TasteExperiment, track_key: str) -> TasteExperimentTrack | None:
        for segment in experiment.segments:
            for item in segment.tracks:
                if self._taste_experiment_track_key(item) == track_key:
                    return item
        return None

    def _apply_taste_experiment_ts_feedback(self, item: TasteExperimentTrack, signal: str, score: float | None) -> None:
        if not item.track.source_id:
            return
        positive = signal in {"completed", "liked", "saved"} or (signal == "rated" and (score or 0) >= 7)
        negative = signal in {"skipped", "disliked"} or (signal == "rated" and (score or 10) <= 4)
        if not positive and not negative:
            return
        track = ExternalTrack(
            external_id=item.track.source_id,
            title=item.track.title,
            artist=item.track.artist or "",
            genre=item.track.genre,
            mood=item.track.mood,
            source=item.track.source,
        )
        self.library.update_ts_feedback(track, positive=positive, weight=1.0 if positive else 0.6)

    def _record_taste_experiment_listen(
        self,
        user_id: str,
        item: TasteExperimentTrack,
        signal: str,
        score: float | None,
    ) -> None:
        """把品味实验反馈也写进 listening_history，让行为锚在下一轮实验/推荐里拿到信号。

        key 用 source_id（网易云在线 id），与候选 _track_id 同命名空间，行为锚才能命中。
        completed/liked/saved 视为听完(+1)；skipped/disliked/too_far/too_safe 视为秒跳(-1)；
        rated 按分数极性，中性不记。这打通了"实验反馈→行为锚→下一轮排序"的闭环。
        """
        source_id = (item.track.source_id or "").strip()
        if not source_id:
            return  # 无在线 id 的曲无法与候选对齐，不记
        if signal in {"completed", "liked", "saved"}:
            completed, duration = True, 180
        elif signal in {"skipped", "disliked", "too_far", "too_safe"}:
            completed, duration = False, 0
        elif signal == "rated":
            if (score or 0) >= 7:
                completed, duration = True, 180
            elif (score or 10) <= 4:
                completed, duration = False, 0
            else:
                return  # 中性评分不记
        else:
            return
        try:
            self.memory.record_listen(user_id, source_id, duration, completed, context=f"taste_lab:{signal}")
        except Exception:
            logger.debug("taste experiment listen record failed", exc_info=True)

    @staticmethod
    def _taste_experiment_feedback_count(experiment: TasteExperiment) -> int:
        total = 0
        for segment in experiment.segments:
            for item in segment.tracks:
                fb = item.feedback
                total += fb.completed + fb.skipped + fb.liked + fb.disliked + fb.saved + fb.rated + fb.too_safe + fb.too_far
        return total

    def _taste_experiment_bucket_stats(self, experiment: TasteExperiment) -> dict[str, dict[str, float | int]]:
        stats: dict[str, dict[str, float | int]] = {}
        for segment in experiment.segments:
            total_tracks = len(segment.tracks)
            completed = skipped = liked = disliked = saved = too_safe = too_far = rated = 0
            scores: list[float] = []
            for item in segment.tracks:
                fb = item.feedback
                completed += fb.completed
                skipped += fb.skipped
                liked += fb.liked
                disliked += fb.disliked
                saved += fb.saved
                too_safe += fb.too_safe
                too_far += fb.too_far
                rated += fb.rated
                scores.extend(fb.scores)
            feedback_count = completed + skipped + liked + disliked + saved + too_safe + too_far + rated
            denom = max(feedback_count, 1)
            stats[segment.name] = {
                "tracks": total_tracks,
                "feedback_count": feedback_count,
                "completed": completed,
                "skipped": skipped,
                "liked": liked,
                "disliked": disliked,
                "saved": saved,
                "too_safe": too_safe,
                "too_far": too_far,
                "completed_rate": round(completed / denom, 3),
                "skip_rate": round(skipped / denom, 3),
                "liked_rate": round((liked + saved) / denom, 3),
                "avg_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
            }
        return stats

    @staticmethod
    def _bucket_label(bucket: str) -> str:
        return {"safe": "安全区", "stretch": "轻微越界", "bold": "大胆探索"}.get(bucket, bucket)

    def recommend_for_query(
        self,
        user_id: str,
        goal: str,
        top_k: int = 5,
        *,
        excluded_tracks: list[dict[str, str]] | None = None,
        search_variants: list[str] | None = None,
        seed_tracks: list[Asset | ExternalTrack] | None = None,
    ) -> DailyRecommendation:
        memory = self.memory.get_memory(user_id)
        memory_query = self.memory.weighted_query(memory, include_artists=False)
        search_goal = _extract_search_query(goal)
        has_entity = self._query_has_entity(search_goal)
        taste_summary = self.summarize_taste(user_id, include_artists=has_entity) if memory.taste_profile else ""
        library_artists = list({a.artist for a in self.list_assets() if a.artist})[:10] if has_entity else []

        # ── 三路搜索策略 ──
        # 精确实体查询 (有歌手/歌名) → 网易云歌曲搜索
        # 情绪/场景/模糊查询 → LLM 候选生成 + 网易云歌单搜索
        trace_lines: list[str] = []
        all_candidates: list[Asset | ExternalTrack] = list(seed_tracks or [])
        seed_supply = sum(
            1 for track in all_candidates
            if _is_verified_recommendation_track(track)
            and _is_recommendation_quality_track(track)
        )

        if seed_supply >= top_k:
            trace_lines.append(f"route=seed_candidates, supplied={seed_supply}")
        elif has_entity:
            # 路由C：精确搜索（网易云歌曲搜索，这个是OK的）
            trace_lines.append(f"route=exact, search_goal={search_goal}")
            # 延续去重时翻页：跳过已展示的那批最热结果，取更深位次新歌，
            # 否则同一查询永远返回 top-N，去重后很快就无新歌可推。
            rec_offset = len(excluded_tracks) if excluded_tracks else 0
            batch = self.search_web_music(
                search_goal, top_k=max(top_k * 2, top_k),
                relevance_query=search_goal, offset=rec_offset, variants=search_variants,
            )
            all_candidates.extend(batch)
        else:
            # 路由B（优先）：网易云歌单搜索——真人策划歌单，质量最高
            from app.search.netease_playlist import search_and_extract
            # 构建精炼的歌单搜索词：品味风格 + 查询意图
            taste_genres = ""
            if memory.taste_profile and memory.taste_profile.top_genres:
                taste_genres = " ".join(g for g, _ in memory.taste_profile.top_genres[:2])
            playlist_query = f"{taste_genres} {search_goal}".strip() or goal
            playlist_tracks = search_and_extract(playlist_query, max_playlists=3, tracks_per_playlist=top_k)
            trace_lines.append(f"route=playlist, query={playlist_query!r}, extracted={len(playlist_tracks)}")
            all_candidates.extend(playlist_tracks)

            # 路由A（补位）：LLM 候选生成 → 网易云验证（歌单不够时补充）
            from app.search.web_music_discovery import discover_from_llm
            llm_tracks = discover_from_llm(
                query=goal,
                taste_summary=taste_summary,
                exclusion_rules=memory.exclusion_rules,
                library_artists=library_artists,
                target_count=top_k,
                llm=self.llm,
            )
            trace_lines.append(f"route=llm_candidates, generated={len(llm_tracks)}")
            all_candidates.extend(llm_tracks)

            # 路由E：Last.fm 发现 → 网易云验证（需要配置 LASTFM_API_KEY）
            from app.search.lastfm_discovery import discover_from_lastfm
            taste_artists = [a for a, _ in (memory.taste_profile.top_artists if memory.taste_profile else [])]
            taste_genre_names = [g for g, _ in (memory.taste_profile.top_genres if memory.taste_profile else [])]
            lastfm_tracks = discover_from_lastfm(
                top_artists=taste_artists or library_artists,
                top_genres=taste_genre_names,
                target_count=top_k,
            )
            if lastfm_tracks:
                trace_lines.append(f"route=lastfm, verified={len(lastfm_tracks)}")
                all_candidates.extend(lastfm_tracks)

        # 本地曲库必须真正参与推荐，而不只是被压缩成画像后再去线上搜。
        # 精确实体查询只加入标题/歌手匹配项；场景查询加入画像/场景相关项。
        local_candidates = self._local_recommendation_candidates(user_id, search_goal or goal, memory)
        trace_lines.append(f"route=local_library, matched={len(local_candidates)}")
        all_candidates.extend(local_candidates)

        # 去重 + 过滤。线上候选必须真实验证；本地曲库本身即可信来源。
        verified = [
            track for track in _dedupe_tracks(all_candidates)
            if _is_verified_recommendation_track(track)
            and not self.library.is_disliked(user_id, track)
            and _is_recommendation_quality_track(
                track, allow_variants=_query_requests_variant_content(goal)
            )
        ]

        # 过滤上一轮已展示的曲目（延续指令去重）
        if excluded_tracks:
            verified = _filter_excluded_tracks(verified, excluded_tracks)

        # 兜底：用 search_goal 再搜一次。带 offset 翻页（已排除 + 已收集数），
        # 否则同查询永远返回 top-N，与首轮 batch 重复，dedup 全跳过、补不了量。
        if len(verified) < top_k and search_goal:
            fb_offset = len(excluded_tracks or []) + len(verified)
            fallback_batch = self.search_web_music(
                search_goal, top_k=max(top_k * 2, top_k), offset=fb_offset,
                variants=search_variants,
            )
            for track in fallback_batch:
                if _is_verified_online_track(track) and not self.library.is_disliked(user_id, track):
                    if not any(_track_key(track) == _track_key(v) for v in verified):
                        verified.append(track)

        if verified:
            # 跨轮优先使用未展示过的歌；不足时才把旧歌放回尾部，避免越刷越空。
            recent = set(memory.recommendation_history[-120:])
            fresh = [track for track in verified if _track_key(track) not in recent]
            repeated = [track for track in verified if _track_key(track) in recent]
            verified = fresh if len(fresh) >= top_k else [*fresh, *repeated]
            rerank_query = search_goal or goal
            # 先对完整候选池排序，再做来源平衡。若这里只取 top_k，标签更完整的
            # local 会在平衡前就把 online 全挤掉，后续再设配额也无候选可选。
            ranked_pool = self._rerank_tracks(
                user_id, rerank_query, _dedupe_tracks(verified), top_k=len(verified),
            )
            ranked = self._balance_recommendation_sources(ranked_pool, top_k)
            self.library.record_exposure([t for t, _ in ranked])
            self.library.decay_exposure_ts([t for t, _ in ranked])
            tracks: list[RecommendedTrack] = []
            for track, breakdown in ranked:
                tracks.append(RecommendedTrack(
                    asset=track,
                    score=breakdown.score,
                    reason=breakdown.reason or _online_candidate_reason(track, memory_query),
                    category="discovery",
                    components=breakdown.components,
                ))
            self._record_recommendation_history(user_id, [track for track, _ in ranked])
            local_count = sum(_is_local_recommendation_track(track) for track, _ in ranked)
            online_count = len(ranked) - local_count
            return DailyRecommendation(
                user_id=user_id,
                tracks=tracks,
                reason_summary=(
                    f"采用 {local_count} 首曲库歌曲 + {online_count} 首真实线上候选，"
                    "经三锚精排、来源平衡与多样性重排。"
                ),
                agent_trace=[
                    *trace_lines,
                    f"online_verified={len(verified)}",
                    f"source_mix=local:{local_count},online:{online_count}",
                    "rerank=tri_anchor+mmr",
                ],
            )

        logger.warning("recommend_for_query: no verified online candidates for goal=%s", goal)
        return DailyRecommendation(
            user_id=user_id,
            tracks=[],
            reason_summary=f"未找到与「{goal}」匹配的真实线上候选，暂不推荐虚构歌曲。",
            agent_trace=[*trace_lines, "online_verified=0"],
        )

    def _local_recommendation_candidates(
        self,
        user_id: str,
        query: str,
        memory: UserMemory,
        limit: int = 120,
    ) -> list[Asset]:
        """Select relevant local songs without mutating the user's library."""
        taste = memory.taste_profile or TasteProfile()
        preferred = {
            *(name.lower() for name, _ in taste.top_genres[:5]),
            *(name.lower() for name, _ in taste.top_moods[:5]),
            *(name.lower() for name, _ in taste.top_artists[:8]),
        }
        query_terms = {
            token.lower() for token in re.findall(r"[A-Za-z0-9&'-]+|[一-鿿㐀-䶿]{2,}", query or "")
            if token.lower() not in _QUERY_NOISE
        }
        scored: list[tuple[int, Asset]] = []
        for track in self.list_assets():
            if track.status != "analyzed" or not _is_recommendation_quality_track(track):
                continue
            if self.library.is_disliked(user_id, track):
                continue
            searchable = " ".join([
                track.title, track.artist or "", *track.genre, *track.mood,
            ]).lower()
            query_hits = sum(1 for term in query_terms if term in searchable)
            taste_hits = sum(1 for term in preferred if term and term in searchable)
            score = query_hits * 3 + taste_hits
            if score > 0:
                scored.append((score, track))
        scored.sort(key=lambda item: (-item[0], item[1].title.lower(), (item[1].artist or "").lower()))
        return [track for _, track in scored[:limit]]

    @staticmethod
    def _balance_recommendation_sources(
        ranked: list[tuple[Asset | ExternalTrack, Any]],
        top_k: int,
        local_ratio: float = 0.4,
    ) -> list[tuple[Asset | ExternalTrack, Any]]:
        """Keep local useful without allowing metadata-rich local tracks to monopolize results.

        The cap is soft: when verified online supply is insufficient, skipped local tracks
        fill the remaining slots so recommendations never become artificially short.
        """
        if not ranked or top_k <= 0:
            return []
        local = [item for item in ranked if _is_local_recommendation_track(item[0])]
        online = [item for item in ranked if not _is_local_recommendation_track(item[0])]
        if not online:
            return local[:top_k]
        if not local:
            return online[:top_k]

        local_target = min(len(local), max(1, round(top_k * local_ratio)))
        online_target = min(len(online), top_k - local_target)
        # 某一来源不足时由另一来源补齐，但只在确实缺货时突破软配额。
        remaining = top_k - local_target - online_target
        if remaining > 0:
            extra_online = min(remaining, len(online) - online_target)
            online_target += extra_online
            remaining -= extra_online
        if remaining > 0:
            local_target += min(remaining, len(local) - local_target)

        total = local_target + online_target
        selected: list[tuple[Asset | ExternalTrack, Any]] = []
        local_used = online_used = 0
        for position in range(total):
            # 按累计目标交错来源，避免“完整 25 首比例正常，但发现页前 8 首全 local”。
            should_have_local = round((position + 1) * local_target / total)
            if local_used < should_have_local and local_used < local_target:
                selected.append(local[local_used])
                local_used += 1
            elif online_used < online_target:
                selected.append(online[online_used])
                online_used += 1
            elif local_used < local_target:
                selected.append(local[local_used])
                local_used += 1
        return selected

    def _record_recommendation_history(self, user_id: str, tracks: list[Asset | ExternalTrack]) -> None:
        keys = [_track_key(track) for track in tracks]
        # 部分嵌入调用和单元测试会用 __new__ 构造无持久层的轻量 Agent。
        # 推荐本身仍应可用，只跳过跨轮历史记录。
        if not keys or not hasattr(self, "store"):
            return
        with self.store.lock("memory", user_id):
            memory = self.memory.get_memory(user_id)
            memory.recommendation_history = [*memory.recommendation_history, *keys][-200:]
            memory.updated_at = utc_now_iso()
            self.store.write_model("memory", user_id, memory)

    @staticmethod
    def _query_has_entity(search_goal: str) -> bool:
        """判断搜索目标是否包含精确实体（歌手名/歌名），而非纯情绪/场景词。

        简洁策略：
        - 英文非风格词 → True（如 Drake, Frank Ocean）
        - 中文且包含"常见歌手姓/名模式"或已知歌手 → True
        - 否则 → False（走 LLM 候选 + 歌单搜索）
        """
        if not search_goal:
            return False
        import re

        # 英文：排除纯风格/情绪词。默认"未知英文词=实体"会把情绪/场景描述词
        # （cozy/dreamy/mellow 等）误判成歌手名，进而走网易云精确单曲搜索而搜空。
        # 这里尽量收全不可能是歌手名的描述性词汇，降低误判（真实歌手名不会落进此表）。
        generic_en = {
            # 氛围/情绪
            "chill", "lofi", "lo-fi", "vibe", "vibes", "mix", "remix", "relax", "relaxing",
            "mood", "moody", "groove", "groovy", "upbeat", "slow", "fast", "happy", "sad",
            "deep", "party", "cozy", "dreamy", "mellow", "smooth", "calm", "calming", "peaceful",
            "soothing", "soft", "warm", "bright", "dark", "melancholy", "melancholic",
            "nostalgic", "uplifting", "energetic", "emotional", "romantic", "sexy", "sensual",
            "dramatic", "epic", "ethereal", "atmospheric", "minimal", "lush",
            # 曲风
            "r&b", "rnb", "soul", "pop", "rock", "rap", "hip", "hop", "hiphop", "jazz",
            "electronic", "edm", "ambient", "acoustic", "indie", "funk", "house", "techno",
            "trap", "disco", "reggae", "blues", "country", "classical", "metal", "punk",
            "folk", "dance", "dreampop", "shoegaze", "synthwave", "instrumental", "vocal",
            # 场景/时间
            "morning", "night", "nighttime", "evening", "afternoon", "midnight", "summer",
            "winter", "autumn", "spring", "rainy", "sunny", "study", "focus", "sleep", "sleepy",
            "workout", "gym", "running", "driving", "coffee", "work", "working", "commute",
            # 功能/描述
            "playlist", "playlists", "songs", "song", "music", "track", "tracks", "recommend",
            "recommendation", "recommendations", "best", "top", "new", "old", "classic",
            "popular", "trending", "favorite", "favourites", "similar", "like", "beats", "tunes",
        }
        english = re.findall(r"[A-Za-z][A-Za-z0-9'&\-]*", search_goal)
        english = [t for t in english if len(t) > 1 and t.lower() not in generic_en]
        if english:
            return True

        # 中文：检测是否包含歌手/歌名模式的词
        # 大多数中文歌手名是 2-4 个字符的人名，而纯功能/情绪查询的特征是：
        # 只有短(2字)的情绪/场景/时间/功能词
        cjk_tokens = re.findall(r"[一-鿿㐀-䶿]{2,}", search_goal)

        # 所有非噪声 CJK token 都在泛化词表里 → 没有实体
        _GENERAL_WORDS = {
            # 情绪/氛围
            "慵懒", "律动", "放松", "治愈", "欢快", "伤感", "浪漫", "激昂", "宁静", "梦幻",
            "轻松", "开心", "忧郁", "温馨", "热血", "安静", "舒缓", "劲爆", "性感", "温柔",
            "甜蜜", "兴奋", "空灵", "愉悦", "感动", "舒服", "烦躁", "低沉", "吵闹",
            # 场景
            "跑步", "运动", "工作", "学习", "睡眠", "开车", "通勤", "派对", "咖啡",
            "健身", "旅行", "约会", "散步", "泡澡", "专注",
            # 时间
            "深夜", "早晨", "下午", "夜晚", "凌晨", "熬夜", "今夜", "今晚", "周末",
            "早上", "晚上", "白天", "午后", "傍晚",
            # 风格/曲风（这些不是实体，是分类词）
            "说唱", "摇滚", "电子", "古典", "爵士", "民谣", "国风", "金属", "朋克",
            "嘻哈", "蓝调", "乡村", "雷鬼", "灵魂", "放克", "迪斯科", "浩室",
            "独立", "后摇", "新浪潮", "实验", "氛围", "新金属",
            # 功能/描述
            "混搭", "推荐", "适合", "流行", "好听", "经典", "热门", "小众", "风格",
            "陪伴", "陪你", "感觉", "能量", "曲风", "节奏", "全部", "一些", "几首", "都有", "全都有",
            "唱歌", "跳舞", "听歌", "背景",
            # 常见功能词
            "从", "到", "帮", "让", "给", "想", "要", "能", "来", "去",
            "一个人", "两个人", "朋友", "恋人", "情侣",
            "好听的音乐", "推荐一些歌", "推荐几首歌", "帮我推荐一些歌",
            "给我推荐", "推荐一些", "推荐几首", "帮我推荐",
        }
        non_general = [t for t in cjk_tokens if t not in _GENERAL_WORDS and t not in _QUERY_NOISE]
        return bool(non_general)

    def _rerank_tracks(self, user_id: str, query: str, tracks: list[Any], top_k: int):
        """三锚精排 + MMR 多样性重排管线。返回 [(track, RankingBreakdown), ...]。"""
        from app.graph.tag_rules import extract_scenario
        from app.memory import compute_behavior_scores
        from app.recommend.rerank import rerank_candidates

        if not settings.enable_rerank or not tracks:
            from app.models import RankingBreakdown
            fallback = [
                (t, RankingBreakdown(
                    title=getattr(t, "title", ""), source=getattr(t, "source", "local"),
                    score=round(1.0 - i * 0.04, 4), reason="顺序兜底（rerank 关闭）",
                ))
                for i, t in enumerate(tracks[:top_k])
            ]
            return fallback

        memory = self.memory.get_memory(user_id)
        taste = memory.taste_profile
        durations = {a.asset_id: a.duration_seconds for a in self.list_assets()}
        behavior = compute_behavior_scores(memory.listening_history, durations)
        scenarios = {s.lower() for s in extract_scenario(query)}
        # 关键修复：在线候选 genre/mood 常为空，导致口味锚 Jaccard 恒为 0、精排空转。
        # 用规则从标题+歌手推断补全，让三锚精排有信号可比。
        self._enrich_candidate_tags(tracks)
        # 语言加权：按曲库的中/英文分布偏好同语言候选（英文歌多则多推英文，但不排斥中文）。
        from app.recommend.rerank import language_distribution
        lang_pref = language_distribution(self.list_assets())
        # 排除规则：用户明确表示不要的风格/类型
        exclusion_rules = memory.exclusion_rules or None
        # P2-H：协同过滤第四锚——跨用户共现。冷启动（无共现/无近期收听）自动关闭，
        # 由 rerank 权重重分配让回三锚。
        cf_scores, cf_ok = self._collaborative_scores(user_id, tracks, memory)
        ts_scores = self.library.sample_ts_scores(tracks) if settings.enable_explore else None
        return rerank_candidates(
            query, tracks, taste,
            behavior_scores=behavior, scenarios=scenarios, top_k=top_k,
            lang_pref=lang_pref, exclusion_rules=exclusion_rules,
            collaborative_scores=cf_scores, collaborative_ok=cf_ok,
            ts_scores=ts_scores,
        )

    def _collaborative_scores(self, user_id: str, tracks: list[Any], memory: Any) -> tuple[list[float] | None, bool]:
        """为候选计算 CF 共现分（归一 [0,1]）。仅在 w_collaborative>0 且有数据时启用。"""
        if settings.tri_anchor_w_collaborative <= 0 or not tracks:
            return None, False
        try:
            from app.recommend.collaborative import (
                build_cooccurrence,
                collaborative_scores,
                recent_listened_ids,
            )
            from app.recommend.rerank import _track_id

            recent = recent_listened_ids(memory.listening_history)
            if not recent:
                return None, False
            histories: list[list[str]] = []
            for uid in self.store.list_keys("memory"):
                um = self.memory.get_memory(uid)
                ids = [getattr(ev, "asset_id", "") for ev in um.listening_history]
                if ids:
                    histories.append(ids)
            cooccurrence = build_cooccurrence(histories)
            scores, ok = collaborative_scores(
                [_track_id(t) for t in tracks], recent, cooccurrence
            )
            return (scores, ok) if ok else (None, False)
        except Exception:
            logger.debug("CF 协同锚计算失败，降级三锚", exc_info=True)
            return None, False

    @staticmethod
    def _enrich_candidate_tags(tracks: list[Any]) -> None:
        """给缺 genre/mood 的候选用关键词规则就地补全（不写库，仅供本次精排）。

        三层补全（复用 _ensure_track_tags 的模式）：
        1. 标题+歌手关键词规则（extract_genre/extract_mood）
        2. 歌手名→风格映射表（extract_genre_from_artist，~140 艺人覆盖）
        3. 歌手名→情绪映射表（extract_mood 补充）
        """
        from app.graph.tag_rules import extract_genre, extract_genre_from_artist, extract_mood

        for t in tracks:
            text = f"{getattr(t, 'title', '')} {getattr(t, 'artist', '') or ''}"
            if not getattr(t, "genre", None):
                inferred = extract_genre(text)
                if inferred and hasattr(t, "genre"):
                    try:
                        t.genre = inferred
                    except Exception:
                        pass
            # 歌手名→风格映射兜底（覆盖 Drake→说唱, The Weeknd→R&B 等已知艺人）
            if not getattr(t, "genre", None):
                artist = getattr(t, "artist", "") or ""
                if artist:
                    inferred = extract_genre_from_artist(artist)
                    if inferred and hasattr(t, "genre"):
                        try:
                            t.genre = inferred
                        except Exception:
                            pass
            if not getattr(t, "mood", None):
                inferred = extract_mood(text)
                if inferred and hasattr(t, "mood"):
                    try:
                        t.mood = inferred
                    except Exception:
                        pass

    def _resolve_asset_context(self, user_id: str, query: str) -> str | None:
        # Keep only explicit/recent media context. Tool selection itself is now
        # delegated to the LangGraph planner, so this method no longer tries to infer
        # broad intent from keywords.
        if not _query_needs_asset_context(query):
            return None
        memory = self.memory.get_memory(user_id)
        if memory.listening_history:
            recent_asset_id = memory.listening_history[-1].asset_id
            if recent_asset_id:
                return recent_asset_id
        assets = self.list_assets()
        if len(assets) == 1:
            return assets[0].asset_id
        return None

    def _infer_time_bucket(self, text: str) -> str | None:
        lowered = text.lower()
        hints = {
            "morning": ["morning", "早晨", "清晨", "早餐"],
            "focus": ["focus", "专注", "工作", "学习"],
            "afternoon": ["afternoon", "午后", "白天"],
            "evening": ["evening", "晚上", "傍晚", "通勤"],
            "night": ["night", "深夜", "睡前", "夜晚"],
        }
        for bucket, keywords in hints.items():
            if any(keyword in lowered for keyword in keywords):
                return bucket
        return None

    def _safe_llm(self, prompt: str, fallback: str) -> str:
        try:
            return self.llm.generate(prompt)
        except LLMError:
            logger.debug("LLM safe call failed; using fallback", exc_info=True)
            return fallback

    def _require_segments(self, asset_id: str) -> list[Segment]:
        segments = self.media.get_segments(asset_id)
        if not segments:
            _, segments = self.analyze_media(asset_id)
        return segments


def _netease_song_id(url: str) -> str | None:
    """从各种网易云 URL 格式中提取 song id。

    支持：
      https://music.163.com/song?id=186016
      https://music.163.com/#/song?id=186016
      https://y.music.163.com/m/song/186016
      https://163cn.tv/AbCdEf  （短链，无法直接解析 id，返回 None）
    """
    return netease_song_id(url)


def _infer_playlist_count(text: str) -> int | None:
    match = re.search(r"(\d{1,3})\s*(?:首|个|tracks?|songs?)?", text, re.IGNORECASE)
    if not match:
        return None
    return max(1, min(int(match.group(1)), 100))


def get_time_bucket_name() -> str:
    """返回当前时间段中文名（用于推荐目标句）。"""
    from app.recommend.daily import get_time_bucket
    _NAMES = {"morning": "早上", "focus": "工作学习", "afternoon": "下午",
              "evening": "晚上", "night": "深夜"}
    return _NAMES.get(get_time_bucket(), "今天")


def _extract_search_query(goal: str) -> str:
    """从自然语言目标句中提取可用于音乐平台搜索的关键词。

    例如：
      "帮我推荐几首Drake的歌" → "Drake"
      "推荐一些周杰伦的歌曲" → "周杰伦"
      "Drake hip hop" → "Drake hip hop"（已是关键词，原样返回）
    """
    # 用 noise 词表清洗：中文功能词直接替换（CJK 无词边界），
    # 英文 noise 词用词边界 \b 避免误伤 "Drake" 里的 "a"。
    cleaned = goal
    # 先替换长中文功能词（避免短词先替换影响长词匹配）
    for noise in sorted(_QUERY_NOISE, key=len, reverse=True):
        if not noise or not all('一' <= c <= '鿿' or '㐀' <= c <= '䶿' for c in noise):
            continue  # 跳过英文 noise，后面用 \b 处理
        if noise in cleaned:
            cleaned = cleaned.replace(noise, " ")
    # 英文 noise 用词边界替换
    for noise in sorted(_QUERY_NOISE, key=len, reverse=True):
        if not noise or not noise.isascii():
            continue
        cleaned = re.sub(r'\b' + re.escape(noise) + r'\b', ' ', cleaned, flags=re.IGNORECASE)

    # 先尝试提取英文词（艺人名/专辑名通常是英文）
    english_tokens = re.findall(r"[A-Za-z][A-Za-z0-9'&\-]*", cleaned)
    english_tokens = [t for t in english_tokens if t.lower() not in _QUERY_NOISE and len(t) > 1]

    # 再提取中文实体词：从清洗后的文本中取连续 CJK 字符段
    cjk_tokens = re.findall(r"[一-鿿㐀-䶿豈-﫿]{2,}", cleaned)
    cjk_tokens = [t for t in cjk_tokens if t not in _QUERY_NOISE]

    candidates = english_tokens + cjk_tokens
    if candidates:
        return " ".join(candidates)
    return goal  # 兜底：返回原始 goal


def _playlist_search_terms(instruction: str) -> str:
    terms = [instruction]
    lowered = instruction.lower()
    if "chill" in lowered or "lofi" in lowered:
        terms.extend(["放松", "治愈", "浪漫", "民谣", "R&B"])
    if "跑步" in instruction or "运动" in instruction:
        terms.extend(["激昂", "热血", "电子", "摇滚"])
    if "工作" in instruction or "专注" in instruction:
        terms.extend(["放松", "宁静", "电子", "爵士"])
    return " ".join(terms)


_SCENARIO_PLAYLIST_SIGNALS = {
    "跑步", "运动", "健身", "workout", "running", "通勤", "开车", "学习", "专注",
    "工作", "睡眠", "助眠", "派对", "聚会", "散步", "旅行", "约会", "泡澡",
}


def _is_scenario_playlist_instruction(instruction: str) -> bool:
    lowered = instruction.lower()
    return any(signal in lowered for signal in _SCENARIO_PLAYLIST_SIGNALS)


def _curated_playlist_query(instruction: str) -> str:
    """把场景需求改写为歌单检索词，而不是容易被 SEO 操纵的单曲检索词。"""
    lowered = instruction.lower()
    if any(token in lowered for token in ("跑步", "运动", "健身", "running", "workout")):
        return "跑步 动感 节奏"
    if any(token in lowered for token in ("学习", "专注", "工作")):
        return "学习 专注 工作 纯音乐"
    if any(token in lowered for token in ("睡眠", "助眠", "泡澡")):
        return "睡眠 放松 舒缓"
    if any(token in lowered for token in ("开车", "通勤")):
        return "开车 通勤 节奏"
    if any(token in lowered for token in ("派对", "聚会")):
        return "派对 高能 热门"
    return _extract_search_query(instruction) or instruction


def _query_requests_variant_content(query: str) -> bool:
    """用户明确点名版本类内容时，不应用推荐质量门禁误伤。"""
    lowered = (query or "").lower()
    return any(token in lowered for token in (
        "type beat", "free beat", "伴奏", "instrumental", "demo", "翻唱", "cover",
        "remix", "纯音乐", "beat版", "beat 版",
    ))


def _is_recommendation_quality_track(track: Any, *, allow_variants: bool = False) -> bool:
    """推荐候选质量门禁；显式搜索版本内容时由 allow_variants 放行。

    这里判断的是“是否适合主动推荐”，不是歌曲是否客观存在。网易云真实 song_id
    只能证明它存在，不能证明它是正式发行、适合进入个性化歌单。
    """
    title = (getattr(track, "title", "") or "").strip()
    artist = (getattr(track, "artist", "") or "").strip()
    if not title or not artist:
        return False
    if allow_variants:
        return True

    lowered_title = title.lower()
    lowered_artist = artist.lower()
    compact_title = re.sub(r"[^a-z0-9一-鿿㐀-䶿]+", "", lowered_title)
    compact_artist = re.sub(r"[^a-z0-9一-鿿㐀-䶿]+", "", lowered_artist)

    noise_patterns = (
        r"\btype\s*beat\b", r"\bfree\s*beat\b", r"\binstrumental\b",
        r"\bbpm\s*\d+\b", r"\bprod\.?(?:\s|$)", r"\bdemo\b",
        r"\bai\s*(?:翻唱|cover|歌曲|音乐)", r"\blive\s*stage\b",
    )
    if any(re.search(pattern, lowered_title) for pattern in noise_patterns):
        return False
    if any(token in lowered_title for token in (
        "伴奏", "翻自", "动态歌词", "歌词版", "无损音质", "完整版试听",
        "步频", "卡点",
    )):
        return False
    if "#" in title or title.startswith(("【free】", "[free]", "（free）", "(free)")):
        return False

    generic_names = {
        "热门歌曲", "热门音乐", "流行音乐", "独立流行", "另类流行", "新灵魂",
        "更新灵魂", "氛围说唱", "律动rnb", "律动rb", "另类rnb", "另类rb",
        "小众rnb", "小众rb", "neosoulbeat", "neosoul", "纯音乐", "伴奏",
        "typebeat", "unknown", "佚名", "群星", "rnb", "rb", "律动", "说唱",
        "学习专注", "专注学习", "学习能量", "专注力读书音乐", "学习专注力读书音乐",
    }
    if compact_title in generic_names or compact_artist in generic_names:
        return False
    if compact_title == compact_artist:
        return False

    genre_words = (
        "r&b", "rnb", "soul", "说唱", "嘻哈", "trap", "chill", "beat",
        "另类", "流行", "爵士", "电子", "梦核", "雷鬼",
    )
    descriptor_count = sum(1 for word in genre_words if word in lowered_title)
    if descriptor_count >= 3:
        return False
    if len(title) > 38 and descriptor_count >= 2:
        return False
    return True


def _playlist_online_queries(search_terms: str) -> list[str]:
    queries = [search_terms]
    lowered = search_terms.lower()
    if "chill" in lowered or "放松" in search_terms:
        queries.extend([
            "chill R&B 放松 歌曲推荐",
            "华语 chill 放松 歌单",
            "R&B 民谣 放松 歌曲",
        ])
    if "跑步" in search_terms or "运动" in search_terms:
        queries.extend([
            "跑步 高能 歌曲推荐",
            "运动 电子 摇滚 歌单",
        ])
    unique: list[str] = []
    for query in queries:
        normalized = query.strip()
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique[:4]


def _journey_phases(instruction: str, taste: TasteProfile | None = None) -> list[dict[str, Any]]:
    """Plan an energy arc from the instruction and current taste profile.

    The stage shape is deterministic for readability; each stage carries several
    retrieval variants, and cross-journey history rotates the actual songs.
    """
    lowered = instruction.lower()
    taste = taste or TasteProfile()
    genres = [name for name, _ in taste.top_genres[:3]]
    anchor = " ".join(genres[:2])

    def phase(name: str, goal: str, energy: float, intents: list[str], transition: str) -> dict[str, Any]:
        queries = [f"{intent} {anchor}".strip() for intent in intents]
        return {
            "name": name,
            "goal": goal,
            "energy": energy,
            "query": queries[0],
            "queries": queries,
            "transition": transition,
        }

    if "清晨" in instruction and "深夜" in instruction:
        return [
            phase("清晨", "温和唤醒", 0.28, ["清晨 治愈 轻快", "晨光 温柔 放松", "早起 清新 节奏"], "从低密度、明亮的声音开始。"),
            phase("上午", "进入状态", 0.46, ["上午 专注 律动", "工作学习 稳定节奏", "通勤 清醒 groove"], "保持清醒感，逐步建立稳定拍点。"),
            phase("午后", "维持活力", 0.64, ["午后 律动 活力", "下午 groove 欢快", "白天 节奏 能量"], "中段抬高律动与辨识度。"),
            phase("傍晚", "释放张力", 0.76, ["傍晚 热血 节奏", "黄昏 高能 律动", "下班 释放 活力"], "在日落前达到整条旅程的能量峰值。"),
            phase("深夜", "放松收束", 0.32, ["深夜 放松 氛围", "夜晚 慵懒 舒缓", "午夜 安静 治愈"], "逐步降低能量与声音密度，安静落幕。"),
        ]
    if any(token in lowered or token in instruction for token in ["跑步", "运动", "running", "workout"]):
        return [
            phase("热身", "轻快进入状态", 0.45, ["热身 轻快 节奏", "运动 开场 groove", "慢跑 活力"], "从低强度节奏进入身体状态。"),
            phase("推进", "稳定耐力", 0.72, ["跑步 稳定 高能", "运动 律动 节奏", "训练 动感"], "稳定拍点，保持持续推进。"),
            phase("冲刺", "高能量峰值", 0.92, ["冲刺 高能 快节奏", "跑步 爆发 热血", "训练 峰值"], "把 BPM 和能量推到峰值。"),
            phase("放松", "降速恢复", 0.30, ["运动后 放松 舒缓", "拉伸 治愈", "恢复 安静"], "尾段降低强度，帮助恢复。"),
        ]
    return [
        phase("开场", "建立氛围", 0.35, [f"{instruction} 开场 氛围", "温和 开场", "稳定 情绪"], "先用稳定情绪铺底。"),
        phase("推进", "提升记忆点", 0.68, [f"{instruction} 推进 律动", "中段 能量", "节奏 提升"], "中段提高辨识度和情绪张力。"),
        phase("收束", "留下余韵", 0.30, [f"{instruction} 收束 放松", "结尾 舒缓", "余韵 安静"], "最后降低声音密度，留下余韵。"),
    ]


def _format_search_summary(
    query: str,
    local: list[Asset],
    external: list[ExternalTrack],
    memory_query: str,
) -> str:
    parts = [f"搜索「{query}」完成：本地 {len(local)} 首，外部候选 {len(external)} 首。"]
    if memory_query:
        parts.append(f"已结合记忆扩展：{memory_query[:80]}。")
    if local:
        parts.append("本地命中：" + "、".join(track.title for track in local[:5]) + "。")
    if external:
        verified = [track for track in external if track.source != "llm"]
        unverified = len(external) - len(verified)
        parts.append("外部候选：" + "、".join(track.title for track in external[:5]) + "。")
        if unverified:
            parts.append(f"其中 {unverified} 首为 LLM 补充候选，尚未真实回查。")
    return " ".join(parts)


def _generic_metadata_title(title: str | None) -> bool:
    if not title:
        return True
    normalized = title.strip().lower()
    generic = {
        "网易云音乐",
        "qq音乐",
        "bilibili",
        "哔哩哔哩",
        "youtube",
        "cinesonic demo asset",
    }
    return normalized in {item.lower() for item in generic} or normalized.startswith("网易云音乐 -")


def _has_reliable_metadata(asset: Asset) -> bool:
    if _generic_metadata_title(asset.title):
        return False
    if asset.title.startswith("网易云歌曲 ") or asset.title == "CineSonic Demo Asset":
        return False
    return bool(asset.artist or asset.album or asset.genre or asset.mood)


def _query_needs_asset_context(query: str) -> bool:
    lowered = query.lower()
    media_terms = [
        "片段", "segment", "video", "素材", "场景", "镜头", "画面",
        "当前视频", "当前素材", "这个视频", "这个素材", "相似片段",
    ]
    return any(term in lowered for term in media_terms)


def _playlist_match_score(track: Asset | ExternalTrack, query: str) -> int:
    searchable = (
        f"{track.title} {getattr(track, 'artist', '') or ''} "
        f"{' '.join(getattr(track, 'genre', []) or [])} "
        f"{' '.join(getattr(track, 'mood', []) or [])}"
    ).lower()
    score = 0
    for term in query.lower().split():
        if term and term in searchable:
            score += 1
    return score


def _track_key(track: Asset | ExternalTrack | dict[str, Any]) -> str:
    if isinstance(track, Asset):
        return f"asset:{track.asset_id}"
    if isinstance(track, ExternalTrack):
        if track.external_id:
            return f"{track.source}:{track.external_id}"
        return f"title:{track.title.lower()}:{track.artist.lower()}"
    title = str(track.get("title", "")).lower().strip()
    artist = str(track.get("artist", "")).lower().strip()
    aid = str(track.get("asset_id", "")).strip()
    return f"asset:{aid}" if aid else f"title:{title}:{artist}"


def _is_verified_online_track(track: Asset | ExternalTrack) -> bool:
    """验证是否为真实线上曲目。纯歌曲推荐只认网易云；视频源仅 MV 搜索时使用。"""
    return isinstance(track, ExternalTrack) and track.source == "netease"


def _is_local_recommendation_track(track: Asset | ExternalTrack) -> bool:
    return isinstance(track, Asset) or (
        isinstance(track, ExternalTrack)
        and track.source == "local"
        and bool(track.external_id or track.playback_url)
    )


def _is_verified_recommendation_track(track: Asset | ExternalTrack) -> bool:
    return _is_local_recommendation_track(track) or _is_verified_online_track(track)


def _is_fallback_track(track: Asset | ExternalTrack) -> bool:
    source = getattr(track, "source", "local")
    return "fallback" in source or source in {"mock", "llm"}


# 推荐卡片允许出现的候选类型：单曲 + 可绑定歌曲的官方 MV。
# unknown 暂时保留（B站/YouTube 兜底里大量无信号标题多半是单曲，
# 全判 unknown 会掏空兜底），噪声四类 playlist/compilation/long_mix/
# lyrics_video 一律过滤。
_ALLOWED_CANDIDATE_KINDS = {"track", "official_mv", "unknown"}


def _classify_candidate_kind(title: str, source: str) -> str:
    """根据标题判断候选类型，七分类：

    track / official_mv / lyrics_video / playlist / compilation / long_mix / unknown

    B站/YouTube 大量返回「合集/连播/串烧/歌单/动态歌词/长混音」类视频，
    这些不是单曲，混进推荐会严重退化体验（如「推荐 The Weeknd」只出合集）。
    用关键词识别，噪声类在 _valid_external_track 中被丢弃；track 与
    official_mv（可绑定歌曲的官方 MV/现场）保留进卡片。
    检查顺序：最具体的噪声类优先，避免被泛词（如 mix 命中 official video）误判。
    """
    t = (title or "").lower()

    # 1. 动态歌词 / 歌词版视频
    lyrics_signals = ["动态歌词", "歌词版", "歌词视频", "lyric video", "lyrics video", "(lyrics)", "[lyrics]"]
    if any(sig in t for sig in lyrics_signals):
        return "lyrics_video"

    # 2. 歌单类（整张歌单/榜单，不是单曲）
    playlist_signals = ["歌单", "playlist", "排行榜", "top chart", "网易云歌单"]
    if any(sig in t for sig in playlist_signals):
        return "playlist"

    # 3. 长混音 / 连续播放（DJ mix、N 小时连播）；排除单曲 Remix
    long_mix_signals = [
        "non-stop", "nonstop", "megamix", "mega mix", "dj mix",
        "连续播放", "一直播放", "纯音乐合集", "睡眠歌单",
    ]
    if "remix" not in t:
        if any(sig in t for sig in long_mix_signals):
            return "long_mix"
        # 以 "mix" 作词收尾的整段混音（"... EDM Mix"），但单个单词 remix 已排除
        if re.search(r"\bmix\b\s*\d*\)?\s*$", t):
            return "long_mix"
    if re.search(r"\d+\s*(?:小时|hours?|hrs?)\b", t):
        return "long_mix"

    # 4. 合集 / 连播 / 串烧 / 精选集 / Greatest Hits / Full Album
    compilation_signals = [
        "合集", "连播", "串烧", "歌曲合集", "精选集", "全部歌曲",
        "全部曲目", "经典回顾", "最全", "歌曲大全", "纯享合集", "金曲合集",
        "full album", "all songs", "greatest hits", "compilation",
        "best of", "歌曲串烧",
    ]
    if any(sig in t for sig in compilation_signals):
        return "compilation"
    # 数量型信号：标题里出现「N首」「N songs」暗示连播合集
    if re.search(r"\d+\s*首", t) or re.search(r"\d+\s*songs?\b", t):
        return "compilation"

    # 5. 官方 MV / 现场（可播，可绑定单曲，保留）
    mv_signals = ["mv", "live", "现场", "演唱会", "官方视频", "official video", "official music video", "music video"]
    if any(sig in t for sig in mv_signals):
        return "official_mv"

    return "track"


# 搜索噪声过滤：中文停用词不作为相关性判据（如"的""歌""我"等）
_QUERY_NOISE = {"的", "了", "在", "是", "我", "你", "他", "她", "它", "和", "与", "或",
                "歌", "曲", "音乐", "首", "些", "几", "个", "找", "要", "想", "帮", "给",
                "推荐", "适合", "播放", "听", "下", "不", "也", "都", "就", "还", "又",
                "几首", "一些", "几个", "来几", "来首", "我想", "帮我", "给我", "来点",
                "好听", "推一", "推几", "推些", "介绍", "分享", "列举", "一下",
                # 常见功能词/动词（之前缺失导致变成相关性过滤 token 杀死搜索结果）
                "生成", "只要", "其他", "不要", "别的", "还有", "有没有",
                "可以", "能", "会", "让", "从", "到", "去", "来", "上", "这", "那",
                "什么", "怎么", "哪些", "如何", "为什么", "多少", "很多", "比较",
                "一点", "稍微", "偏", "微", "更",
                "帮我搜索", "帮我找", "给我推荐", "帮我推荐", "来几首", "弄几首",
                "做", "弄", "搞", "弄个", "做个", "生成个",
                "songs", "song", "music", "me", "some", "please",
                "a", "an", "the", "of", "in", "on", "is", "to", "for", "my", "and",
                "or", "it", "s", "t", "m"}


def _normalize_match_text(text: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKC", text or "").lower()
    normalized = normalized.replace("r&b", "rnb").replace("r and b", "rnb").replace("hip-hop", "hiphop")
    return re.sub(r"[^a-z0-9一-鿿㐀-䶿]+", "", normalized)


_ARTIST_ALIAS_STOPWORDS = {
    "the", "and", "band", "music", "official", "feat", "featuring",
    "with", "from", "west",  # 方位/连接词单独出现时不应触发歌手页
}


def _artist_credit_parts(artist: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"[、,/;&]|\b(?:feat\.?|featuring|with)\b", artist or "", flags=re.IGNORECASE)
        if part.strip()
    ]


def _artist_alias_keys(artist: str) -> set[str]:
    """Build conservative aliases from one credited artist string.

    ``Kanye West、Ye`` yields the full names plus ``kanye`` and ``ye``;
    Chinese names remain whole to avoid turning each character into an alias.
    """
    aliases: set[str] = set()
    for part in _artist_credit_parts(artist):
        full_key = _normalize_match_text(part)
        if full_key:
            aliases.add(full_key)
        cleaned_key = _normalize_match_text(_extract_search_query(part))
        if cleaned_key:
            aliases.add(cleaned_key)
        if re.search(r"[A-Za-z]", part):
            for token in re.findall(r"[A-Za-z0-9]+", part.lower()):
                if len(token) >= 3 and token not in _ARTIST_ALIAS_STOPWORDS:
                    aliases.add(_normalize_match_text(token))
    return aliases


def _artist_query_matches(query: str, artist: str, *, allow_fuzzy: bool = False) -> bool:
    query_key = _normalize_match_text(_extract_search_query(query) or query)
    if not query_key or query_key in _ARTIST_ALIAS_STOPWORDS:
        return False
    aliases = _artist_alias_keys(artist)
    if query_key in aliases:
        return True
    return bool(
        allow_fuzzy
        and len(query_key) >= 6
        and max((_string_similarity(query_key, alias) for alias in aliases), default=0.0) >= 88.0
    )


def _string_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    try:
        from rapidfuzz import fuzz

        return float(fuzz.ratio(a, b))
    except Exception:
        from difflib import SequenceMatcher

        return SequenceMatcher(None, a, b).ratio() * 100.0


def _fuzzy_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    try:
        from rapidfuzz import fuzz

        return float(fuzz.partial_ratio(a, b))
    except Exception:
        from difflib import SequenceMatcher

        return SequenceMatcher(None, a, b).ratio() * 100.0


def _match_token(token: str, text: str, *, fuzzy: bool = False) -> bool:
    token_norm = _normalize_match_text(token)
    text_norm = _normalize_match_text(text)
    if not token_norm:
        return True
    if token_norm in text_norm:
        return True
    if fuzzy and len(token_norm) >= 3:
        return _fuzzy_ratio(token_norm, text_norm) >= settings.fuzzy_threshold
    return False


def _query_matches_track(query: str, track: ExternalTrack) -> bool:
    """搜索相关性过滤：基于 entity 锚点的宽松匹配。

    网易云搜索 API 做全字段模糊匹配（歌词/评论/标签都会命中），
    导致搜 "Drake" 返回一堆歌名里根本没有 Drake 的中文歌。

    策略（v2 — 精排友好型）：
    1. 将 token 分为 entity 类（英文专有名词 >2 字符）和泛化类（CJK/短 token）。
    2. 如果至少 1 个 entity 类 token 命中了 title+artist → 放行。
       entity 是锚点（歌手名），歌曲标题不含歌手名是正常的（如 "God's Plan"），
       相关性交给下游三锚精排处理。
    3. 如果 0 个 entity 命中 → 保持严格：所有 token 必须命中（防止 API 垃圾）。
    """

    def _split_tokens(text: str) -> list[str]:
        # 先按空格/标点拆，再对每段按 ASCII/CJK 边界二次拆分
        # 例如 "帮我推荐几首Drake的歌" → ["Drake"] (中文功能词被噪声表过滤掉)
        raw = re.split(r"[\s,，、·\-|/\\]+", text.strip())
        result = []
        for seg in raw:
            # 把连续 ASCII 字符和连续 CJK 字符分开
            sub_tokens = re.findall(r"[A-Za-z]+&[A-Za-z]+|[A-Za-z0-9]+|[一-鿿㐀-䶿豈-﫿]+", seg)
            result.extend(sub_tokens if sub_tokens else [seg])
        return result

    tokens = _split_tokens(query)
    tokens = [t for t in tokens if t and t.lower() not in _QUERY_NOISE and len(t) > 1]
    if not tokens:
        return True  # 纯噪声查询不过滤

    searchable = f"{(track.title or '')} {(track.artist or '')}"
    searchable_parts = [track.title or "", track.artist or "", searchable]

    # 分离 entity 类 token（英文专有名词）和泛化类 token（CJK + 短英文）
    entity_tokens: list[str] = []  # 英文 >2 字符，可能是歌手名/乐队名
    general_tokens: list[str] = []  # CJK token 或短英文
    for token in tokens:
        lowered = token.lower()
        is_ascii = bool(re.fullmatch(r"[A-Za-z0-9&]+", token))
        if is_ascii and len(token) > 2:
            entity_tokens.append(lowered)
        else:
            general_tokens.append(token)

    # 检查 entity 类 token 是否命中
    entity_hit = any(
        _match_token(e, part, fuzzy=True)
        for e in entity_tokens
        for part in searchable_parts
    )

    # 如果有 entity 命中 → 放行（entity 是歌手名锚点，歌曲标题不含歌手名是正常的）
    if entity_hit:
        return True

    # 无 entity 命中：泛化类 token 全部必须命中（保持严格，防止 API 垃圾）
    if not general_tokens:
        # 只有 entity token 但都没命中 → 拒绝
        return False

    for token in general_tokens:
        token_is_ascii = bool(re.fullmatch(r"[A-Za-z0-9&]+", token))
        if not any(_match_token(token, part, fuzzy=token_is_ascii) for part in searchable_parts):
            return False
    return True


def _valid_external_track(track: ExternalTrack, query: str) -> bool:
    title = (track.title or "").strip()
    if not title:
        return False
    lowered_title = title.lower()
    lowered_query = query.lower().strip()
    if lowered_title == lowered_query:
        return False
    if lowered_title in {"网易云音乐", "bilibili", "youtube", "搜索结果"}:
        return False
    if len(title) > 80 and " - " not in title:
        return False
    # 合集/连播/歌单/动态歌词/长混音类视频污染推荐，只保留单曲/官方MV/未知兜底。
    if getattr(track, "candidate_kind", "track") not in _ALLOWED_CANDIDATE_KINDS:
        return False
    # 相关性过滤：搜索词的每个显著词元必须至少命中 title 或 artist 之一。
    # 否则网易云模糊搜索会返回大量歌词/评论里提到关键词但实际无关的歌曲。
    if not _query_matches_track(query, track):
        return False
    return True


def _online_candidate_reason(track: ExternalTrack, memory_query: str) -> str:
    source_label = {
        "netease": "网易云真实曲目",
        "bilibili": "B 站真实视频/MV",
        "youtube": "YouTube 真实视频",
    }.get(track.source, "真实线上候选")
    if memory_query:
        return f"online_candidate：来自{source_label}，并结合你的记忆偏好「{memory_query[:40]}」排序。"
    return f"online_candidate：来自{source_label}，不是本地 mock 结果。"


def _dedupe_tracks(tracks: list[Asset | ExternalTrack]) -> list[Asset | ExternalTrack]:
    seen: set[str] = set()
    seen_titles: set[str] = set()
    unique: list[Asset | ExternalTrack] = []
    for track in tracks:
        key = _track_key(track)
        title_key = f"{track.title.strip().lower()}|{(track.artist or '').strip().lower()}"
        if key in seen or title_key in seen_titles:
            continue
        seen.add(key)
        seen_titles.add(title_key)
        unique.append(track)
    return unique


def _merge_search_queries(query: str, variants: list[str] | None = None) -> list[str]:
    """主查询 + query_plan 变体合并去重，限制总变体数避免无界外部请求。"""
    out: list[str] = []
    seen: set[str] = set()
    for item in [query, *(variants or [])]:
        value = (item or "").strip()
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        out.append(value)
        if len(out) >= settings.max_search_variants + 1:
            break
    return out


def _filter_excluded_tracks(
    tracks: list[Asset | ExternalTrack],
    excluded: list[dict[str, str]],
) -> list[Asset | ExternalTrack]:
    """过滤掉上一轮已展示给用户的歌曲（延续指令去重）。

    匹配策略：(title.lower, source_id) 组合键；source_id 为空时退化为 title。
    """
    if not excluded:
        return tracks
    seen_keys: set[tuple[str, str]] = set()
    seen_titles: set[str] = set()
    for ex in excluded:
        title = ex.get("title", "").lower().strip()
        sid = ex.get("source_id", "").strip()
        if title:
            seen_titles.add(title)
            if sid:
                seen_keys.add((title, sid))
    filtered: list[Asset | ExternalTrack] = []
    for t in tracks:
        t_title = (getattr(t, "title", "") or "").lower().strip()
        t_sid = getattr(t, "external_id", "") or getattr(t, "asset_id", "") or ""
        if t_title and t_sid and (t_title, t_sid) in seen_keys:
            continue
        if t_title and t_title in seen_titles:
            continue
        filtered.append(t)
    return filtered


def _fill_tracks(
    tracks: list[Asset | ExternalTrack],
    candidates: list[Asset | ExternalTrack],
    target_count: int,
) -> list[Asset | ExternalTrack]:
    merged = _dedupe_tracks(tracks)
    seen = {_track_key(track) for track in merged}
    for candidate in candidates:
        if len(merged) >= target_count:
            break
        key = _track_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        merged.append(candidate)
    return merged[:target_count]


# 向后兼容别名
CineSonicAgent = AudioVisualAgent
