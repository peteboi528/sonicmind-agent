from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
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
    SearchResponse,
    Segment,
    SimilarAssetResult,
    SimilarSegmentResult,
    TasteProfile,
    UserMemory,
    utc_now_iso,
)
from app.prompts import (
    AUTO_PLAYLIST_TEMPLATE,
    GENERATE_PLAYLIST_TEMPLATE,
    IDENTIFY_FROM_URL_TEMPLATE,
    LLM_SEARCH_TEMPLATE,
)
from app.react_loop import ReActLoop
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
        self.store = store or JsonStore()
        self.media = MediaPipeline(self.store)
        self.memory = MemoryManager(self.store)
        self.similarity = AssetSimilarity(self.store)
        self.library = ResourceLibrary(settings.resource_library_path)
        self.llm: LLMProvider = build_llm()
        self.source: ExternalSource = _build_source()
        self.engine = RecommendEngine()
        self.daily = DailyRecommender(self.engine, self.source, self.llm)
        self.react = ReActLoop(self)
        self.graph = None
        self.library.sync_assets(self.list_assets())
        try:
            from app.graph.builder import build_agent_graph
            self.graph = build_agent_graph(self)
        except Exception:
            logger.debug("LangGraph wrapper unavailable; using ReAct fallback", exc_info=True)

    def ingest_video(self, url: str, force_refresh: bool = False) -> Asset:
        return self.media.ingest_video(url, force_refresh=force_refresh)

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
        except Exception:
            logger.debug("URL identity inference failed for asset_id=%s", asset.asset_id, exc_info=True)

    def analyze_media(self, asset_id: str, force_refresh: bool = False) -> tuple[Asset, list[Segment]]:
        asset, segments = self.media.analyze_media(asset_id, force_refresh=force_refresh)
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
        keys = self.store.list_keys("assets")
        assets: list[Asset] = []
        for key in keys:
            asset = self.store.read_model("assets", key, Asset)
            if asset:
                assets.append(asset)
        return assets

    def delete_asset(self, asset_id: str, user_id: str | None = None) -> bool:
        deleted_asset = self.store.delete_key("assets", asset_id)
        deleted_segments = self.store.delete_key("segments", asset_id)
        deleted = deleted_asset or deleted_segments
        if deleted:
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
        if not preserve_memory:
            cleared["memory"] = self.store.clear_collection("memory")
        return cleared

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

    def search(self, user_id: str, query: str, include_external: bool = True, top_k: int = 20) -> SearchResponse:
        memory = self.memory.get_memory(user_id)
        memory_query = self.memory.weighted_query(memory)
        expanded_query = f"{query} {memory_query}".strip()
        search_goal = _extract_search_query(query)
        # 本地搜索：只用核心搜索词（search_goal），不用 memory 扩展词。
        # 之前用 expanded_query.split() 做 any() 匹配，memory 的风格标签如
        # "说唱""R&B""chill" 会匹配库里几乎所有歌，淹没真正相关的结果。
        local_terms = search_goal.lower().split() if search_goal else query.lower().split()
        local_results: list[Asset] = []
        for asset in self.list_assets():
            searchable = f"{asset.title} {asset.artist or ''} {' '.join(asset.genre)} {' '.join(asset.mood)}".lower()
            if any(term in searchable for term in local_terms):
                local_results.append(asset)

        evidences = self.retrieve_library_evidence(expanded_query, top_k=min(top_k, 6))
        local_by_id = {asset.asset_id: asset for asset in local_results}
        for evidence in evidences:
            asset_id = str(evidence.metadata.get("asset_id", ""))
            asset = next((item for item in self.list_assets() if item.asset_id == asset_id), None)
            if asset is not None:
                local_by_id.setdefault(asset.asset_id, asset)

        external_results: list[ExternalTrack] = []
        if include_external:
            # 用 expanded_query 搜索拿更广结果，但相关性过滤用核心词 search_goal
            external_results = self.search_web_music(
                expanded_query, top_k=top_k, relevance_query=search_goal,
            )

        summary = _format_search_summary(
            query=query,
            local=list(local_by_id.values())[:top_k],
            external=external_results[:top_k],
            memory_query=memory_query,
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

    def search_web_music(self, query: str, top_k: int = 5, relevance_query: str = "", include_video_sources: bool = False) -> list[ExternalTrack]:
        """Agent tool wrapper for explicit online search.

        The default product flow remains offline-first. This method is only
        called when the ReAct loop decides the user needs real platform data.
        每个候选都必须回查到真实曲目元数据；回查失败的候选直接丢弃，
        绝不把搜索词 query 当成歌名返回（这是幻觉的主要来源之一）。

        Args:
            query: 传给搜索 API 的完整查询词（可含 memory 扩展词，获取更广结果）。
            top_k: 目标候选数量。
            relevance_query: 相关性过滤用的核心查询词。为空时默认等于 query。
            include_video_sources: 是否包含 B站/YouTube 视频源。默认 False，
                只返回网易云歌曲。用户明确要 MV/视频时才传 True。
        """
        tracks: list[ExternalTrack] = []

        # 网易云为主候选源：用多结果搜索拿真实歌曲（之前只取 1 首，导致大量缺口
        # 被 B站/YouTube 的合集视频/SEO 垃圾填补，搜索质量差）。
        try:
            from app.sources.netease import search_netease_many
            for meta in search_netease_many(query, limit=top_k):
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

    def search_videos(self, query: str, top_k: int = 5) -> list[ExternalTrack]:
        """搜索 MV/现场/演唱会视频，B站优先、YouTube 补位。不走网易云。

        用于 video 意图：用户明确要 MV/现场/Live 视频时调用。
        """
        tracks: list[ExternalTrack] = []

        # B站优先：华语 MV/现场命中率高，嵌入稳定
        try:
            bili_results = bilibili_source.search_bilibili_many(query, limit=min(top_k, 5))
            for item in bili_results:
                tracks.append(ExternalTrack(
                    external_id=item["bvid"],
                    title=item["title"],
                    artist=item.get("author", ""),
                    source="bilibili",
                    candidate_kind=_classify_candidate_kind(item["title"], "bilibili"),
                    playback_url=f"https://player.bilibili.com/player.html?bvid={item['bvid']}&autoplay=0&high_quality=1&danmaku=0",
                ))
        except Exception:
            logger.debug("Bilibili video search failed for query=%s", query, exc_info=True)

        # YouTube 补位：国际音乐覆盖
        remaining = max(top_k - len(tracks), 0)
        if remaining > 0:
            try:
                yt_results = youtube_source.search_youtube_many(query, limit=min(remaining, 3))
                for item in yt_results:
                    vid = item["video_id"]
                    title = item.get("title") or youtube_source.fetch_youtube_title(
                        f"https://www.youtube.com/watch?v={vid}"
                    ) or ""
                    tracks.append(ExternalTrack(
                        external_id=vid,
                        title=title,
                        artist="",
                        source="youtube",
                        candidate_kind=_classify_candidate_kind(title, "youtube"),
                        playback_url=f"https://www.youtube.com/embed/{vid}?autoplay=1&rel=0",
                    ))
            except Exception:
                logger.debug("YouTube video search failed for query=%s", query, exc_info=True)

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

    def search_artist_info(self, query: str) -> list[dict[str, str]]:
        """用 Tavily/DuckDuckGo 搜索歌手/乐队百科信息。

        用于 artist_info 意图：用户要了解歌手背景时调用。
        返回 [{"title": ..., "content": ..., "url": ...}] 搜索摘要列表。
        """
        return web_search_source.search_web_info(
            query, max_results=5, api_key=settings.tavily_api_key,
        )

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

    def chat(self, user_id: str, message: str, history: list[dict[str, Any]] | None = None) -> AgentAnswer:
        asset_id = self._resolve_asset_context(user_id, message)
        if self.graph is not None:
            try:
                return self.graph.invoke(user_id=user_id, asset_id=asset_id, query=message, history=history, top_k=5)
            except Exception:
                logger.debug("LangGraph invoke failed; falling back to ReActLoop", exc_info=True)
        return self.react.run(user_id=user_id, asset_id=asset_id, query=message, top_k=5, history=history)

    def stream_chat(self, user_id: str, message: str, history: list[dict[str, Any]] | None = None):
        asset_id = self._resolve_asset_context(user_id, message)
        if self.graph is not None:
            yield from self.graph.stream(user_id=user_id, asset_id=asset_id, query=message, history=history, top_k=5)
            return
        answer = self.react.run(user_id=user_id, asset_id=asset_id, query=message, top_k=5, history=history)
        from app.models import StreamEvent
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
        self.library.sync_assets(self.list_assets())
        return self.library.list_tracks(limit)

    def generate_music_journey(self, user_id: str, instruction: str) -> dict[str, Any]:
        phases = _journey_phases(instruction)
        out = {"user_id": user_id, "instruction": instruction, "phases": []}
        for phase in phases:
            query = f"{instruction} {phase['query']}"
            candidates = [
                track for track in self.search_web_music(query, top_k=8)
                if not self.library.is_disliked(user_id, track)
            ]
            self.library.record_exposure(candidates[:3])
            out["phases"].append({
                "name": phase["name"],
                "goal": phase["goal"],
                "transition": phase["transition"],
                "tracks": [track.model_dump(mode="json") for track in candidates[:3]],
            })
        return out

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
        taste_summary = self.summarize_taste(user_id) if memory.taste_profile else ""
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
        tracks = _fill_tracks(tracks, candidates, target_count)

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
        external: list[ExternalTrack] = []
        for online_query in _playlist_online_queries(search_terms):
            if len(_dedupe_tracks([*seed_tracks, *external])) >= target_count:
                break
            # 关键修复：用核心词做相关性过滤，扩展词只用于引导搜索API
            external.extend(self.search_web_music(
                online_query, top_k=min(max(target_count, 8), 25),
                relevance_query=relevance_core,
            ))

        if len(_dedupe_tracks([*seed_tracks, *external])) < target_count:
            external.extend(
                self.source.get_recommendations(
                    seed_genres=["流行", "民谣", "R&B", "说唱", "电子"],
                    seed_moods=["放松", "治愈", "浪漫", "伤感"],
                    limit=max(target_count * 2, 40),
                )
            )

        library_ranked = sorted(
            library,
            key=lambda asset: _playlist_match_score(asset, search_terms),
            reverse=True,
        )
        ordered: list[Asset | ExternalTrack] = [*seed_tracks, *external, *library_ranked]
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
        tracks = _fill_tracks(matched, candidates or [], target_count)
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
            ranked.extend(self.retrieve_evidence(asset.asset_id, query, top_k=min(3, top_k)))
        ranked.sort(key=lambda evidence: evidence.similarity, reverse=True)
        return ranked[:top_k]

    def recommend_with_memory(self, asset_id: str, user_id: str, goal: str, top_k: int = 3) -> AgentAnswer:
        memory = self.memory.get_memory(user_id)
        memory_query = self.memory.weighted_query(memory)
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

    def summarize_taste(self, user_id: str) -> str:
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
        if artist_text:
            parts.append(f"偏好的艺人有 {artist_text}，")
        parts.append(f"显式表达过的偏好包括 {pref_text}。")
        return "".join(parts)

    def recommend_for_query(self, user_id: str, goal: str, top_k: int = 5, *, excluded_tracks: list[dict[str, str]] | None = None) -> DailyRecommendation:
        memory = self.memory.get_memory(user_id)
        memory_query = self.memory.weighted_query(memory)
        search_goal = _extract_search_query(goal)
        taste_summary = self.summarize_taste(user_id) if memory.taste_profile else ""
        library_artists = list({a.artist for a in self.list_assets() if a.artist})[:10]

        # ── 三路搜索策略 ──
        # 精确实体查询 (有歌手/歌名) → 网易云歌曲搜索
        # 情绪/场景/模糊查询 → LLM 候选生成 + 网易云歌单搜索
        trace_lines: list[str] = []
        all_candidates: list[ExternalTrack] = []

        has_entity = self._query_has_entity(search_goal)

        if has_entity:
            # 路由C：精确搜索（网易云歌曲搜索，这个是OK的）
            trace_lines.append(f"route=exact, search_goal={search_goal}")
            batch = self.search_web_music(search_goal, top_k=max(top_k * 2, top_k), relevance_query=search_goal)
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

        # 去重 + 过滤
        verified = [
            track for track in _dedupe_tracks(all_candidates)
            if _is_verified_online_track(track) and not self.library.is_disliked(user_id, track)
        ]

        # 过滤上一轮已展示的曲目（延续指令去重）
        if excluded_tracks:
            verified = _filter_excluded_tracks(verified, excluded_tracks)

        # 兜底：用 search_goal 再搜一次
        if len(verified) < top_k and search_goal:
            fallback_batch = self.search_web_music(search_goal, top_k=max(top_k * 2, top_k))
            for track in fallback_batch:
                if _is_verified_online_track(track) and not self.library.is_disliked(user_id, track):
                    if not any(_track_key(track) == _track_key(v) for v in verified):
                        verified.append(track)

        if verified:
            rerank_query = search_goal or goal
            ranked = self._rerank_tracks(user_id, rerank_query, _dedupe_tracks(verified), top_k)
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
            return DailyRecommendation(
                user_id=user_id,
                tracks=tracks,
                reason_summary=f"采用 {len(tracks)} 首真实线上候选（LLM候选+歌单搜索+网易云验证），经三锚精排+MMR多样性重排。",
                agent_trace=[
                    *trace_lines,
                    f"online_verified={len(verified)}",
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

        # 英文：排除纯风格/情绪词
        generic_en = {"chill", "lofi", "vibe", "mix", "remix", "relax", "mood", "groove",
                       "upbeat", "slow", "fast", "happy", "sad", "deep", "party",
                       "r&b", "soul", "pop", "rock", "rap", "hip", "hop", "jazz",
                       "electronic", "ambient", "acoustic", "indie", "funk",
                       "morning", "night", "evening", "summer", "winter",
                       "playlist", "songs", "music", "recommend"}
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
            "健身", "旅行", "约会", "散步", "泡澡",
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
        return rerank_candidates(
            query, tracks, taste,
            behavior_scores=behavior, scenarios=scenarios, top_k=top_k,
            lang_pref=lang_pref, exclusion_rules=exclusion_rules,
        )

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
        # delegated to the ReAct loop, so this method no longer tries to infer
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


def _journey_phases(instruction: str) -> list[dict[str, str]]:
    lowered = instruction.lower()
    if any(token in lowered or token in instruction for token in ["跑步", "运动", "running", "workout"]):
        return [
            {"name": "热身", "goal": "轻快进入状态", "query": "热身 轻快 节奏", "transition": "从低强度节奏进入身体状态。"},
            {"name": "冲刺", "goal": "高能量推进", "query": "跑步 高能 快节奏", "transition": "中段提升 BPM 和能量，适合冲起来。"},
            {"name": "放松", "goal": "降速恢复", "query": "运动后 放松 舒缓", "transition": "尾段降低强度，帮助恢复。"},
        ]
    return [
        {"name": "开场", "goal": "建立氛围", "query": "开场 氛围 音乐", "transition": "先用稳定情绪铺底。"},
        {"name": "推进", "goal": "提升记忆点", "query": "高潮 推荐 音乐", "transition": "中段提高辨识度和情绪张力。"},
        {"name": "收束", "goal": "留下余韵", "query": "结尾 放松 音乐", "transition": "最后用更耐听的曲目收尾。"},
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
                "帮我搜索", "帮我找", "给我推荐", "帮我推荐", "来几首", "弄几首",
                "做", "弄", "搞", "弄个", "做个", "生成个",
                "songs", "song", "music", "me", "some", "please",
                "a", "an", "the", "of", "in", "on", "is", "to", "for", "my", "and",
                "or", "it", "s", "t", "m"}


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
            sub_tokens = re.findall(r"[A-Za-z0-9]+|[一-鿿㐀-䶿豈-﫿]+", seg)
            result.extend(sub_tokens if sub_tokens else [seg])
        return result

    tokens = _split_tokens(query)
    tokens = [t for t in tokens if t and t.lower() not in _QUERY_NOISE and len(t) > 1]
    if not tokens:
        return True  # 纯噪声查询不过滤

    searchable = f"{(track.title or '')} {(track.artist or '')}".lower()

    # 分离 entity 类 token（英文专有名词）和泛化类 token（CJK + 短英文）
    entity_tokens: list[str] = []  # 英文 >2 字符，可能是歌手名/乐队名
    general_tokens: list[str] = []  # CJK token 或短英文
    for token in tokens:
        lowered = token.lower()
        is_ascii = bool(re.fullmatch(r"[A-Za-z0-9]+", token))
        if is_ascii and len(token) > 2:
            entity_tokens.append(lowered)
        else:
            general_tokens.append(token)

    # 检查 entity 类 token 是否命中
    entity_hit = any(e in searchable for e in entity_tokens)

    # 如果有 entity 命中 → 放行（entity 是歌手名锚点，歌曲标题不含歌手名是正常的）
    if entity_hit:
        return True

    # 无 entity 命中：泛化类 token 全部必须命中（保持严格，防止 API 垃圾）
    if not general_tokens:
        # 只有 entity token 但都没命中 → 拒绝
        return False

    for token in general_tokens:
        if token.lower() not in searchable:
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
    unique: list[Asset | ExternalTrack] = []
    for track in tracks:
        key = _track_key(track)
        if key in seen:
            continue
        seen.add(key)
        unique.append(track)
    return unique


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
