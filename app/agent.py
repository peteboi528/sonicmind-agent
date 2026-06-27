from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.catalog import CatalogService
from app.rules.discover import (
    _QUERY_NOISE,
    _artist_alias_keys,
    _artist_credit_parts,
    _artist_query_matches,
    _curated_playlist_query,
    _extract_search_query,
    _format_search_summary,
    _is_scenario_playlist_instruction,
    _journey_phases,
    _local_ratio_from_query,
    _looks_like_bare_artist_query,
    _normalize_match_text,
    _playlist_online_queries,
    _playlist_search_terms,
    _query_matches_track,
    _query_requests_variant_content,
    _scene_playlist_queries,
    _string_similarity,
)
from app.services.discover import DiscoverService
from app.services.journey import JourneyService
from app.library import ResourceLibrary
from app.services.library import LibraryService
from app.llm.client import build_llm
from app.llm.protocol import LLMError, LLMProvider
from app.llm.structured import extract_json_dict, extract_json_list
from app.media.pipeline import MediaPipeline
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
from app.services.playlist import PlaylistService
from app.services.playback import PlaybackService
from app.prompts import (
    IDENTIFY_FROM_URL_TEMPLATE,
    LLM_SEARCH_TEMPLATE,
)
from app.recommend.daily import DailyRecommender
from app.recommend.engine import RecommendEngine
from app.rules.recommend import (
    RecommendationAnchors,
    _extract_recommendation_anchors,
    _infer_playlist_count,
    _is_playlist_context_compatible,
    _is_recommendation_quality_track,
    _netease_song_id,
    _recommendation_search_seeds,
    _track_matches_recommendation_anchors,
    get_time_bucket_name,
)
from app.services.recommend import RecommendationService
from app.recommend.source_balance import balance_recommendation_sources
from app.retrieval.vector_store import HybridRetriever
from app.services.search import SearchService
from app.similarity import AssetSimilarity
from app.sources import bilibili as bilibili_source
from app.sources import netease as netease_source
from app.sources import web_search as web_search_source
from app.sources import youtube as youtube_source
from app.sources.mock_source import MockSource
from app.sources.netease_source import NeteaseSource
from app.sources.protocol import ExternalSource
from app.storage import JsonStore
from app.services.taste_experiment import TasteExperimentService
from app.rules.taste_experiment import (
    apply_taste_experiment_ts_feedback,
    bucket_label,
    bucket_taste_experiment_candidates,
    candidate_key,
    filter_taste_experiment_candidates,
    find_taste_experiment_track,
    record_taste_experiment_listen,
    slice_for_bucket,
    taste_experiment_bucket_stats,
    taste_experiment_feedback_count,
    taste_experiment_track_key,
    taste_familiarity,
)
from app.rules.track import (
    _classify_candidate_kind,
    _dedupe_tracks,
    _fill_tracks,
    _filter_excluded_tracks,
    _generic_metadata_title,
    _has_reliable_metadata,
    _is_fallback_track,
    _is_local_recommendation_track,
    _is_verified_online_track,
    _is_verified_recommendation_track,
    _merge_search_queries,
    _online_candidate_reason,
    _playlist_match_score,
    _query_needs_asset_context,
    _track_key,
    _valid_external_track,
)

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
        self.media = MediaPipeline(self.store)
        self.memory = MemoryManager(self.store)
        self.similarity = AssetSimilarity(self.store)
        # 注入临时 store（测试/隔离运行）时，资源库必须落在该 store 自己的目录内，实现真正
        # 按测试隔离。曾用 .parent 导致所有 tmp store 共享同一个 /tmp/resource_library.sqlite，
        # 跨测试互相污染（_dense_library_fallback 召回别家入库的「夜曲(钢琴曲)」）。
        resource_path = Path(self.store.root) / "resource_library.sqlite" if store is not None else settings.resource_library_path
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
        self.library_svc = LibraryService(
            store=self.store,
            media=self.media,
            memory=self.memory,
            library=self.library,
            llm_provider=lambda: self.llm,
        )
        self.source: ExternalSource = _build_source()
        self.engine = RecommendEngine()
        self.daily = DailyRecommender(self.engine, self.source, self.llm)
        self.recommendation = RecommendationService(
            store=self.store,
            memory=self.memory,
            library=self.library,
            list_assets=self.list_assets,
            track_key=_track_key,
            is_quality_track=_is_recommendation_quality_track,
            query_noise=_QUERY_NOISE,
        )
        self.playlists = PlaylistService(
            store=self.store,
            memory=self.memory,
            llm=self.llm,
            list_assets=self.list_assets,
            search_web_music=self.search_web_music,
            source=self.source,
            summarize_taste=self.summarize_taste,
            query_has_entity=self._query_has_entity,
        )
        self.search_service = SearchService(
            library=self.library,
            source=self.source,
            track_key=_track_key,
            dedupe_tracks=_dedupe_tracks,
            merge_search_queries=_merge_search_queries,
            classify_candidate_kind=_classify_candidate_kind,
            valid_external_track=_valid_external_track,
            sync_search_web_music=lambda **kwargs: self.search_web_music(**kwargs),
            search_videos=self.search_videos,
            search_videos_async=self.search_videos_async,
            search_bilibili_detail=self._search_bilibili_detail,
            search_youtube_video=self._search_youtube_video,
            lexical_query_noise=_QUERY_NOISE,
        )
        self.discover = DiscoverService(
            memory=self.memory,
            list_assets=self.list_assets,
            library=self.library,
            retrieve_library_evidence=self.retrieve_library_evidence,
            search_web_music=self.search_web_music,
            track_key=_track_key,
            dedupe_tracks=_dedupe_tracks,
            classify_candidate_kind=_classify_candidate_kind,
            sync_search_videos=lambda **kwargs: self.search_videos(**kwargs),
            sync_search_artist_info=lambda **kwargs: self.search_artist_info(**kwargs),
            extract_search_query=_extract_search_query,
            format_search_summary=_format_search_summary,
            valid_verified_online_track=_is_verified_online_track,
            is_fallback_track=_is_fallback_track,
            artist_query_matches=_artist_query_matches,
            normalize_match_text=_normalize_match_text,
            artist_credit_parts=_artist_credit_parts,
            artist_alias_keys=_artist_alias_keys,
            looks_like_bare_artist_query=_looks_like_bare_artist_query,
            string_similarity=_string_similarity,
        )
        self.playback = PlaybackService(artist_name_matches=self.artist_name_matches)
        self.catalog = CatalogService(
            store=self.store,
            enrich_asset=self.enrich_asset,
            fetch_video_title=self._fetch_video_title,
            sync_recommend_artist_albums=lambda **kwargs: self.recommend_artist_albums(**kwargs),
            search_netease_detail=self._search_netease_detail,
            search_bilibili_detail=self._search_bilibili_detail,
            has_reliable_metadata=_has_reliable_metadata,
            generic_metadata_title=_generic_metadata_title,
        )
        self.taste_experiments = TasteExperimentService(
            store=self.store,
            memory=self.memory,
            library=self.library,
            recommend_for_query=self.recommend_for_query,
            search_web_music=self.search_web_music,
            rerank_tracks=self._rerank_tracks,
            dedupe_tracks=_dedupe_tracks,
            is_recommendation_quality_track=_is_recommendation_quality_track,
        )
        self.journeys = JourneyService(
            store=self.store,
            memory=self.memory,
            library=self.library,
            list_assets=self.list_assets,
            search_web_music=self.search_web_music,
            rerank_tracks=self._rerank_tracks,
            track_key=_track_key,
            dedupe_tracks=_dedupe_tracks,
            is_recommendation_quality_track=_is_recommendation_quality_track,
        )
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
        self.library_svc.enable_cache()
        # 后台预热候选池 embedding：避免首次语义检索在请求路径里同步算几百个向量
        # 拖垮 web_music_search（冷启动曾 21s+ 撞超时）。daemon 线程，进程退出不阻塞。
        threading.Thread(target=self._warm_pool_embeddings, daemon=True).start()

    def _warm_pool_embeddings(self) -> None:
        try:
            warmed = self.library.warm_embeddings()
            if warmed:
                logger.info("候选池 embedding 预热完成：%d 行", warmed)
        except Exception:
            logger.debug("候选池 embedding 预热失败，跳过", exc_info=True)

    # --- 音乐库（薄委托到 LibraryService）---

    def ingest_video(self, url: str, force_refresh: bool = False) -> Asset:
        return self.library_svc.ingest_video(url, force_refresh=force_refresh)

    def enrich_asset(self, asset_id: str, use_network: bool = False) -> EnrichResponse:
        return self.library_svc.enrich_asset(asset_id, use_network=use_network)

    def _fetch_video_title(self, url: str) -> str | None:
        return self.library_svc._fetch_video_title(url)

    def _enrich_from_netease(self, asset: Asset, song_id: str) -> bool:
        return self.library_svc._enrich_from_netease(asset, song_id)

    def _apply_title_artist_hint(self, asset: Asset, video_title: str | None) -> None:
        self.library_svc._apply_title_artist_hint(asset, video_title)

    def _identify_from_url(self, asset: Asset, video_title: str | None = None) -> None:
        self.library_svc._identify_from_url(asset, video_title)

    def analyze_media(self, asset_id: str, force_refresh: bool = False) -> tuple[Asset, list[Segment]]:
        return self.library_svc.analyze_media(asset_id, force_refresh=force_refresh)

    def _playlist_tags_to_genres(self, tags: list[str]) -> list[str]:
        return self.library_svc._playlist_tags_to_genres(tags)

    def _batch_classify_tracks(self, pairs: list[tuple[str, str]]) -> list[dict[str, list[str]]]:
        return self.library_svc._batch_classify_tracks(pairs)

    def _classify_once(self, pairs: list[tuple[str, str]]) -> list[dict[str, list[str]]]:
        return self.library_svc._classify_once(pairs)

    def _ensure_track_tags(
        self,
        title: str,
        artist: str,
        genre: list[str],
        mood: list[str],
        playlist_genres: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        return self.library_svc._ensure_track_tags(
            title, artist, genre, mood, playlist_genres=playlist_genres,
        )

    def import_netease_playlist(
        self,
        playlist_ref: str,
        cookie: str = "",
        user_id: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        return self.library_svc.import_netease_playlist(
            playlist_ref, cookie=cookie, user_id=user_id, limit=limit,
        )

    def list_assets(self) -> list[Asset]:
        return self.library_svc.list_assets()

    def _invalidate_assets_cache(self) -> None:
        self.library_svc._invalidate_assets_cache()

    def delete_asset(self, asset_id: str, user_id: str | None = None) -> bool:
        return self.library_svc.delete_asset(asset_id, user_id=user_id)

    def clear_cache(self, preserve_memory: bool = True) -> dict[str, int]:
        return self.library_svc.clear_cache(preserve_memory=preserve_memory)

    def cleanup_resource_library(self) -> dict[str, int]:
        return self.library_svc.cleanup_resource_library()

    # --- 推荐功能 ---

    def _apply_netease_cookie(self, user_id: str) -> None:
        """把当前 user 绑定的网易云登录 cookie 注入搜索请求（治匿名限流）。

        搜索默认匿名，网易云按 IP 限得很狠；绑定登录后带上 MUSIC_U 额度远高。未绑定则
        退回匿名（兼容旧行为）。在推荐/搜索入口调用一次，下游所有网易云请求都带上。
        """
        try:
            from app.netease_auth import load_cookie
            from app.sources import netease as netease_source
            info = load_cookie(user_id) or {}
            netease_source.set_default_cookie(info.get("cookie") or "")
        except Exception:
            logger.debug("apply netease cookie skipped for %s", user_id, exc_info=True)

    def daily_recommend(
        self,
        user_id: str,
        time_of_day: str | None = None,
        count: int | None = None,
        no_local: bool = False,
    ) -> DailyRecommendation:
        self._apply_netease_cookie(user_id)
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

        # 每日是纯「时间+风格+情绪」氛围推荐、无实体 → 走 route B（策划歌单 + LLM/Last.fm
        # 发现），而非 route C（歌曲关键词搜索，会把风格词搜成业余「(R&B版)」翻唱）。
        # local_ratio：默认 0.3（略压本地、让位线上发现）；每日 tab「仅线上」开关 → 0.0。
        local_ratio = 0.0 if no_local else 0.3
        return self.recommend_for_query(
            user_id, goal, top_k=count, prefer_playlist=True, local_ratio=local_ratio,
        )

    def find_similar_assets(self, asset_id: str, top_k: int = 5) -> list[SimilarAssetResult]:
        return self.similarity.find_similar_assets(asset_id, top_k)

    def find_similar_segments(self, asset_id: str, segment_id: str, top_k: int = 5) -> list[SimilarSegmentResult]:
        return self.similarity.find_similar_segments(asset_id, segment_id, top_k)

    # --- 搜索 ---

    def search(self, user_id: str, query: str, include_external: bool = True, top_k: int = 20, offset: int = 0) -> SearchResponse:
        self._apply_netease_cookie(user_id)
        return self._discover_service().search(
            user_id,
            query,
            include_external=include_external,
            top_k=top_k,
            offset=offset,
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
        return self._search_service().search_web_music(
            query=query,
            top_k=top_k,
            relevance_query=relevance_query,
            include_video_sources=include_video_sources,
            offset=offset,
            variants=variants,
        )

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
        return await self._search_service().search_web_music_async(
            query=query,
            top_k=top_k,
            relevance_query=relevance_query,
            include_video_sources=include_video_sources,
            offset=offset,
            variants=variants,
        )

    def _dense_library_fallback(self, query: str, existing: list[ExternalTrack], limit: int = 5) -> list[ExternalTrack]:
        return self._search_service().dense_library_fallback(query, existing, limit)

    def _lexical_resource_fallback(self, query: str, limit: int = 10) -> list[ResourceTrack]:
        """Zero-network fallback over verified resource metadata when embeddings are unavailable."""
        return self._search_service().lexical_resource_fallback(query, limit)

    def search_videos(self, query: str, top_k: int = 5) -> list[ExternalTrack]:
        return self._discover_service().search_videos(query, top_k=top_k)

    async def search_videos_async(self, query: str, top_k: int = 5) -> list[ExternalTrack]:
        return await self._discover_service().search_videos_async(query, top_k=top_k)

    def search_artist_info(self, query: str) -> list[dict[str, str]]:
        return self._discover_service().search_artist_info(query)

    async def search_artist_info_async(self, query: str) -> list[dict[str, str]]:
        return await self._discover_service().search_artist_info_async(query)

    def classify_discover_query(self, query: str) -> dict[str, Any]:
        return self._discover_service().classify_discover_query(query)

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
        return self._catalog_service().fetch_track_metadata(asset_id=asset_id, url=url, use_network=use_network)

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

    def chat(
        self,
        user_id: str,
        message: str,
        history: list[dict[str, Any]] | None = None,
        thread_id: str | None = None,
        run_id: str | None = None,
    ) -> AgentAnswer:
        """Backward-compatible sync wrapper used by local scripts and smoke tools."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.chat_async(
                    user_id=user_id,
                    message=message,
                    history=history,
                    thread_id=thread_id,
                    run_id=run_id,
                )
            )
        raise RuntimeError("AudioVisualAgent.chat() cannot run inside an active event loop; use chat_async().")

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
        return self.library_svc.list_resource_tracks(limit)

    def generate_music_journey(self, user_id: str, instruction: str, target_count: int | None = None) -> dict[str, Any]:
        return self._journey_service().generate_music_journey(
            user_id,
            instruction,
            target_count=target_count,
            journey_phases=_journey_phases,
            record_journey_history=self._record_journey_history,
        )

    def _record_journey_history(self, user_id: str, tracks: list[Asset | ExternalTrack]) -> None:
        self._journey_service().record_journey_history(user_id, tracks)

    # --- 播放 ---

    def get_playback_url(self, track: Asset | ExternalTrack, netease_cookie: str = "") -> str | None:
        return self._playback_service_with_instance_overrides().get_playback_url(track, netease_cookie)

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
        return self._playback_service_with_instance_overrides().get_audio_url(track, netease_cookie)

    def get_mv_url(self, track: Asset | ExternalTrack) -> str | None:
        return self._playback_service_with_instance_overrides().get_mv_url(track)

    def _extract_youtube_id(self, url: str) -> str | None:
        return self._playback_service().extract_youtube_id(url)

    def _extract_bilibili_id(self, url: str) -> tuple[str, str] | None:
        return self._playback_service().extract_bilibili_id(url)

    def _search_youtube_video(self, query: str) -> str | None:
        return self._playback_service().search_youtube_video(query)

    def _search_bilibili_video(self, query: str) -> str | None:
        """搜 B 站视频，返回 bvid。华语 MV 命中率高，嵌入不弹机器人验证。"""
        return self._playback_service().search_bilibili_video(query)

    def _search_netease(self, query: str) -> str | None:
        return self._playback_service().search_netease(query)

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
        return self._playback_service().get_netease_audio_url(song_id, cookie)

    def get_lyrics(self, title: str, artist: str, source_id: str = "") -> dict:
        return self._playback_service_with_instance_overrides().get_lyrics(title, artist, source_id)

    # --- 歌单 ---

    def generate_playlist(
        self,
        user_id: str,
        instruction: str,
        seed_tracks: list[Asset | ExternalTrack] | None = None,
        target_count: int | None = None,
    ) -> Playlist:
        return self._playlist_service().generate_playlist(
            user_id,
            instruction=instruction,
            seed_tracks=seed_tracks,
            target_count=target_count,
            infer_playlist_count=_infer_playlist_count,
            playlist_candidates_builder=self._playlist_candidates,
            extract_search_query=_extract_search_query,
            track_key=_track_key,
            dedupe_tracks=_dedupe_tracks,
            is_quality_track=_is_recommendation_quality_track,
            is_playlist_context_compatible=_is_playlist_context_compatible,
            query_requests_variant_content=_query_requests_variant_content,
        )

    def auto_playlists(self, user_id: str) -> list[Playlist]:
        return self._playlist_service().auto_playlists(
            user_id,
            fallback_auto_playlists=self._fallback_auto_playlists,
        )

    def save_playlist(self, user_id: str, playlist: Playlist) -> None:
        self._playlist_service().save_playlist(user_id, playlist)

    def list_playlists(self, user_id: str) -> list[Playlist]:
        return self._playlist_service().list_playlists(user_id)

    def delete_playlist(self, user_id: str, playlist_id: str) -> bool:
        return self._playlist_service().delete_playlist(user_id, playlist_id)

    # ── 收藏专辑（与歌单同构：collection=saved_albums，key=f"{user_id}_{album_id}"） ──

    def save_album(self, user_id: str, album: SavedAlbum) -> SavedAlbum:
        self.store.write_model("saved_albums", f"{user_id}_{album.album_id}", album)
        try:
            self.memory.refresh_taste_profile(user_id, self.list_assets())
        except Exception:
            logger.debug("refresh_taste_profile failed after save_album(%s)", album.album_id, exc_info=True)
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
        deleted = self.store.delete_key("saved_albums", f"{user_id}_{album_id}")
        if deleted:
            try:
                self.memory.refresh_taste_profile(user_id, self.list_assets())
            except Exception:
                logger.debug("refresh_taste_profile failed after delete_saved_album(%s)", album_id, exc_info=True)
        return deleted

    def is_album_saved(self, user_id: str, album_id: str) -> bool:
        return self.store.read_model("saved_albums", f"{user_id}_{album_id}", SavedAlbum) is not None

    def _playlist_candidates(
        self,
        instruction: str,
        library: list[Asset],
        seed_tracks: list[Asset | ExternalTrack],
        target_count: int,
    ) -> list[Asset | ExternalTrack]:
        return self._playlist_service().playlist_candidates(
            instruction,
            library,
            seed_tracks,
            target_count,
            playlist_search_terms=_playlist_search_terms,
            extract_search_query=_extract_search_query,
            query_requests_variant_content=_query_requests_variant_content,
            is_quality_track=_is_recommendation_quality_track,
            is_playlist_context_compatible=_is_playlist_context_compatible,
            is_scenario_playlist_instruction=_is_scenario_playlist_instruction,
            curated_playlist_query=_curated_playlist_query,
            playlist_online_queries=_playlist_online_queries,
            playlist_match_score=_playlist_match_score,
            dedupe_tracks=_dedupe_tracks,
        )

    def _fallback_playlist(
        self,
        user_id: str,
        instruction: str,
        library: list[Asset],
        target_count: int | None = None,
        candidates: list[Asset | ExternalTrack] | None = None,
    ) -> Playlist:
        return self._playlist_service().fallback_playlist(
            user_id,
            instruction,
            library,
            target_count=target_count or _infer_playlist_count(instruction) or 12,
            candidates=candidates,
            save_playlist=self.save_playlist,
            is_quality_track=_is_recommendation_quality_track,
            query_requests_variant_content=_query_requests_variant_content,
            fill_tracks=_fill_tracks,
        )

    def _fallback_auto_playlists(self, user_id: str, library: list[Asset]) -> list[Playlist]:
        return self._playlist_service().fallback_auto_playlists(user_id, library)

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

    def summarize_taste(self, user_id: str, *, include_artists: bool = True, memory: UserMemory | None = None) -> str:
        # memory 可由调用方传入（recommend_for_query 已 get_memory 过），省掉同请求内
        # 对同一用户的重复读盘+校验。不传则照旧自行读取，行为不变。
        if memory is None:
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

    def profile_context_text(self, user_id: str) -> str:
        """压缩画像仪表盘（app/profile/）为 query_plan 用的品位上下文（软参考）。

        与 summarize_taste（听歌历史频次）互补：这里带场景偏好、探索风格、画像级排除/回避，
        以及被用户「纠错」后落地的信号。空画像/异常返 ""——调用方据此跳过注入，行为不变。
        """
        try:
            from app.services.profile import UserProfileService
            ctx = UserProfileService(self.store, self.memory).get_context_for_llm(user_id)
        except Exception:
            logger.debug("profile_context_text 失败，跳过画像注入", exc_info=True)
            return ""
        parts: list[str] = []
        if ctx.taste_summary:
            parts.append(f"当前品味：{ctx.taste_summary}")
        if ctx.active_scene_preference:
            parts.append(f"常听场景：{ctx.active_scene_preference}")
        if ctx.discovery_mode:
            parts.append(f"探索风格：{ctx.discovery_mode}")
        if ctx.hard_constraints:
            parts.append(f"明确排除：{'、'.join(ctx.hard_constraints[:5])}")
        if ctx.avoid_features:
            parts.append(f"场景应回避：{'、'.join(ctx.avoid_features[:5])}")
        if ctx.rejected_signals:
            # 用户在画像页「纠错」否定过的判断——推荐应反转/回避，别再按这些推。
            parts.append(f"用户已否定（勿据此推荐）：{'；'.join(ctx.rejected_signals[:4])}")
        return "；".join(parts)

    def _profile_rerank_signals(self, user_id: str) -> tuple[set[str], set[str]]:
        """从画像仪表盘取 rerank 艺人信号：core/rising→加分，avoid→减分。

        与 memory.taste_profile（听歌频次）互补——画像是可解释的艺人关系判断。
        **纠错回流**：用户在画像页否定的艺人洞察会反转关系——
          否定「X 是核心艺人」→ X 从加分移到减分（别再按核心推）；
          否定「不要推 X」→ X 从减分移除（用户其实不排斥 X）。
        空画像/异常返回空集，rerank 行为不变。
        """
        try:
            from app.services.profile import UserProfileService
            profile = UserProfileService(self.store, self.memory).get_profile(user_id)
        except Exception:
            logger.debug("_profile_rerank_signals 失败，降级无画像信号", exc_info=True)
            return set(), set()
        if getattr(profile, "is_empty", True):
            return set(), set()
        # 纠错回流：被否定的艺人洞察（title 含艺人名）决定关系反转方向——
        #   core/rising 被否定 → 从加分改减分；avoid 被否定 → 不再减分。
        # 直接按关系+是否被否定一次建表，避免先加进 penalty 又被第二轮扫描误删。
        rejected_titles = [
            i.title for i in profile.insights
            if i.status == "rejected" and i.dimension == "artist"
        ]

        def _rejected(name: str) -> bool:
            return any(name in t for t in rejected_titles)

        boost: set[str] = set()
        penalty: set[str] = set()
        for a in profile.artists:
            if not a.artist:
                continue
            if a.relation_type in {"core", "rising"}:
                (penalty if _rejected(a.artist) else boost).add(a.artist)
            elif a.relation_type == "avoid" and not _rejected(a.artist):
                penalty.add(a.artist)
        return boost, penalty

    def recommend_artist_albums(self, user_id: str, artist: str, limit: int = 12) -> list[dict[str, Any]]:
        return self._catalog_service().recommend_artist_albums(user_id, artist, limit)

    async def recommend_artist_albums_async(
        self, user_id: str, artist: str, limit: int = 12,
    ) -> list[dict[str, Any]]:
        return await self._catalog_service().recommend_artist_albums_async(user_id, artist, limit)

    def generate_taste_experiment(self, user_id: str, prompt: str, total: int = 12) -> TasteExperiment:
        """生成 safe/stretch/bold 三档品味实验。"""
        return self._taste_experiment_service().generate_taste_experiment(
            user_id,
            prompt,
            total=total,
            taste_experiment_hypothesis=self._taste_experiment_hypothesis,
            taste_experiment_search_seeds=self._taste_experiment_search_seeds,
            collect_taste_candidates=self._collect_taste_candidates,
            taste_prompt_exclusions=self._taste_prompt_exclusions,
            filter_taste_experiment_candidates=self._filter_taste_experiment_candidates,
            bucket_taste_experiment_candidates=self._bucket_taste_experiment_candidates,
            taste_experiment_track=self._taste_experiment_track,
            new_taste_experiment_id=self._new_taste_experiment_id,
            save_taste_experiment=self._save_taste_experiment,
        )

    def _collect_taste_candidates(
        self,
        user_id: str,
        seeds: list[str],
        total: int,
    ) -> list[tuple[Any, dict[str, float], str, float]]:
        return self._taste_experiment_service().collect_taste_candidates(user_id, seeds, total)

    @staticmethod
    def _taste_prompt_exclusions(prompt: str) -> list[str]:
        return TasteExperimentService.taste_prompt_exclusions(prompt)

    def regenerate_taste_experiment_bucket(self, user_id: str, experiment_id: str, bucket: str) -> TasteExperiment:
        return self._taste_experiment_service().regenerate_taste_experiment_bucket(
            user_id,
            experiment_id,
            bucket,
            taste_experiment_seeds_for_bucket=self._taste_experiment_seeds_for_bucket,
            collect_taste_candidates=self._collect_taste_candidates,
            filter_taste_experiment_candidates=self._filter_taste_experiment_candidates,
            taste_experiment_track_key=self._taste_experiment_track_key,
            candidate_key=self._candidate_key,
            taste_familiarity=self._taste_familiarity,
            slice_for_bucket=self._slice_for_bucket,
            taste_experiment_track=self._taste_experiment_track,
        )

    def _taste_experiment_seeds_for_bucket(self, memory: UserMemory, prompt: str, bucket: str) -> list[str]:
        return TasteExperimentService.taste_experiment_seeds_for_bucket(memory, prompt, bucket)

    @staticmethod
    def _dedupe_seeds(seeds: list[str]) -> list[str]:
        return TasteExperimentService.dedupe_seeds(seeds)

    def list_taste_experiments(self, user_id: str) -> list[TasteExperiment]:
        return self._taste_experiment_service().list_taste_experiments(user_id)

    def get_taste_experiment(self, user_id: str, experiment_id: str) -> TasteExperiment | None:
        return self._taste_experiment_service().get_taste_experiment(user_id, experiment_id)

    def delete_taste_experiment(self, user_id: str, experiment_id: str) -> bool:
        return self._taste_experiment_service().delete_taste_experiment(user_id, experiment_id)

    def record_taste_experiment_feedback(self, request: TasteExperimentFeedbackRequest) -> TasteExperiment:
        return self._taste_experiment_service().record_taste_experiment_feedback(
            request,
            find_taste_experiment_track=self._find_taste_experiment_track,
            apply_taste_experiment_ts_feedback=self._apply_taste_experiment_ts_feedback,
            record_taste_experiment_listen=self._record_taste_experiment_listen,
            taste_experiment_feedback_count=self._taste_experiment_feedback_count,
        )

    def summarize_taste_experiment(self, user_id: str, experiment_id: str) -> TasteExperimentReport:
        return self._taste_experiment_service().summarize_taste_experiment(
            user_id,
            experiment_id,
            taste_experiment_bucket_stats=self._taste_experiment_bucket_stats,
            bucket_label=self._bucket_label,
        )

    def _save_taste_experiment(self, experiment: TasteExperiment) -> None:
        self._taste_experiment_service().save_taste_experiment(experiment)

    @staticmethod
    def _new_taste_experiment_id(user_id: str, prompt: str) -> str:
        return TasteExperimentService.new_taste_experiment_id(user_id, prompt)

    def _taste_experiment_hypothesis(self, memory: UserMemory) -> str:
        return TasteExperimentService.taste_experiment_hypothesis(memory)

    def _taste_experiment_search_seeds(self, memory: UserMemory, prompt: str) -> list[str]:
        return TasteExperimentService.taste_experiment_search_seeds(memory, prompt)

    def _filter_taste_experiment_candidates(
        self,
        user_id: str,
        candidates: list[tuple[Any, dict[str, float], str, float]],
        exclusion_rules: list[str],
    ) -> list[tuple[Any, dict[str, float], str, float]]:
        return filter_taste_experiment_candidates(
            library=self.library,
            user_id=user_id,
            candidates=candidates,
            exclusion_rules=exclusion_rules,
            is_quality_track=self._is_taste_experiment_quality_track,
        )

    @staticmethod
    def _is_taste_experiment_quality_track(track: Any) -> bool:
        """Taste Lab 候选质量门槛：挡掉明显不像正式歌曲的搜索噪声。"""
        return _is_recommendation_quality_track(track)

    def _bucket_taste_experiment_candidates(
        self,
        candidates: list[tuple[Any, dict[str, float], str, float]],
        per_bucket: int,
    ) -> dict[str, list[tuple[Any, dict[str, float], str, float]]]:
        return bucket_taste_experiment_candidates(candidates, per_bucket)

    @staticmethod
    def _taste_familiarity(item: tuple[Any, dict[str, float], str, float]) -> float:
        return taste_familiarity(item)

    @staticmethod
    def _slice_for_bucket(
        ranked: list[tuple[Any, dict[str, float], str, float]],
        bucket: str,
        per_bucket: int,
    ) -> list[tuple[Any, dict[str, float], str, float]]:
        return slice_for_bucket(ranked, bucket, per_bucket)

    @staticmethod
    def _candidate_key(item: tuple[Any, dict[str, float], str, float]) -> str:
        return candidate_key(item)

    @staticmethod
    def _taste_experiment_track(
        track: Any,
        bucket: str,
        components: dict[str, float],
        reason: str,
        score: float,
    ) -> TasteExperimentTrack:
        return TasteExperimentService.taste_experiment_track(track, bucket, components, reason, score)

    @staticmethod
    def _taste_experiment_track_key(item: TasteExperimentTrack) -> str:
        return taste_experiment_track_key(item)

    def _find_taste_experiment_track(self, experiment: TasteExperiment, track_key: str) -> TasteExperimentTrack | None:
        return find_taste_experiment_track(experiment, track_key)

    def _apply_taste_experiment_ts_feedback(self, item: TasteExperimentTrack, signal: str, score: float | None) -> None:
        apply_taste_experiment_ts_feedback(
            library=self.library,
            item=item,
            signal=signal,
            score=score,
        )

    def _record_taste_experiment_listen(
        self,
        user_id: str,
        item: TasteExperimentTrack,
        signal: str,
        score: float | None,
    ) -> None:
        try:
            record_taste_experiment_listen(
                memory=self.memory,
                user_id=user_id,
                item=item,
                signal=signal,
                score=score,
            )
        except Exception:
            logger.debug("taste experiment listen record failed", exc_info=True)

    @staticmethod
    def _taste_experiment_feedback_count(experiment: TasteExperiment) -> int:
        return taste_experiment_feedback_count(experiment)

    def _taste_experiment_bucket_stats(self, experiment: TasteExperiment) -> dict[str, dict[str, float | int]]:
        return taste_experiment_bucket_stats(experiment)

    @staticmethod
    def _bucket_label(bucket: str) -> str:
        return bucket_label(bucket)

    def recommend_for_query(
        self,
        user_id: str,
        goal: str,
        top_k: int = 5,
        *,
        excluded_tracks: list[dict[str, str]] | None = None,
        search_variants: list[str] | None = None,
        seed_tracks: list[Asset | ExternalTrack] | None = None,
        prefer_playlist: bool = False,
        local_ratio: float = 0.4,
        search_query_override: str | None = None,
    ) -> DailyRecommendation:
        self._apply_netease_cookie(user_id)
        rec_service = self._recommendation_service()
        ctx = rec_service.build_context(
            user_id=user_id,
            goal=goal,
            top_k=top_k,
            local_ratio=_local_ratio_from_query(goal, local_ratio),
            search_query_override=search_query_override,
            seed_tracks=seed_tracks,
            extract_search_query=_extract_search_query,
            extract_recommendation_anchors=_extract_recommendation_anchors,
            scene_playlist_queries=_scene_playlist_queries,
            query_has_entity=self._query_has_entity,
            summarize_taste=self.summarize_taste,
            is_verified_track=_is_verified_recommendation_track,
            is_quality_track=lambda track: _is_recommendation_quality_track(track),
        )
        memory = ctx.memory
        trace_lines: list[str] = []
        all_candidates: list[Asset | ExternalTrack] = list(seed_tracks or [])

        if ctx.seed_supply >= top_k:
            trace_lines.append(f"route=seed_candidates, supplied={ctx.seed_supply}")
        elif not prefer_playlist and ((ctx.has_entity and not ctx.scene_queries) or ctx.anchors.explicit):
            all_candidates, route_trace = rec_service.extend_exact_route_candidates(
                candidates=all_candidates,
                search_goal=ctx.search_goal,
                goal=goal,
                anchors=ctx.anchors,
                search_variants=search_variants,
                top_k=top_k,
                excluded_tracks=excluded_tracks,
                search_web_music=self.search_web_music,
                dedupe_tracks=_dedupe_tracks,
                recommendation_search_seeds=_recommendation_search_seeds,
            )
            trace_lines.extend(route_trace)
        else:
            from app.search.netease_playlist import search_and_extract
            from app.search.lastfm_discovery import discover_from_lastfm
            from app.search.web_music_discovery import discover_from_llm

            all_candidates, route_trace = rec_service.extend_discovery_route_candidates(
                candidates=all_candidates,
                goal=goal,
                search_goal=ctx.search_goal,
                scene_queries=ctx.scene_queries,
                prefer_playlist=prefer_playlist,
                top_k=top_k,
                memory=memory,
                taste_summary=ctx.taste_summary,
                library_artists=ctx.library_artists,
                dedupe_tracks=_dedupe_tracks,
                search_and_extract=search_and_extract,
                discover_from_llm=lambda **kwargs: discover_from_llm(**kwargs, llm=self.llm),
                discover_from_lastfm=discover_from_lastfm,
            )
            trace_lines.extend(route_trace)

        # 本地曲库必须真正参与推荐，而不只是被压缩成画像后再去线上搜。
        # 精确实体查询只加入标题/歌手匹配项；场景查询加入画像/场景相关项。
        local_candidates = self._local_recommendation_candidates(user_id, ctx.search_goal or goal, memory)
        trace_lines.append(f"route=local_library, matched={len(local_candidates)}")
        all_candidates.extend(local_candidates)
        allow_variants = _query_requests_variant_content(goal)

        verified = rec_service.filter_verified_candidates(
            candidates=all_candidates,
            user_id=user_id,
            goal=goal,
            excluded_tracks=excluded_tracks,
            dedupe_tracks=_dedupe_tracks,
            is_verified_track=_is_verified_recommendation_track,
            is_quality_track=_is_recommendation_quality_track,
            is_context_compatible=_is_playlist_context_compatible,
            allow_variants=allow_variants,
            filter_excluded_tracks=_filter_excluded_tracks,
        )

        # 兜底：用 search_goal 再搜一次。带 offset 翻页（已排除 + 已收集数），
        # 否则同查询永远返回 top-N，与首轮 batch 重复，dedup 全跳过、补不了量。
        verified = rec_service.extend_with_online_fallback(
            verified=verified,
            user_id=user_id,
            goal=goal,
            search_goal=ctx.search_goal,
            top_k=top_k,
            excluded_tracks=excluded_tracks,
            search_variants=search_variants,
            can_fallback=not ctx.anchors.explicit and not ctx.scene_queries,
            search_web_music=self.search_web_music,
            is_verified_online_track=_is_verified_online_track,
            is_quality_track=_is_recommendation_quality_track,
            is_context_compatible=_is_playlist_context_compatible,
            allow_variants=allow_variants,
        )

        # 候选池兜底：网易云限流时所有在线路由同时空，但 SQLite 已攒了大量已验证真歌。
        # 与其返回空，不如从候选池语义/词法召回补位——这些是历史搜到、验证过的真实曲目，
        # 可播放、不是幻觉。限流是间歇的，今天搜不到的昨天可能已入池。
        # 复用 _dense_library_fallback：它做 semantic→lexical 召回 + 转 ExternalTrack +
        # 按 existing 去重，类型与下游 rerank 兼容。
        # prefer_playlist（每日）：歌单本身就是策划，不再用风格锚点严格过滤——否则会把
        # 歌单 166 首砍到剩 1 首。锚点过滤留给 route C 的实体搜索。
        if ctx.anchors.explicit and not prefer_playlist:
            before = len(verified)
            verified = [track for track in verified if _track_matches_recommendation_anchors(track, ctx.anchors)]
            dropped = before - len(verified)
            if dropped:
                trace_lines.append(f"anchor_filter=dropped:{dropped}")

        verified, pool_hit_count = rec_service.extend_with_resource_pool(
            verified=verified,
            user_id=user_id,
            goal=goal,
            search_goal=ctx.search_goal or goal,
            top_k=top_k,
            anchors_explicit=ctx.anchors.explicit,
            prefer_playlist=prefer_playlist,
            dense_library_fallback=self._dense_library_fallback,
            is_quality_track=_is_recommendation_quality_track,
            is_context_compatible=_is_playlist_context_compatible,
            anchor_matcher=lambda track: _track_matches_recommendation_anchors(track, ctx.anchors),
            allow_variants=allow_variants,
        )
        if pool_hit_count:
            trace_lines.append(f"route=resource_pool, recalled={pool_hit_count}")

        if verified:
            verified = rec_service.prioritize_fresh_candidates(
                verified, memory.recommendation_history, top_k=top_k,
            )
            rerank_query = ctx.search_goal or goal
            # 先对完整候选池排序，再做来源平衡。若这里只取 top_k，标签更完整的
            # local 会在平衡前就把 online 全挤掉，后续再设配额也无候选可选。
            ranked_pool = self._rerank_tracks(
                user_id, rerank_query, _dedupe_tracks(verified), top_k=len(verified),
            )
            ranked = self._balance_recommendation_sources(ranked_pool, top_k, local_ratio=ctx.local_ratio)
            self.library.record_exposure([t for t, _ in ranked])
            self.library.decay_exposure_ts([t for t, _ in ranked])
            tracks: list[RecommendedTrack] = []
            for track, breakdown in ranked:
                tracks.append(RecommendedTrack(
                    asset=track,
                    score=breakdown.score,
                    reason=breakdown.reason or _online_candidate_reason(track, ctx.memory_query),
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
        return self._recommendation_service().local_recommendation_candidates(
            user_id, query, memory, limit=limit,
        )

    @staticmethod
    def _balance_recommendation_sources(
        ranked: list[tuple[Asset | ExternalTrack, Any]],
        top_k: int,
        local_ratio: float = 0.4,
    ) -> list[tuple[Asset | ExternalTrack, Any]]:
        """Thin wrapper so source balancing can move out of the agent incrementally."""
        return balance_recommendation_sources(
            ranked,
            top_k,
            local_ratio=local_ratio,
            is_local_track=_is_local_recommendation_track,
        )

    def _record_recommendation_history(self, user_id: str, tracks: list[Asset | ExternalTrack]) -> None:
        self._recommendation_service().record_recommendation_history(user_id, tracks)

    def _recommendation_service(self) -> RecommendationService:
        service = getattr(self, "recommendation", None)
        if service is None:
            service = RecommendationService(
                store=getattr(self, "store", None),
                memory=self.memory,
                library=self.library,
                list_assets=self.list_assets,
                track_key=_track_key,
                is_quality_track=_is_recommendation_quality_track,
                query_noise=_QUERY_NOISE,
            )
            self.recommendation = service
        return service

    def _playlist_service(self) -> PlaylistService:
        service = getattr(self, "playlists", None)
        if service is None:
            service = PlaylistService(
                store=getattr(self, "store", None),
                memory=self.memory,
                llm=self.llm,
                list_assets=self.list_assets,
                search_web_music=self.search_web_music,
                source=self.source,
                summarize_taste=self.summarize_taste,
                query_has_entity=self._query_has_entity,
            )
            self.playlists = service
        return service

    def _search_service(self) -> SearchService:
        service = getattr(self, "search_service", None)
        if service is None or not isinstance(service, SearchService):
            service = SearchService(
                library=self.library,
                source=getattr(self, "source", MockSource()),
                track_key=_track_key,
                dedupe_tracks=_dedupe_tracks,
                merge_search_queries=_merge_search_queries,
                classify_candidate_kind=_classify_candidate_kind,
                valid_external_track=_valid_external_track,
                sync_search_web_music=lambda **kwargs: self.search_web_music(**kwargs),
                search_videos=self.search_videos,
                search_videos_async=self.search_videos_async,
                search_bilibili_detail=self._search_bilibili_detail,
                search_youtube_video=self._search_youtube_video,
                lexical_query_noise=_QUERY_NOISE,
            )
            self.search_service = service
        return service

    def _taste_experiment_service(self) -> TasteExperimentService:
        service = getattr(self, "taste_experiments", None)
        if service is None or not isinstance(service, TasteExperimentService):
            service = TasteExperimentService(
                store=getattr(self, "store", None),
                memory=self.memory,
                library=self.library,
                recommend_for_query=self.recommend_for_query,
                search_web_music=self.search_web_music,
                rerank_tracks=self._rerank_tracks,
                dedupe_tracks=_dedupe_tracks,
                is_recommendation_quality_track=_is_recommendation_quality_track,
            )
            self.taste_experiments = service
        return service

    def _journey_service(self) -> JourneyService:
        service = getattr(self, "journeys", None)
        if service is None or not isinstance(service, JourneyService):
            service = JourneyService(
                store=getattr(self, "store", None),
                memory=self.memory,
                library=self.library,
                list_assets=self.list_assets,
                search_web_music=self.search_web_music,
                rerank_tracks=self._rerank_tracks,
                track_key=_track_key,
                dedupe_tracks=_dedupe_tracks,
                is_recommendation_quality_track=_is_recommendation_quality_track,
            )
            self.journeys = service
        return service

    def _discover_service(self) -> DiscoverService:
        service = getattr(self, "discover", None)
        if service is None or not isinstance(service, DiscoverService):
            service = DiscoverService(
                memory=self.memory,
                list_assets=self.list_assets,
                library=self.library,
                retrieve_library_evidence=self.retrieve_library_evidence,
                search_web_music=self.search_web_music,
                track_key=_track_key,
                dedupe_tracks=_dedupe_tracks,
                classify_candidate_kind=_classify_candidate_kind,
                sync_search_videos=lambda **kwargs: self.search_videos(**kwargs),
                sync_search_artist_info=lambda **kwargs: self.search_artist_info(**kwargs),
                extract_search_query=_extract_search_query,
                format_search_summary=_format_search_summary,
                valid_verified_online_track=_is_verified_online_track,
                is_fallback_track=_is_fallback_track,
                artist_query_matches=_artist_query_matches,
                normalize_match_text=_normalize_match_text,
                artist_credit_parts=_artist_credit_parts,
                artist_alias_keys=_artist_alias_keys,
                looks_like_bare_artist_query=_looks_like_bare_artist_query,
                string_similarity=_string_similarity,
            )
            self.discover = service
        return service

    def _playback_service(self) -> PlaybackService:
        service = getattr(self, "playback", None)
        if service is None or not isinstance(service, PlaybackService):
            service = PlaybackService(artist_name_matches=self.artist_name_matches)
            self.playback = service
        return service

    def _playback_service_with_instance_overrides(self) -> PlaybackService:
        search_netease = self.__dict__.get("_search_netease")
        get_audio = self.__dict__.get("_get_netease_audio_url")
        if search_netease is None and get_audio is None:
            return self._playback_service()
        return PlaybackService(
            search_netease=search_netease,
            get_netease_audio_url=get_audio,
            artist_name_matches=self.artist_name_matches,
        )

    def _catalog_service(self) -> CatalogService:
        service = getattr(self, "catalog", None)
        if service is None or not isinstance(service, CatalogService):
            service = CatalogService(
                store=getattr(self, "store", None),
                enrich_asset=self.enrich_asset,
                fetch_video_title=self._fetch_video_title,
                sync_recommend_artist_albums=lambda **kwargs: self.recommend_artist_albums(**kwargs),
                search_netease_detail=self._search_netease_detail,
                search_bilibili_detail=self._search_bilibili_detail,
                has_reliable_metadata=_has_reliable_metadata,
                generic_metadata_title=_generic_metadata_title,
            )
            self.catalog = service
        return service

    @staticmethod
    def _query_has_entity(search_goal: str) -> bool:
        return RecommendationService.query_has_entity(search_goal, _QUERY_NOISE)

    def _rerank_tracks(self, user_id: str, query: str, tracks: list[Any], top_k: int):
        """三锚精排 + MMR 多样性重排管线。返回 [(track, RankingBreakdown), ...]。"""
        return self._recommendation_service().rerank_tracks(
            user_id,
            query,
            tracks,
            top_k,
            profile_signal_provider=self._profile_rerank_signals,
        )

    def _collaborative_scores(self, user_id: str, tracks: list[Any], memory: Any) -> tuple[list[float] | None, bool]:
        """兼容薄包装：CF 共现分现由 RecommendationService 负责。"""
        return self._recommendation_service().collaborative_scores(user_id, tracks, memory)

    @staticmethod
    def _enrich_candidate_tags(tracks: list[Any]) -> None:
        """兼容薄包装：候选标签补全现由 RecommendationService 负责。"""
        RecommendationService.enrich_candidate_tags(tracks)

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
# 向后兼容别名
CineSonicAgent = AudioVisualAgent
