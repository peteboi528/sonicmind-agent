from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse

from app.agent import AudioVisualAgent
from app.config import settings
from app.models import (
    AgentResumeRequest,
    AlbumTracksRequest,
    AlbumTracksResponse,
    ArtistInfoRequest,
    ArtistInfoResponse,
    BrowseRequest,
    ChatHistoryTurnRequest,
    ChatRequest,
    DailyRequest,
    DiscoverQueryClassification,
    DiscoverQueryRequest,
    DislikeRequest,
    EnrichRequest,
    FeedbackRequest,
    IngestRequest,
    JourneyRequest,
    ListenRequest,
    LyricsRequest,
    MemoryUpdateRequest,
    PlaylistFromAssetsRequest,
    PlaylistRequest,
    ProfileInsightFeedbackRequest,
    RatingRequest,
    RecommendationHistoryRequest,
    SaveAlbumRequest,
    SearchRequest,
    SearchResponse,
    TasteExperimentFeedbackRequest,
    TasteExperimentRegenerateRequest,
    TasteExperimentReportRequest,
    TasteExperimentRequest,
    TrendingRequest,
)
from app.rate_limit import RateLimiter
from app.security.ingest_guard import IngestURLError, validate_ingest_url

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    from app.services.tools import checkpoint_store, trace_store

    checkpoint_store.cleanup()
    trace_store.cleanup()
    # 启动时把存量明文 cookie 迁移为加密格式（幂等；无明文/未启用加密则空操作）。
    try:
        from app.security import secret_box

        migrated = secret_box.migrate_plaintext_cookies(settings.store_root).get("migrated", 0)
        if migrated:
            logger.info("启动迁移：%d 个明文 cookie 文件已加密重写", migrated)
    except Exception:
        logger.debug("启动 cookie 迁移失败", exc_info=True)
    if settings.agent_checkpoints and agent.graph is not None:
        await agent.graph.initialize_checkpointing(settings.agent_checkpoint_path)
    try:
        yield
    finally:
        from app.llm.client import close_shared_async_client
        from app.services.tools import tool_runtime
        from app.sources.http_transport import source_transport

        if agent.graph is not None:
            await agent.graph.close()
        await source_transport.close()
        await close_shared_async_client()
        await tool_runtime.close()
        # 共享有界线程池：cancel 未启动任务后立即返回（已在阻塞 syscall 的 worker 仍会跑到
        # 自身 socket 超时才退出，Python 无法强取消已开始线程）。
        from app.concurrency import shutdown_shared_executor

        shutdown_shared_executor()

app = FastAPI(
    title="智能影音推荐助手 API",
    description="音视频内容分析、个性化推荐、每日歌单、LLM 语义搜索。",
    version="0.3.0",
    lifespan=_lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 分档限流器（模块级单例，桶状态跨请求保持）。配置在 import 时读取；运行时改 settings
# 需 monkeypatch 此单例（见 tests/test_rate_limit.py）。
_rate_limiter = RateLimiter({
    "chat": settings.chat_rpm,
    "playback": settings.playback_rpm,
    "ingest": settings.ingest_rpm,
})


@app.middleware("http")
async def _enforce_rate_limit(request, call_next):
    """分档限流（RATE_LIMIT_ENABLED=true 时生效）。

    按 user_id|IP 维度对聊天 / 播放代理端点限流，超限 429 + Retry-After。
    本中间件先注册、由随后注册的鉴权中间件包在外层，因此 keying 优先使用
    鉴权绑定的 user_id；未鉴权时退回按 IP 限流。
    body 里的 user_id 不进 key（客户端可伪造，playback 漏洞正是它）。
    """
    if settings.rate_limit_enabled:
        path = request.url.path
        if path.startswith(("/chat", "/agent/run", "/agent/stream")):
            tier = "chat"
        elif path.startswith("/api/playback/"):
            tier = "playback"
        elif path.startswith("/assets/ingest"):
            tier = "ingest"
        else:
            tier = None
        if tier:
            uid = getattr(request.state, "auth_user_id", None)
            client_host = request.client.host if request.client else "unknown"
            key = uid or f"ip:{client_host}"
            ok, retry = _rate_limiter.acquire(tier, key)
            if not ok:
                wait = int(retry) + 1
                return JSONResponse(
                    status_code=429,
                    content={"detail": "rate limited", "retry_after": wait},
                    headers={"Retry-After": str(wait)},
                )
    return await call_next(request)


@app.middleware("http")
async def _enforce_api_key(request, call_next):
    """API 鉴权门禁（AUTH_ENABLED=true 时生效）。

    默认关闭以保持本地 demo / 前端 / 测试零改动；部署多租户时开启。
    公开端点（/ /health /docs /openapi.json /redoc）始终放行，其余需带 X-API-Key。
    USER_API_KEYS 非空时，key 会绑定到 user_id，并覆盖客户端传入的 user_id。
    未配置 USER_API_KEYS 时退回共享 API_KEY，仅做访问门禁，兼容旧部署。
    """
    if settings.auth_enabled:
        path = request.url.path.rstrip("/")
        # 仅放行前端入口和静态资源；不能用 /web 前缀，否则 /webhook/* 也会被放行。
        public_paths = {"", "/health", "/docs", "/openapi.json", "/redoc", "/web"}
        is_public = path in public_paths or path.startswith("/web/assets/")
        if not is_public:
            api_key = request.headers.get("X-API-Key")
            bound_user_id = settings.user_id_for_api_key(api_key)
            if settings.user_api_keys:
                if not bound_user_id:
                    return JSONResponse(status_code=401, content={"detail": "invalid or missing X-API-Key"})
                request.state.auth_user_id = bound_user_id
            elif not hmac.compare_digest(api_key or "", settings.api_key):
                return JSONResponse(status_code=401, content={"detail": "invalid or missing X-API-Key"})
    return await call_next(request)


agent = AudioVisualAgent()

# 用户画像服务（计划 §16.4）：复用 agent 的 store/memory，无独立状态。
from app.services.history import HistoryService
from app.services.profile import UserProfileService

profile_service = UserProfileService(agent.store, agent.memory)
history_service = HistoryService(agent.store)


def _effective_user_id(
    request: Request,
    provided_user_id: str | None,
    *,
    bind_in_anonymous_mode: bool = False,
) -> str:
    """解析请求应归属的用户 ID。

    优先使用 API key 绑定的 auth_user_id（多租户/鉴权模式）；未鉴权时退回 body
    里的 provided_user_id。对于播放代理等涉及真实付费凭证的端点，可设置
    ``bind_in_anonymous_mode=True``：在匿名模式（AUTH_ENABLED=false）下不信任
    客户端传入的 user_id，统一固定为 ``web_user``，防止显式借用他人 VIP cookie。
    """
    auth_user_id = getattr(request.state, "auth_user_id", None)
    if auth_user_id:
        return auth_user_id
    if bind_in_anonymous_mode and not settings.auth_enabled:
        return "web_user"
    return provided_user_id or "web_user"


def _frontend_build_hash() -> str:
    index = Path(__file__).resolve().parents[1] / "web" / "dist" / "index.html"
    try:
        text = index.read_text(encoding="utf-8")
    except OSError:
        return ""
    marker = "/assets/index-"
    if marker not in text:
        return ""
    suffix = text.split(marker, 1)[1]
    return suffix.split(".", 1)[0]


def _last_smoke_status() -> dict[str, Any]:
    report = Path(__file__).resolve().parents[2] / "artifacts" / "long_dialogue_smoke_report.json"
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"available": False}
    items = data if isinstance(data, list) else data.get("checks", [])
    total = len(items)
    passed = sum(1 for item in items if item.get("ok"))
    return {"available": True, "passed": passed, "total": total}


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _clean_artist_bio_text(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", "", text or "")
    cleaned = re.sub(r"Read more on Last\.fm.*$", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \n\t-")
    return cleaned


def _bio_needs_supplement(text: str) -> bool:
    cleaned = _clean_artist_bio_text(text)
    return not cleaned or len(cleaned) < 260 or "read more on last.fm" in (text or "").lower()


def _bio_needs_localization(text: str) -> bool:
    cleaned = _clean_artist_bio_text(text)
    if not cleaned:
        return False
    cjk_count = len(_CJK_RE.findall(cleaned))
    ascii_letters = len(re.findall(r"[A-Za-z]", cleaned))
    return ascii_letters >= max(30, cjk_count * 2)


def _bio_is_cjk_rich(text: str) -> bool:
    cleaned = _clean_artist_bio_text(text)
    if not cleaned:
        return False
    cjk_count = len(_CJK_RE.findall(cleaned))
    ascii_letters = len(re.findall(r"[A-Za-z]", cleaned))
    return cjk_count >= max(24, ascii_letters)


def _llm_provider_available(provider: Any) -> bool:
    if not hasattr(provider, "agenerate"):
        return False
    # 测试/热更新场景里，全局 agent 可能在无 key 时以 MockLLM 启动，随后再补上
    # settings.llm_api_key 并对 provider 打桩。此时按“类名是否 Mock”判不可用，会把
    # 本可工作的本地化链路直接短路成空白 bio。
    return provider.__class__.__name__ != "MockLLM" or bool(settings.llm_api_key)


def _prefer_bio(primary: str, fallback: str) -> str:
    primary_clean = _clean_artist_bio_text(primary)
    fallback_clean = _clean_artist_bio_text(fallback)
    if not fallback_clean:
        return primary_clean
    if not primary_clean:
        return fallback_clean
    if _bio_needs_localization(primary_clean) and len(fallback_clean) >= 120:
        return fallback_clean
    if _bio_needs_supplement(primary_clean) and len(fallback_clean) > len(primary_clean):
        return fallback_clean
    return primary_clean


def _merge_bio_context(primary: str, fallback: str) -> str:
    primary_clean = _clean_artist_bio_text(primary)
    fallback_clean = _clean_artist_bio_text(fallback)
    if not primary_clean:
        return fallback_clean[:2600]
    if not fallback_clean or fallback_clean in primary_clean:
        return primary_clean[:2600]
    if primary_clean in fallback_clean:
        return fallback_clean[:2600]
    return f"{primary_clean}\n\n补充资料：{fallback_clean}"[:2600]


def _best_bio_fallback(primary: str, fallback: str) -> str:
    primary_clean = _clean_artist_bio_text(primary)
    fallback_clean = _clean_artist_bio_text(fallback)
    if _bio_is_cjk_rich(fallback_clean):
        return fallback_clean[:1600]
    if _bio_is_cjk_rich(primary_clean):
        return primary_clean[:1600]
    return ""


async def _localize_artist_bio(artist: str, bio: str, tags: list[str], fallback_bio: str = "") -> str:
    cleaned = _clean_artist_bio_text(bio)
    if not cleaned:
        return _clean_artist_bio_text(fallback_bio)[:1600]
    if not _bio_needs_localization(cleaned):
        return cleaned[:1600]
    if not _llm_provider_available(agent.llm_fast):
        return _best_bio_fallback(cleaned, fallback_bio)

    tag_text = "、".join(tags[:5])
    from app.prompts.untrusted_boundary import UNTRUSTED_CONTENT_RULE
    system = (
        "你是严谨的音乐资料编辑。只基于给定资料改写，不补充额外事实。"
        "把英文资料整理成自然、完整的中文歌手介绍。" + UNTRUSTED_CONTENT_RULE
    )
    async def _rewrite(source: str, timeout: float) -> str:
        from app.prompts.untrusted_boundary import strip_directive_phrases, wrap_untrusted
        safe_source = wrap_untrusted(strip_directive_phrases(source), "歌手资料")
        prompt = (
            f"歌手：{artist}\n"
            f"标签：{tag_text or '无'}\n\n"
            f"资料：{safe_source}\n\n"
            "请输出 320-700 字中文简介，优先交代身份背景、创作/制作特点、风格线索、代表阶段或影响力。"
            "若资料涉及多个阶段，尽量覆盖从早期出道到后续代表阶段的变化。"
            "行文要像资料卡，不要只写两三句概述；重点艺人可适当写得更完整。"
            "不要出现“Read more on Last.fm”，不要保留英文残句，也不要编造未出现的信息。"
        )
        rewritten = await asyncio.wait_for(
            agent.llm_fast.agenerate(prompt, system=system, temperature=0.2),
            timeout=timeout,
        )
        normalized = _clean_artist_bio_text(rewritten)
        return normalized[:1600] if normalized and not _bio_needs_localization(normalized) else ""

    for source, timeout in ((cleaned[:1800], 12.0), (cleaned[:1100], 8.0)):
        try:
            rewritten = await _rewrite(source, timeout)
        except Exception:
            logger.debug("Artist bio localization failed for %s", artist, exc_info=True)
            continue
        if rewritten:
            return rewritten
    return _best_bio_fallback(cleaned, fallback_bio)

# ---- 挂载 Web 前端 & Bot 路由 ----
from app.api.web_routes import router as _web_router

app.include_router(_web_router)

try:
    from app.api.bot_routes import router as _bot_router
    app.include_router(_bot_router)
except ImportError:
    pass  # bot_routes 尚未创建时不报错

try:
    from app.api.auth_routes import router as _auth_router
    app.include_router(_auth_router)
except ImportError:
    pass  # auth_routes 尚未创建时不报错


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    try:
        agent.store.list_keys("assets")
        checks["store"] = True
    except Exception:
        logger.exception("Health check failed for JsonStore")
        checks["store"] = False

    llm_mode = "mock" if agent.llm.__class__.__name__ == "MockLLM" else "configured"
    checks["llm"] = llm_mode == "mock" or bool(settings.llm_api_key)
    details["llm_mode"] = llm_mode
    details["store_root"] = str(agent.store.root)
    details["store_candidates"] = settings.store_candidates
    details["duplicate_store_warning"] = sum(count > 0 for count in settings.store_candidates.values()) > 1
    details["resource_library_path"] = str(settings.resource_library_path)
    details["frontend_build_hash"] = _frontend_build_hash()
    details["auth_mode"] = (
        "per_user_key" if settings.auth_enabled and settings.user_api_keys
        else "shared_key" if settings.auth_enabled
        else "disabled"
    )
    details["allowed_origins"] = ",".join(settings.allowed_origins)
    details["smoke"] = _last_smoke_status()
    details["sources"] = {
        "netease": True,
        "bilibili": True,
        "youtube": True,
        "lastfm": bool(settings.lastfm_api_key),
        "tavily": bool(settings.tavily_api_key),
    }

    status = "ok" if all(checks.values()) else "degraded"
    return {"status": status, "checks": checks, "details": details}


@app.get("/assets")
def list_assets():
    return {"assets": [a.model_dump(mode="json") for a in agent.list_assets()]}


def _check_ingest_allowed(url: str) -> None:
    """入库前的统一治理闸：URL 校验 + 数量上限。失败 raise HTTPException（400/507）。"""
    try:
        validate_ingest_url(url)
    except IngestURLError as exc:
        raise HTTPException(status_code=400, detail=f"invalid or disallowed ingest url: {exc}") from exc
    if settings.max_assets > 0:
        try:
            current = len(agent.list_assets())
        except Exception:
            current = 0
        if current >= settings.max_assets:
            raise HTTPException(status_code=507, detail="asset library size limit reached")


@app.post("/assets/ingest")
def ingest(request: IngestRequest):
    _check_ingest_allowed(request.url)
    asset = agent.ingest_video(request.url, force_refresh=request.force_refresh)
    return asset


@app.post("/assets/ingest_full")
def ingest_full(request: IngestRequest):
    """完整入库：解析 URL → 联网识别歌名歌手 → 生成片段/曲风。

    Web 前端用这一个调用复刻 Streamlit 的三步流程（之前 Web 只调 ingest，
    导致入库的音视频停在占位标题、未识别——本端点修复该回归）。
    enrich 失败不阻断（标题至少有 URL 解析的结果），analyze 仍会执行。
    """
    _check_ingest_allowed(request.url)
    asset = agent.ingest_video(request.url, force_refresh=request.force_refresh)
    try:
        enriched = agent.enrich_asset(asset.asset_id, use_network=True)
        asset = enriched.asset
    except Exception:
        logger.warning("enrich step failed during ingest_full for %s", asset.asset_id, exc_info=True)
    # 分类（enrich 拿到真实歌名/歌手后）：补 genre/mood + 估算 tempo/energy。
    # 否则入库的歌只剩 DemoAnalyzer 的「未分类」，没标签、进不了推荐/品味。
    try:
        classified = agent.classify_asset(asset.asset_id)
        if classified is not None:
            asset = classified
    except Exception:
        logger.warning("classify step failed during ingest_full for %s", asset.asset_id, exc_info=True)
    try:
        asset, _ = agent.analyze_media(asset.asset_id, force_refresh=request.force_refresh)
    except Exception:
        logger.warning("analyze step failed during ingest_full for %s", asset.asset_id, exc_info=True)
    return asset


@app.post("/assets/{asset_id}/enrich")
def enrich(asset_id: str, request: EnrichRequest):
    try:
        return agent.enrich_asset(asset_id, use_network=request.use_network)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/assets/{asset_id}/analyze")
def analyze(asset_id: str, force_refresh: bool = Query(default=False)):
    try:
        asset, segments = agent.analyze_media(asset_id, force_refresh=force_refresh)
        return {"asset": asset, "segments": segments}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/assets/backfill_features")
def backfill_features(user_id: str | None = Query(default=None)):
    """一次性回填：为 tempo/energy 为 None 的资产按 genre/mood 标签估算填充。

    现网全部为 None（网易云无音频可分析，DemoAnalyzer 保持 None）。回填后推荐 energy/tempo
    锚与 tempo_range 才有真实信号。估算值用 features_source='estimated' 标注，非 measured。
    """
    result = agent.backfill_estimated_features()
    if user_id:
        # 可选：立即刷新品味档案，让 preferred_energy/tempo_range 马上吃上估算值；
        # 不传则在下次推荐时按当前资产懒重算。
        try:
            library = [a for a in agent.list_assets() if a.status == "analyzed"]
            agent.memory.refresh_taste_profile(user_id, library)
        except Exception:
            logger.warning("taste refresh after backfill failed for user=%s", user_id, exc_info=True)
    return {"backfilled": True, **result}


@app.post("/assets/cleanup_play_pollution")
def cleanup_play_pollution(user_id: str | None = Query(default=None)):
    """一次性清理：删除 source=external 且无标签的「播放自动入库」垃圾，并把 local 未分类
    重新分类。修旧 bug（play 曾把每首播过的歌入库且无标签）。保守：只动无标签条目。
    """
    result = agent.cleanup_play_pollution(user_id=user_id)
    return {"cleaned": True, **result}


@app.delete("/assets/{asset_id}")
def delete_asset(asset_id: str, request: Request, user_id: str | None = Query(default=None)):
    deleted = agent.delete_asset(asset_id, user_id=_effective_user_id(request, user_id) if user_id else None)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Unknown asset_id: {asset_id}")
    return {"deleted": True, "asset_id": asset_id}


@app.delete("/cache")
def clear_cache(preserve_memory: bool = Query(default=True)):
    return {"cleared": agent.clear_cache(preserve_memory=preserve_memory), "preserve_memory": preserve_memory}


@app.post("/rate")
def rate_asset(payload: RatingRequest, request: Request):
    uid = _effective_user_id(request, payload.user_id)
    try:
        memory = agent.rate_asset(uid, payload.asset_id, payload.score)
        return {
            "rated": True,
            "asset_id": payload.asset_id,
            "score": payload.score,
            "taste_updated": True,
            "top_genres": memory.taste_profile.top_genres[:5] if memory.taste_profile else [],
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/ratings/{user_id}")
def get_ratings(user_id: str, request: Request):
    memory = agent.memory.get_memory(_effective_user_id(request, user_id))
    return {"ratings": [r.model_dump(mode="json") for r in memory.ratings]}


@app.post("/recommend/daily")
def daily_recommend(payload: DailyRequest, request: Request):
    return agent.daily_recommend(
        _effective_user_id(request, payload.user_id), payload.time_of_day,
        no_local=payload.no_local,
    )


@app.get("/recommend/daily/{user_id}")
def get_daily(user_id: str, request: Request):
    return agent.daily_recommend(_effective_user_id(request, user_id))


@app.get("/assets/{asset_id}/similar")
def similar_assets(asset_id: str, top_k: int = 5):
    try:
        results = agent.find_similar_assets(asset_id, top_k)
        return {"similar_assets": [r.model_dump(mode="json") for r in results]}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/search")
def search(payload: SearchRequest, request: Request):
    return agent.search(_effective_user_id(request, payload.user_id), payload.query, payload.include_external, payload.top_k)


@app.post("/lyrics")
def get_lyrics(payload: LyricsRequest, request: Request):
    uid = _effective_user_id(request, payload.user_id)
    agent._apply_netease_cookie(uid)  # 带登录 cookie，避免歌词接口匿名限流
    return agent.get_lyrics(payload.title, payload.artist, payload.source_id)


@app.post("/listen")
def listen(payload: ListenRequest, request: Request):
    memory = agent.record_listen(
        _effective_user_id(request, payload.user_id),
        payload.asset_id,
        payload.duration,
        payload.completed,
        payload.context,
        title=payload.title,
        artist=payload.artist,
        cover_url=payload.cover_url,
        source=payload.source,
        source_id=payload.source_id,
    )
    return {"memory_updated": True, "history_count": len(memory.listening_history)}


@app.get("/history/listening/{user_id}")
def list_listening_history(user_id: str, request: Request, limit: int = Query(default=100, ge=1, le=200)):
    """听歌记录（最近在前）。每条回填展示元数据：
    - 新格式事件（写入时已带 title）直接用；
    - 旧格式事件（只有 asset_id）用曲库回查补 title/artist/cover；
    - 在线曲/已删曲目（asset_id 不在库里）标 available=False，前端展示「已移除/在线曲目」。
    """
    uid = _effective_user_id(request, user_id)
    memory = agent.memory.get_memory(uid)
    by_id = {a.asset_id: a for a in agent.list_assets()}
    events = list(reversed(memory.listening_history))[:limit]
    items = []
    for ev in events:
        title, artist, cover, available = ev.title, ev.artist, ev.cover_url, bool(ev.title)
        if not available:
            asset = by_id.get(ev.asset_id)
            if asset is not None:
                title = asset.title or ev.asset_id
                artist = asset.artist or ""
                cover = asset.cover_url or ""
                available = True
        items.append({
            "asset_id": ev.asset_id,
            "title": title,
            "artist": artist,
            "cover_url": cover,
            "source": ev.source,
            "source_id": ev.source_id,
            "timestamp": ev.timestamp,
            "duration_listened": ev.duration_listened,
            "completed": ev.completed,
            "available": available,
        })
    return {"items": items}


@app.post("/chat")
async def chat(payload: ChatRequest, request: Request):
    history = [{"role": m.role, "content": m.content} for m in payload.history]
    return await agent.chat_async(
        _effective_user_id(request, payload.user_id), payload.message,
        history=history or None, thread_id=payload.thread_id,
    )


@app.post("/agent/run")
async def agent_run(payload: ChatRequest, request: Request):
    history = [{"role": m.role, "content": m.content} for m in payload.history]
    return await agent.chat_async(
        _effective_user_id(request, payload.user_id), payload.message,
        history=history or None, thread_id=payload.thread_id,
    )


@app.post("/agent/stream")
async def agent_stream(payload: ChatRequest, request: Request):
    uid = _effective_user_id(request, payload.user_id)
    thread_id = payload.thread_id or f"{uid}:default"
    history = [{"role": m.role, "content": m.content} for m in payload.history]

    async def events():
        from app.services.tools import checkpoint_store, trace_store

        run_id = hashlib.sha256(f"{thread_id}:{time.time_ns()}".encode()).hexdigest()[:24]
        started = time.monotonic()
        checkpoint_store.touch_thread(thread_id)
        trace_store.start_run(run_id, thread_id, uid, payload.message)
        status = "ok"
        try:
            async for event in agent.stream_chat_async(
                uid, payload.message, history=history or None, thread_id=thread_id, run_id=run_id
            ):
                if await request.is_disconnected():
                    status = "cancelled"
                    break
                yield f"data: {json.dumps(event.model_dump(mode='json'), ensure_ascii=False)}\n\n"
                yield ": heartbeat\n\n"
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except Exception:
            status = "error"
            logger.exception("Agent stream failed")
            raise
        finally:
            trace_store.finish_run(run_id, status, (time.monotonic() - started) * 1000)

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/history/chat/{user_id}")
def list_chat_history(user_id: str, request: Request):
    uid = _effective_user_id(request, user_id)
    threads = history_service.list_chat_threads(uid)
    return {"threads": [thread.model_dump(mode="json") for thread in threads]}


@app.get("/history/chat/{user_id}/{thread_id}")
def get_chat_history_thread(user_id: str, thread_id: str, request: Request):
    uid = _effective_user_id(request, user_id)
    thread = history_service.get_chat_thread(uid, thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="chat thread not found")
    return thread


@app.post("/history/chat/turn")
def save_chat_history_turn(payload: ChatHistoryTurnRequest, request: Request):
    uid = _effective_user_id(request, payload.user_id)
    thread = history_service.append_chat_turn(
        uid,
        payload.thread_id,
        payload.user_message,
        payload.assistant_message,
        cards=payload.cards,
        trace_summary=payload.trace_summary,
    )
    return {"saved": True, "thread": thread.model_dump(mode="json")}


@app.delete("/history/chat/{user_id}")
def clear_chat_history(user_id: str, request: Request):
    uid = _effective_user_id(request, user_id)
    return {"deleted": history_service.clear_chat_threads(uid)}


@app.delete("/history/chat/{user_id}/{thread_id}")
def delete_chat_history_thread(user_id: str, thread_id: str, request: Request):
    uid = _effective_user_id(request, user_id)
    deleted = history_service.delete_chat_thread(uid, thread_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="chat thread not found")
    return {"deleted": True}


@app.get("/history/recommendations/{user_id}")
def list_recommendation_history(user_id: str, request: Request):
    uid = _effective_user_id(request, user_id)
    items = history_service.list_recommendations(uid)
    return {"recommendations": [item.model_dump(mode="json") for item in items]}


@app.post("/history/recommendations")
def save_recommendation_history(payload: RecommendationHistoryRequest, request: Request):
    uid = _effective_user_id(request, payload.user_id)
    item = history_service.save_recommendation(
        uid,
        payload.query,
        answer=payload.answer,
        cards=payload.cards,
        thread_id=payload.thread_id,
        ttl_days=payload.ttl_days,
    )
    return {"saved": True, "recommendation": item.model_dump(mode="json")}


@app.delete("/history/recommendations/{user_id}/{record_id}")
def delete_recommendation_history(user_id: str, record_id: str, request: Request):
    uid = _effective_user_id(request, user_id)
    deleted = history_service.delete_recommendation(uid, record_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="recommendation history not found")
    return {"deleted": True}


@app.post("/agent/resume")
async def agent_resume(payload: AgentResumeRequest, request: Request):
    from app.services.tools import checkpoint_store, tool_runtime
    from app.tools.contracts import ToolCall, ToolContext

    uid = _effective_user_id(request, payload.user_id)
    resolved = checkpoint_store.resolve(payload.action_id, payload.thread_id, uid, payload.approved)

    async def events():
        if resolved is None:
            event = {"type": "error", "content": "确认请求不存在、已处理或不属于当前会话。", "payload": {}}
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            return
        yield f"data: {json.dumps({'type': 'resumed', 'content': '已恢复操作。', 'payload': {'action_id': payload.action_id, 'approved': payload.approved}}, ensure_ascii=False)}\n\n"
        graph_resumed = False
        if agent.graph is not None and agent.graph.checkpointing_ready:
            try:
                async for event in agent.graph.resume(
                    thread_id=payload.thread_id,
                    action_id=payload.action_id,
                    approved=payload.approved,
                ):
                    graph_resumed = True
                    yield f"data: {json.dumps(event.model_dump(mode='json'), ensure_ascii=False)}\n\n"
            except Exception:
                logger.debug(
                    "LangGraph resume unavailable for action %s; using compatibility path",
                    payload.action_id,
                    exc_info=True,
                )
            if graph_resumed:
                return
        if not payload.approved:
            return
        # Compatibility for pending actions created before graph interrupts were enabled.
        arguments = {**resolved["arguments"], "confirm": True}
        result = await tool_runtime.execute(
            ToolCall(call_id=payload.action_id, name=resolved["tool"], arguments=arguments),
            ToolContext(
                thread_id=payload.thread_id,
                user_id=uid,
                query="恢复已确认的账号操作",
                confirmation={"action_id": payload.action_id, "approved": True},
                agent=agent,
            ),
        )
        event = {
            "type": "tool_result",
            "content": result.summary or (result.error.message if result.error else "操作完成。"),
            "payload": result.model_dump(mode="json"),
        }
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


# ── Discover / Browse ──

@app.post("/discover/classify", response_model=DiscoverQueryClassification)
def discover_classify(request: DiscoverQueryRequest):
    """Classify Discover input before the frontend launches expensive detail requests."""
    return agent.classify_discover_query(request.query)


@app.post("/discover/search", response_model=SearchResponse)
async def discover_search(payload: SearchRequest, request: Request):
    """Discover 专用搜索：本地与在线源拆成两次独立请求，互不阻塞。

    前端对同一 query 并发两次调用：
    - 本地那次（include_external=False）只读曲库，毫秒级返回，先把结果铺出来；
    - 在线那次（external_only=True）只跑网易云搜索，不重复本地检索，因此能给
      它更宽裕的时限（12s）。网易云间歇限流时靠多端点轮询+重试扛，需要时间多试
      几次，故不再 6s 硬超时丢结果——超时/失败时如实回报，而不是谎称已展示。

    两条路彼此独立：在线慢不拖累本地，本地结果也永不因在线超时被丢弃。
    """
    user_id = _effective_user_id(request, payload.user_id)

    # 在线专用请求：跳过本地检索，只补在线候选。
    if payload.external_only:
        response = SearchResponse()
        try:
            external = await asyncio.wait_for(
                asyncio.to_thread(
                    agent.search_web_music,
                    payload.query,
                    payload.top_k,
                    payload.query,
                ),
                timeout=12.0,
            )
            response.external = external[:payload.top_k]
            response.agent_trace.append(f"discover_external_hits={len(external)}")
            response.summary = (
                f"在线找到 {len(response.external)} 首相关曲目。"
                if response.external else "在线来源这次没有匹配，换个说法试试。"
            )
        except TimeoutError:
            response.agent_trace.append("discover_external_timeout=12s")
            response.summary = "在线来源响应较慢，稍后可再点搜索补充在线结果。"
        except Exception:
            logger.debug("Discover external search failed for %s", payload.query, exc_info=True)
            response.agent_trace.append("discover_external_error=true")
            response.summary = "在线来源暂不可用。"
        return response

    # 本地专用请求：只读曲库，秒级返回。
    return await asyncio.to_thread(
        agent.search, user_id, payload.query, False, payload.top_k,
    )

@app.post("/discover/browse")
def discover_browse(request: BrowseRequest):
    """按曲风/心情/场景浏览歌曲。轻量双路搜索（网易云歌单 + Last.fm 标签），不走 LLM，秒级返回。

    seed 实现"换一批"：按 seed 轮换搜索关键词、加大候选歌单数，使同一分类能取到
    不同批次的曲目（否则同一关键词 → 同一批歌单 → 每次结果一模一样）。
    """
    from app.search.netease_playlist import search_and_extract
    from app.sources.lastfm_client import LastfmClient

    tracks: list[dict] = []
    seen: set[str] = set()

    def _dedup_add(title: str, artist: str, extra: dict) -> bool:
        key = f"{title.strip().lower()}|{artist.strip().lower()}"
        if key in seen or not title.strip():
            return False
        seen.add(key)
        tracks.append({"title": title.strip(), "artist": artist.strip(), **extra})
        return True

    # 关键词轮换：seed 决定从哪组词开始，保证"换一批"能拿到不同歌单。
    keywords = [f"{request.value}音乐", f"{request.value}歌曲", f"{request.value}经典",
                f"{request.value}热门", f"{request.value}精选"]
    start = request.seed % len(keywords)
    ordered_kw = keywords[start:] + keywords[:start]
    # 候选歌单数随 seed 增长（2/3/4），从更多歌单里收歌，结果更丰富。
    max_playlists = 2 + (request.seed % 3)

    # 1) 网易云歌单搜索（轮换关键词）
    try:
        for kw in ordered_kw:
            extracted = search_and_extract(kw, max_playlists=max_playlists, tracks_per_playlist=request.limit)
            for t in extracted:
                _dedup_add(t.title, t.artist, {
                    "source": t.source or "netease",
                    "source_id": t.external_id or "",
                    "cover_url": t.cover_url,
                    "playback_url": t.playback_url,
                })
                if len(tracks) >= request.limit:
                    break
            if len(tracks) >= request.limit:
                break
    except Exception:
        logger.debug("Browse netease search failed for %s", request.value, exc_info=True)

    # 2) Last.fm 标签搜索（genre 用英文 tag，mood/scene 用网易云结果即可）
    if len(tracks) < request.limit and settings.lastfm_api_key:
        try:
            lfm = LastfmClient(settings.lastfm_api_key)
            _GENRE_EN = {
                "流行": "pop", "摇滚": "rock", "电子": "electronic", "说唱": "hip-hop",
                "R&B": "r&b", "爵士": "jazz", "民谣": "folk", "古典": "classical",
                "国风": "chinese", "金属": "metal",
            }
            _MOOD_EN = {
                "放松": "chill", "治愈": "healing", "运动": "workout", "专注": "focus",
                "浪漫": "romantic", "伤感": "sad", "深夜": "night", "清晨": "morning",
                "通勤": "commute", "派对": "party",
            }
            tag = (_GENRE_EN if request.category == "genre" else _MOOD_EN).get(request.value, "")
            # 未映射的 value（如新加的曲风/场景）用 value 自身做 tag 兜底，
            # 否则 Last.fm 路直接放弃，只剩网易云一条路，一旦限流就空。
            if not tag:
                tag = request.value.strip().lower()
            if tag:
                for t in lfm.get_tag_top_tracks(tag, limit=request.limit):
                    _dedup_add(t["title"], t.get("artist", ""), {
                        "source": "lastfm", "cover_url": None,
                    })
                    if len(tracks) >= request.limit:
                        break
        except Exception:
            logger.debug("Browse Last.fm tag search failed for %s", request.value, exc_info=True)

    result = tracks[:request.limit]
    # 两路皆空时给明确提示，前端据此显示"换一批试试"，而不是一张白板。
    if not result:
        summary = "该分类暂时没搜到结果，可能是接口限流，点「换一批」再试试。"
    else:
        summary = f"为你找到 {len(result)} 首{request.value}相关歌曲"
    return {"tracks": result, "summary": summary}


@app.post("/discover/trending")
def discover_trending(request: TrendingRequest):
    """热门趋势：网易云榜单与 Last.fm 全球榜，附真实来源和更新时间。"""
    from app.search.netease_playlist import get_playlist_detail
    from app.sources.lastfm_client import LastfmClient

    per_chart = min(request.limit, 8)

    # 网易云官方榜单 ID
    _NETEASE_CHARTS = [
        {"name": "网易云热歌榜", "id": 3778678, "expected": "热歌榜", "icon": "🔥"},
        {"name": "网易云飙升榜", "id": 19723756, "expected": "飙升榜", "icon": "📈"},
        {"name": "美国 Billboard", "id": 60198, "expected": "美国Billboard榜", "icon": "🇺🇸"},
        {"name": "UK 排行榜", "id": 180106, "expected": "UK排行榜周榜", "icon": "🇬🇧"},
        {"name": "Beatport 电子榜", "id": 3812895, "expected": "Beatport全球电子舞曲榜", "icon": "🎛️"},
    ]

    charts: list[dict] = []

    def _load_netease_chart(chart_def: dict[str, Any]) -> dict[str, Any] | None:
        try:
            detail = get_playlist_detail(chart_def["id"], limit=per_chart)
            if detail and detail["tracks"]:
                actual_name = detail["name"].replace(" ", "")
                expected_name = chart_def["expected"].replace(" ", "")
                if actual_name != expected_name:
                    logger.warning(
                        "Trending chart id=%s expected %r but received %r; skipping mismatched data",
                        chart_def["id"], chart_def["expected"], detail["name"],
                    )
                    return None
                return {
                    "name": chart_def["name"],
                    "icon": chart_def["icon"],
                    "chart_id": str(chart_def["id"]),
                    "source": "netease",
                    "source_name": detail["name"],
                    "updated_at": detail["updated_at"],
                    "tracks": [{
                        "title": t.title, "artist": t.artist,
                        "source": t.source or "netease",
                        "source_id": t.external_id or "",
                        "cover_url": t.cover_url,
                        "playback_url": t.playback_url,
                    } for t in detail["tracks"]],
                }
        except Exception:
            logger.debug("Trending chart %s failed", chart_def["name"], exc_info=True)
        return None

    # 1) 各榜单彼此独立，并行获取，避免一个慢源拖住整个发现页。
    with ThreadPoolExecutor(max_workers=len(_NETEASE_CHARTS)) as executor:
        for chart in executor.map(_load_netease_chart, _NETEASE_CHARTS):
            if chart:
                charts.append(chart)

    # 2) Last.fm 全球榜
    if settings.lastfm_api_key:
        try:
            lfm = LastfmClient(settings.lastfm_api_key)
            raw = lfm.get_chart_top_tracks(limit=per_chart)
            if raw:
                charts.append({
                    "name": "Last.fm 全球榜",
                    "icon": "🌐",
                    "chart_id": "lastfm-global",
                    "source": "lastfm",
                    "source_name": "Last.fm Global Top Tracks",
                    "updated_at": None,
                    "tracks": [{
                        "title": t["title"], "artist": t.get("artist", ""),
                        "source": "lastfm", "source_id": "",
                        "cover_url": None, "playback_url": None,
                    } for t in raw],
                })
        except Exception:
            logger.debug("Trending Last.fm global chart failed", exc_info=True)

    return {"charts": charts}


@app.post("/artist/info")
async def artist_info(request: ArtistInfoRequest) -> ArtistInfoResponse:
    """获取歌手资料：简介、代表专辑、热门歌曲。

    三个外部请求（Last.fm 歌手资料 / Last.fm 代表专辑 / 网易云热门歌曲）并发执行，
    延迟从"三者之和"降到"最慢者"。旧实现串行，歌手卡加载明显偏慢。
    """
    import asyncio

    from app.models import ExternalTrack

    info = {
        "name": request.artist, "requested_name": request.artist, "matched": False,
        "image": "", "bio": "", "tags": [], "top_albums": [], "top_tracks": [],
    }

    lfm = None
    if settings.lastfm_api_key:
        from app.sources.lastfm_client import LastfmClient
        lfm = LastfmClient(settings.lastfm_api_key)

    async def _artist_data():
        if not lfm:
            return None
        try:
            return await asyncio.to_thread(lfm.get_artist_info, request.artist)
        except Exception:
            logger.debug("Last.fm artist info failed", exc_info=True)
            return None

    async def _albums():
        if not lfm:
            return []
        try:
            return await asyncio.to_thread(lfm.get_artist_top_albums, request.artist, 12)
        except Exception:
            logger.debug("Last.fm top albums failed", exc_info=True)
            return []

    async def _netease_albums():
        # 网易云专辑搜索拿到真实 album_id/track_count/cover；Last.fm 的专辑没 id，
        # 点进去还得按名字二次猜匹配（容易猜错）。网易云优先，Last.fm 仅补位。
        try:
            from app.sources.netease import search_netease_artist_albums
            return await asyncio.to_thread(search_netease_artist_albums, request.artist, 12)
        except Exception:
            logger.debug("NetEase artist albums failed", exc_info=True)
            return []

    async def _hot_tracks():
        try:
            raw_tracks = await asyncio.wait_for(
                asyncio.to_thread(agent.search_web_music, request.artist, 12, request.artist),
                timeout=5.5,
            )
            return [
                ExternalTrack(
                    external_id=t.external_id or "",
                    title=t.title, artist=t.artist or "",
                    cover_url=t.cover_url, source=t.source,
                    playback_url=t.playback_url, candidate_kind=t.candidate_kind,
                )
                for t in raw_tracks
                if agent.artist_name_matches(request.artist, t.artist or "")
                and t.title.strip().lower() not in {"热门歌曲", "热门单曲", "top songs", "popular songs"}
            ][:6]
        except TimeoutError:
            logger.debug("Artist hot tracks search timed out for %s", request.artist)
            return []
        except Exception:
            logger.debug("Artist hot tracks search failed", exc_info=True)
            return []

    async def _netease_image():
        # 网易云歌手头像（type=100）比 Last.fm 的 image 字段可靠得多——直接去搜一张图。
        try:
            from app.sources.netease import search_netease_artist_image
            return await asyncio.to_thread(search_netease_artist_image, request.artist)
        except Exception:
            logger.debug("NetEase artist image failed", exc_info=True)
            return None

    async def _web_bio(artist: str, prefer_cjk: bool = False) -> str:
        """Tavily 兜底简介——中文歌手 Last.fm 常缺/错(autocorrect 把 Asen 纠到挪威朋克 Aasen)。
        当 Last.fm 误识或无 bio 时，用搜索引擎抓一段背景简介，避免张冠李戴或空白。"""
        queries = (
            [f"{artist} 歌手 简介 生涯 风格 代表作", f"{artist} singer rapper biography career style"]
            if prefer_cjk else
            [f"{artist} singer rapper 歌手 简介 背景"]
        )
        items: list[dict[str, str]] = []
        try:
            from app.sources.web_search import search_web_info
            for query in queries:
                items = await asyncio.wait_for(
                    asyncio.to_thread(
                        search_web_info, query, 4, settings.tavily_api_key, 8,
                    ),
                    timeout=4.0,
                )
                if items:
                    break
        except Exception:
            return ""
        candidates: list[tuple[int, str]] = []
        for it in (items or []):
            content = _clean_artist_bio_text(it.get("content", ""))
            if len(content) < 40:
                continue
            cjk_count = len(_CJK_RE.findall(content))
            score = (cjk_count * 4 if prefer_cjk else cjk_count) + min(len(content), 600)
            candidates.append((score, content[:800]))
        if not candidates:
            return ""
        candidates.sort(key=lambda item: item[0], reverse=True)
        merged: list[str] = []
        for _, content in candidates:
            if any(content in existing or existing in content for existing in merged):
                continue
            merged.append(content)
            if len(merged) >= 2:
                break
        return "\n\n".join(merged)[:1800]

    artist_data, lfm_albums, ext, netease_img, netease_albums_raw = await asyncio.gather(
        _artist_data(), _albums(), _hot_tracks(), _netease_image(), _netease_albums()
    )

    # 交叉校验 Last.fm autocorrect：它可能把 "Asen" 纠正成同名异艺人(挪威朋克 Aasen)，
    # 导致 bio/tags 张冠李戴。网易云专辑的 artist 是可靠真相——不一致则弃用 Last.fm 文字资料。
    def _norm_name(s: str) -> str:
        return re.sub(r"[^a-z0-9一-鿿]+", "", (s or "").lower())

    netease_artist = next((a.get("artist") for a in (netease_albums_raw or []) if a.get("artist")), "")
    lfm_mismatch = bool(
        artist_data and netease_artist
        and _norm_name(artist_data.get("name", "")) != _norm_name(netease_artist)
    )
    if artist_data and not lfm_mismatch:
        info["name"] = artist_data.get("name") or request.artist
        info["bio"] = _clean_artist_bio_text(artist_data.get("bio", ""))
        info["tags"] = artist_data.get("tags", [])
    else:
        # Last.fm 误识或无 bio：回落 Tavily 搜索简介，避免错认成异艺人或留白。
        info["name"] = request.artist
        web_bio = await _web_bio(netease_artist or request.artist, prefer_cjk=True)
        if web_bio:
            info["bio"] = _clean_artist_bio_text(web_bio)
    # 头像优先网易云（可靠），Last.fm image 字段兜底（误识时不用 Last.fm 图）
    info["image"] = netease_img or (artist_data.get("image", "") if (artist_data and not lfm_mismatch) else "")

    # Last.fm bio 常偏短，中文艺人还可能留空；此时再用搜索补位。
    # 若只是英文但内容已经够完整，直接中文化改写即可，避免额外再跑一轮 web 搜索拖慢艺人卡。
    supplemental_bio = ""
    if _bio_needs_supplement(info["bio"]) or _bio_needs_localization(info["bio"]):
        supplemental_bio = await _web_bio(
            netease_artist or info["name"] or request.artist,
            prefer_cjk=_bio_needs_localization(info["bio"]),
        )
    bio_context = _merge_bio_context(info["bio"], supplemental_bio)
    info["bio"] = await _localize_artist_bio(
        info["name"],
        _prefer_bio(bio_context, supplemental_bio),
        info["tags"],
        fallback_bio=supplemental_bio,
    )

    # 代表专辑：网易云优先（带真实 album_id，点击直达专辑详情），Last.fm 补位。
    # 按归一化名称去重，避免两源同名专辑重复展示。
    top_albums: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for a in (netease_albums_raw or []):
        name = (a.get("name") or "").strip()
        if not name:
            continue
        key = name.strip().lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        top_albums.append({
            "id": a.get("id", ""),
            "name": name,
            "image": a.get("image", ""),
            "artist": a.get("artist") or info["name"],
            "track_count": a.get("track_count"),
        })
    for a in (lfm_albums or []):  # Last.fm: {name, image}，无 id，仅补位
        name = (a.get("name") or "").strip()
        if not name:
            continue
        key = name.strip().lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        top_albums.append({
            "id": "", "name": name, "image": a.get("image", ""),
            "artist": info["name"], "track_count": None,
        })
        if len(top_albums) >= 12:
            break
    info["top_albums"] = top_albums[:12]
    info["top_tracks"] = ext

    resolved_names = {info["name"]} if artist_data else set()
    resolved_names.update(a.get("artist", "") for a in top_albums)
    for track in ext:
        resolved_names.update(part for part in re.split(r"[、,/&]", track.artist or ""))
    resolved_names.discard("")
    info["matched"] = any(agent.artist_name_matches(request.artist, name) for name in resolved_names)

    return ArtistInfoResponse(**info)


@app.post("/artist/album_tracks")
async def artist_album_tracks(request: AlbumTracksRequest) -> AlbumTracksResponse:
    """获取专辑曲目：按网易云专辑原始顺序返回完整清单。"""
    from app.models import ArtistAlbum, ExternalTrack
    from app.sources.netease import fetch_netease_album_tracks, search_netease_album

    album_meta = None
    album_id = (request.album_id or "").strip()
    if not album_id:
        album_meta = await asyncio.to_thread(search_netease_album, request.artist, request.album)
        album_id = str((album_meta or {}).get("id") or "")

    detail = await asyncio.to_thread(fetch_netease_album_tracks, album_id, request.limit) if album_id else None
    if not detail:
        fallback_album = ArtistAlbum(
            id=album_id,
            name=request.album,
            image=(album_meta or {}).get("cover", ""),
            artist=request.artist,
            track_count=(album_meta or {}).get("track_count"),
        )
        return AlbumTracksResponse(album=fallback_album, tracks=[], summary=f"没找到《{request.album}》的专辑曲目。")

    album = ArtistAlbum(
        id=str(detail.get("id") or album_id),
        name=detail.get("name") or request.album,
        image=detail.get("cover") or (album_meta or {}).get("cover", ""),
        artist=detail.get("artist") or request.artist,
        track_count=detail.get("track_count") or len(detail.get("tracks") or []),
    )
    tracks = [
        ExternalTrack(
            external_id=t["song_id"],
            title=t["title"],
            artist=t.get("artist") or album.artist,
            album=t.get("album") or album.name,
            cover_url=t.get("cover") or album.image or None,
            source="netease",
            playback_url=f"https://music.163.com/song?id={t['song_id']}",
        )
        for t in detail.get("tracks", [])
    ]
    return AlbumTracksResponse(album=album, tracks=tracks, summary=f"已加载《{album.name}》{len(tracks)} 首曲目。")


# ── 收藏专辑 ──

@app.post("/album/save")
def save_album(payload: SaveAlbumRequest, request: Request):
    """收藏一张专辑：保存元数据 + 完整曲目，供「我的库」整张播放。"""
    from app.models import SavedAlbum

    uid = _effective_user_id(request, payload.user_id)
    album = SavedAlbum(
        album_id=payload.album_id, user_id=uid, name=payload.name,
        artist=payload.artist, image=payload.image, track_count=payload.track_count,
        tags=payload.tags,
        tracks=payload.tracks,
    )
    saved = agent.save_album(uid, album)
    return {"saved": True, "album": saved}


@app.get("/albums/saved/{user_id}")
def list_saved_albums(user_id: str, request: Request):
    return {"albums": [a.model_dump(mode="json") for a in agent.list_saved_albums(_effective_user_id(request, user_id))]}


@app.get("/album/saved/{user_id}/{album_id}")
def album_saved_status(user_id: str, album_id: str, request: Request):
    return {"saved": agent.is_album_saved(_effective_user_id(request, user_id), album_id)}


@app.delete("/album/saved/{user_id}/{album_id}")
def delete_saved_album(user_id: str, album_id: str, request: Request):
    deleted = agent.delete_saved_album(_effective_user_id(request, user_id), album_id)
    return {"deleted": deleted, "album_id": album_id}


@app.get("/taste/{user_id}")
def taste_profile(user_id: str, request: Request):
    return agent.get_taste_profile(_effective_user_id(request, user_id))


# ── 用户画像（可解释品味仪表盘）──

@app.get("/profile/{user_id}")
def get_profile(user_id: str, request: Request):
    """完整用户画像（计划 §13.1）：品味摘要 / 声音指纹 / 情绪地图 / 场景偏好 /
    艺术家关系 / 探索倾向 / 带置信度的可纠错洞察。数据不足时返回空状态引导。"""
    return profile_service.get_profile(_effective_user_id(request, user_id))


@app.post("/profile/insights/{insight_id}/feedback")
def profile_insight_feedback(insight_id: str, payload: ProfileInsightFeedbackRequest, request: Request):
    """用户纠错某条画像判断（计划 §13.2）：confirm/reject/temporary/disable_for_recommendation。"""
    uid = _effective_user_id(request, payload.user_id)
    try:
        return profile_service.update_insight_feedback(uid, insight_id, payload.action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/profile/insights/{user_id}/{insight_id}")
def delete_profile_insight(user_id: str, insight_id: str, request: Request):
    """删除某条 insight 的反馈，恢复默认（计划 §24）。"""
    uid = _effective_user_id(request, user_id)
    deleted = profile_service.delete_insight(uid, insight_id)
    return {"deleted": deleted, "insight_id": insight_id}


@app.delete("/profile/{user_id}")
def clear_profile(user_id: str, request: Request):
    """清除画像反馈数据（计划 §24，隐私可控性）。不删除底层偏好记忆。"""
    uid = _effective_user_id(request, user_id)
    cleared = profile_service.clear_profile_feedback(uid)
    return {"cleared": cleared, "user_id": uid}


@app.post("/taste/experiment/generate")
def generate_taste_experiment(payload: TasteExperimentRequest, request: Request):
    uid = _effective_user_id(request, payload.user_id)
    return agent.generate_taste_experiment(uid, payload.prompt, total=payload.total, online_only=payload.online_only)


@app.get("/taste/experiments/{user_id}")
def list_taste_experiments(user_id: str, request: Request):
    uid = _effective_user_id(request, user_id)
    return {"experiments": [exp.model_dump(mode="json") for exp in agent.list_taste_experiments(uid)]}


@app.get("/taste/experiment/{user_id}/{experiment_id}")
def get_taste_experiment(user_id: str, experiment_id: str, request: Request):
    uid = _effective_user_id(request, user_id)
    exp = agent.get_taste_experiment(uid, experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp


@app.post("/taste/experiment/feedback")
def taste_experiment_feedback(payload: TasteExperimentFeedbackRequest, request: Request):
    payload.user_id = _effective_user_id(request, payload.user_id)
    try:
        return agent.record_taste_experiment_feedback(payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/taste/experiment/report")
def taste_experiment_report(payload: TasteExperimentReportRequest, request: Request):
    uid = _effective_user_id(request, payload.user_id)
    try:
        return agent.summarize_taste_experiment(uid, payload.experiment_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/taste/experiment/{user_id}/{experiment_id}")
def delete_taste_experiment(user_id: str, experiment_id: str, request: Request):
    deleted = agent.delete_taste_experiment(_effective_user_id(request, user_id), experiment_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"deleted": True}


@app.post("/taste/experiment/regenerate")
def taste_experiment_regenerate(payload: TasteExperimentRegenerateRequest, request: Request):
    uid = _effective_user_id(request, payload.user_id)
    try:
        exp = agent.regenerate_taste_experiment_bucket(uid, payload.experiment_id, payload.bucket)
        return exp.model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/memory/update")
def update_memory(payload: MemoryUpdateRequest, request: Request):
    payload.user_id = _effective_user_id(request, payload.user_id)
    memory, changed = agent.update_memory(payload)
    return {"memory": memory, "updated": changed}


@app.post("/memory/feedback")
def memory_feedback(payload: FeedbackRequest, request: Request):
    payload.user_id = _effective_user_id(request, payload.user_id)
    try:
        memory = agent.record_feedback(payload)
        return {"memory": memory, "updated": True}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/feedback/dislike")
def dislike(payload: DislikeRequest, request: Request):
    payload.user_id = _effective_user_id(request, payload.user_id)
    memory = agent.record_dislike(payload)
    return {"updated": True, "memory": memory}


@app.get("/memory/{user_id}")
def get_memory(user_id: str, request: Request):
    return agent.memory.get_memory(_effective_user_id(request, user_id))


# ---- 排除规则（用户偏好设置） ----

@app.get("/exclusions/{user_id}")
def list_exclusions(user_id: str, request: Request):
    return {"rules": agent.memory.list_exclusions(_effective_user_id(request, user_id))}


@app.post("/exclusions/{user_id}")
def add_exclusion(user_id: str, body: dict[str, str], request: Request):
    uid = _effective_user_id(request, user_id)
    rule = body.get("rule", "").strip()
    if not rule:
        raise HTTPException(status_code=400, detail="rule is required")
    added = agent.memory.add_exclusion(uid, rule)
    return {"added": added, "rules": agent.memory.list_exclusions(uid)}


@app.delete("/exclusions/{user_id}/{rule:path}")
def remove_exclusion(user_id: str, rule: str, request: Request):
    uid = _effective_user_id(request, user_id)
    removed = agent.memory.remove_exclusion(uid, rule)
    return {"removed": removed, "rules": agent.memory.list_exclusions(uid)}


@app.get("/library/tracks")
def library_tracks(limit: int = Query(default=100, ge=1, le=500)):
    return {"tracks": [track.model_dump(mode="json") for track in agent.list_resource_tracks(limit)]}


@app.post("/playlist/generate")
def generate_playlist(payload: PlaylistRequest, request: Request):
    playlist = agent.generate_playlist(_effective_user_id(request, payload.user_id), payload.instruction)
    return playlist


@app.post("/playlist/from_assets")
def create_playlist_from_assets(payload: PlaylistFromAssetsRequest, request: Request):
    uid = _effective_user_id(request, payload.user_id)
    try:
        return agent.create_playlist_from_assets(uid, payload.name, payload.asset_ids, payload.description)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/journey/generate")
def generate_journey(payload: JourneyRequest, request: Request):
    return agent.generate_music_journey(_effective_user_id(request, payload.user_id), payload.instruction)


@app.post("/playlist/auto/{user_id}")
def auto_playlists(user_id: str, request: Request):
    playlists = agent.auto_playlists(_effective_user_id(request, user_id))
    return {"playlists": playlists}


@app.get("/playlists/{user_id}")
def list_playlists(user_id: str, request: Request):
    return {"playlists": agent.list_playlists(_effective_user_id(request, user_id))}


@app.delete("/playlist/{user_id}/{playlist_id}")
def delete_playlist(user_id: str, playlist_id: str, request: Request):
    deleted = agent.delete_playlist(_effective_user_id(request, user_id), playlist_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Playlist not found")
    return {"deleted": True}
