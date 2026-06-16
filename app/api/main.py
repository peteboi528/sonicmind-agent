from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse

from app.agent import AudioVisualAgent
from app.config import settings
from app.models import (
    ArtistInfoRequest,
    ArtistInfoResponse,
    BrowseRequest,
    ChatRequest,
    DailyRequest,
    DislikeRequest,
    EnrichRequest,
    FeedbackRequest,
    IngestRequest,
    JourneyRequest,
    ListenRequest,
    MemoryUpdateRequest,
    PlaylistRequest,
    RatingRequest,
    SearchRequest,
    TrendingRequest,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="智能影音推荐助手 API",
    description="音视频内容分析、个性化推荐、每日歌单、LLM 语义搜索。",
    version="0.3.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _enforce_api_key(request, call_next):
    """共享密钥鉴权门禁（AUTH_ENABLED=true 时生效）。

    默认关闭以保持本地 demo / 前端 / 测试零改动；部署多租户时开启。
    公开端点（/ /health /docs /openapi.json /redoc）始终放行，其余需带 X-API-Key。
    注意：共享密钥只挡「未授权访问」；彻底防「伪造他人 user_id」需 per-user key→user_id 绑定（后续扩展）。
    """
    if settings.auth_enabled:
        path = request.url.path.rstrip("/")
        if path not in {"", "/health", "/docs", "/openapi.json", "/redoc"}:
            if request.headers.get("X-API-Key") != settings.api_key:
                return JSONResponse(status_code=401, content={"detail": "invalid or missing X-API-Key"})
    return await call_next(request)


agent = AudioVisualAgent()

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
    details: dict[str, str] = {}

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

    status = "ok" if all(checks.values()) else "degraded"
    return {"status": status, "checks": checks, "details": details}


@app.get("/assets")
def list_assets():
    return {"assets": [a.model_dump(mode="json") for a in agent.list_assets()]}


@app.post("/assets/ingest")
def ingest(request: IngestRequest):
    asset = agent.ingest_video(request.url, force_refresh=request.force_refresh)
    return asset


@app.post("/assets/ingest_full")
def ingest_full(request: IngestRequest):
    """完整入库：解析 URL → 联网识别歌名歌手 → 生成片段/曲风。

    Web 前端用这一个调用复刻 Streamlit 的三步流程（之前 Web 只调 ingest，
    导致入库的音视频停在占位标题、未识别——本端点修复该回归）。
    enrich 失败不阻断（标题至少有 URL 解析的结果），analyze 仍会执行。
    """
    asset = agent.ingest_video(request.url, force_refresh=request.force_refresh)
    try:
        enriched = agent.enrich_asset(asset.asset_id, use_network=True)
        asset = enriched.asset
    except Exception:
        logger.warning("enrich step failed during ingest_full for %s", asset.asset_id, exc_info=True)
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


@app.delete("/assets/{asset_id}")
def delete_asset(asset_id: str, user_id: str | None = Query(default=None)):
    deleted = agent.delete_asset(asset_id, user_id=user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Unknown asset_id: {asset_id}")
    return {"deleted": True, "asset_id": asset_id}


@app.delete("/cache")
def clear_cache(preserve_memory: bool = Query(default=True)):
    return {"cleared": agent.clear_cache(preserve_memory=preserve_memory), "preserve_memory": preserve_memory}


@app.post("/rate")
def rate_asset(request: RatingRequest):
    try:
        memory = agent.rate_asset(request.user_id, request.asset_id, request.score)
        return {
            "rated": True,
            "asset_id": request.asset_id,
            "score": request.score,
            "taste_updated": True,
            "top_genres": memory.taste_profile.top_genres[:5] if memory.taste_profile else [],
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/ratings/{user_id}")
def get_ratings(user_id: str):
    memory = agent.memory.get_memory(user_id)
    return {"ratings": [r.model_dump(mode="json") for r in memory.ratings]}


@app.post("/recommend/daily")
def daily_recommend(request: DailyRequest):
    return agent.daily_recommend(request.user_id, request.time_of_day)


@app.get("/recommend/daily/{user_id}")
def get_daily(user_id: str):
    return agent.daily_recommend(user_id)


@app.get("/assets/{asset_id}/similar")
def similar_assets(asset_id: str, top_k: int = 5):
    try:
        results = agent.find_similar_assets(asset_id, top_k)
        return {"similar_assets": [r.model_dump(mode="json") for r in results]}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/search")
def search(request: SearchRequest):
    return agent.search(request.user_id, request.query, request.include_external, request.top_k)


@app.post("/listen")
def listen(request: ListenRequest):
    memory = agent.record_listen(request.user_id, request.asset_id, request.duration, request.completed, request.context)
    return {"memory_updated": True, "history_count": len(memory.listening_history)}


@app.post("/chat")
def chat(request: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in request.history]
    return agent.chat(request.user_id, request.message, history=history or None)


@app.post("/agent/run")
def agent_run(request: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in request.history]
    return agent.chat(request.user_id, request.message, history=history or None)


_SENTINEL = object()


def _safe_next(gen):
    """Wrap next() so StopIteration doesn't leak into run_in_executor."""
    try:
        return next(gen)
    except StopIteration:
        return _SENTINEL


@app.post("/agent/stream")
async def agent_stream(request: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in request.history]

    async def events():
        loop = asyncio.get_event_loop()
        gen = agent.stream_chat(request.user_id, request.message, history=history or None)
        while True:
            event = await loop.run_in_executor(None, _safe_next, gen)
            if event is _SENTINEL:
                break
            yield f"data: {json.dumps(event.model_dump(mode='json'), ensure_ascii=False)}\n\n"
            yield ": heartbeat\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


# ── Discover / Browse ──

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
        summary = f"该分类暂时没搜到结果，可能是接口限流，点「换一批」再试试。"
    else:
        summary = f"为你找到 {len(result)} 首{request.value}相关歌曲"
    return {"tracks": result, "summary": summary}


@app.post("/discover/trending")
def discover_trending(request: TrendingRequest):
    """热门趋势：官方榜单（网易云热歌榜/飙升榜/欧美榜 + Last.fm 全球榜）分组返回。"""
    from app.search.netease_playlist import get_playlist_tracks
    from app.sources.lastfm_client import LastfmClient

    per_chart = min(request.limit, 8)

    # 网易云官方榜单 ID
    _NETEASE_CHARTS = [
        {"name": "网易云热歌榜", "id": 3779629, "icon": "🔥"},
        {"name": "网易云飙升榜", "id": 19723756, "icon": "📈"},
        {"name": "美国 Billboard", "id": 11641012, "icon": "🇺🇸"},
        {"name": "UK 排行榜", "id": 60198, "icon": "🇬🇧"},
        {"name": "Beatport 电子榜", "id": 3812895, "icon": "🎛️"},
    ]

    charts: list[dict] = []

    # 1) 网易云官方榜单
    for chart_def in _NETEASE_CHARTS:
        try:
            raw = get_playlist_tracks(chart_def["id"], limit=per_chart)
            if raw:
                charts.append({
                    "name": chart_def["name"],
                    "icon": chart_def["icon"],
                    "tracks": [{
                        "title": t.title, "artist": t.artist,
                        "source": t.source or "netease",
                        "source_id": t.external_id or "",
                        "cover_url": t.cover_url,
                        "playback_url": t.playback_url,
                    } for t in raw],
                })
        except Exception:
            logger.debug("Trending chart %s failed", chart_def["name"], exc_info=True)

    # 2) Last.fm 全球榜
    if settings.lastfm_api_key:
        try:
            lfm = LastfmClient(settings.lastfm_api_key)
            raw = lfm.get_chart_top_tracks(limit=per_chart)
            if raw:
                charts.append({
                    "name": "Last.fm 全球榜",
                    "icon": "🌐",
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

    info = {"name": request.artist, "image": "", "bio": "", "tags": [], "top_albums": [], "top_tracks": []}

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
            return await asyncio.to_thread(lfm.get_artist_top_albums, request.artist, 6)
        except Exception:
            logger.debug("Last.fm top albums failed", exc_info=True)
            return []

    async def _hot_tracks():
        try:
            raw_tracks = await asyncio.to_thread(agent.search_web_music, f"{request.artist} 热门歌曲", 6)
            return [
                ExternalTrack(
                    external_id=t.external_id or "",
                    title=t.title, artist=t.artist or request.artist,
                    cover_url=t.cover_url, source=t.source,
                    playback_url=t.playback_url, candidate_kind=t.candidate_kind,
                )
                for t in raw_tracks[:6]
            ]
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

    artist_data, albums, ext, netease_img = await asyncio.gather(
        _artist_data(), _albums(), _hot_tracks(), _netease_image()
    )

    if artist_data:
        info["name"] = artist_data.get("name") or request.artist
        info["bio"] = artist_data.get("bio", "")
        info["tags"] = artist_data.get("tags", [])
    # 头像优先网易云（可靠），Last.fm image 字段兜底
    info["image"] = netease_img or (artist_data.get("image", "") if artist_data else "")
    info["top_albums"] = [{"name": a["name"], "image": a["image"]} for a in albums]
    info["top_tracks"] = ext

    return ArtistInfoResponse(**info)


@app.get("/taste/{user_id}")
def taste_profile(user_id: str):
    return agent.get_taste_profile(user_id)


@app.post("/memory/update")
def update_memory(request: MemoryUpdateRequest):
    memory, changed = agent.update_memory(request)
    return {"memory": memory, "updated": changed}


@app.post("/memory/feedback")
def memory_feedback(request: FeedbackRequest):
    try:
        memory = agent.record_feedback(request)
        return {"memory": memory, "updated": True}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/feedback/dislike")
def dislike(request: DislikeRequest):
    memory = agent.record_dislike(request)
    return {"updated": True, "memory": memory}


@app.get("/memory/{user_id}")
def get_memory(user_id: str):
    return agent.memory.get_memory(user_id)


# ---- 排除规则（用户偏好设置） ----

@app.get("/exclusions/{user_id}")
def list_exclusions(user_id: str):
    return {"rules": agent.memory.list_exclusions(user_id)}


@app.post("/exclusions/{user_id}")
def add_exclusion(user_id: str, body: dict[str, str]):
    rule = body.get("rule", "").strip()
    if not rule:
        raise HTTPException(status_code=400, detail="rule is required")
    added = agent.memory.add_exclusion(user_id, rule)
    return {"added": added, "rules": agent.memory.list_exclusions(user_id)}


@app.delete("/exclusions/{user_id}/{rule:path}")
def remove_exclusion(user_id: str, rule: str):
    removed = agent.memory.remove_exclusion(user_id, rule)
    return {"removed": removed, "rules": agent.memory.list_exclusions(user_id)}


@app.get("/library/tracks")
def library_tracks(limit: int = Query(default=100, ge=1, le=500)):
    return {"tracks": [track.model_dump(mode="json") for track in agent.list_resource_tracks(limit)]}


@app.post("/playlist/generate")
def generate_playlist(request: PlaylistRequest):
    playlist = agent.generate_playlist(request.user_id, request.instruction)
    return playlist


@app.post("/journey/generate")
def generate_journey(request: JourneyRequest):
    return agent.generate_music_journey(request.user_id, request.instruction)


@app.post("/playlist/auto/{user_id}")
def auto_playlists(user_id: str):
    playlists = agent.auto_playlists(user_id)
    return {"playlists": playlists}


@app.get("/playlists/{user_id}")
def list_playlists(user_id: str):
    return {"playlists": agent.list_playlists(user_id)}


@app.delete("/playlist/{user_id}/{playlist_id}")
def delete_playlist(user_id: str, playlist_id: str):
    deleted = agent.delete_playlist(user_id, playlist_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Playlist not found")
    return {"deleted": True}
