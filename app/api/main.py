from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse

from app.agent import AudioVisualAgent
from app.config import settings
from app.models import (
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
        rated = next((r for r in memory.ratings if r.asset_id == request.asset_id), None)
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


@app.post("/agent/stream")
def agent_stream(request: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in request.history]

    def events():
        for event in agent.stream_chat(request.user_id, request.message, history=history or None):
            yield f"data: {json.dumps(event.model_dump(mode='json'), ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


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
