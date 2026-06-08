from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse

from app.agent import AudioVisualAgent
from app.config import settings
from app.models import (
    ChatRequest,
    DailyRequest,
    EnrichRequest,
    FeedbackRequest,
    IngestRequest,
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
agent = AudioVisualAgent()


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


@app.get("/memory/{user_id}")
def get_memory(user_id: str):
    return agent.memory.get_memory(user_id)


@app.post("/playlist/generate")
def generate_playlist(request: PlaylistRequest):
    playlist = agent.generate_playlist(request.user_id, request.instruction)
    return playlist


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
