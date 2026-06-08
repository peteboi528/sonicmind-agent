from fastapi.testclient import TestClient

from app.api.main import app


def test_api_flow():
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200

    ingest = client.post("/assets/ingest", json={"url": "https://example.com/api-test"})
    assert ingest.status_code == 200
    asset_id = ingest.json()["asset_id"]

    analyze = client.post(f"/assets/{asset_id}/analyze")
    assert analyze.status_code == 200
    assert len(analyze.json()["segments"]) == 6

    listen = client.post("/listen", json={"user_id": "api-user", "asset_id": asset_id, "duration": 180, "completed": True})
    assert listen.status_code == 200
    assert listen.json()["memory_updated"] is True

    update = client.post("/memory/update", json={"user_id": "api-user", "event": "我喜欢电子音乐和放松的氛围"})
    assert update.status_code == 200
    assert update.json()["updated"] is True

    taste = client.get("/taste/api-user")
    assert taste.status_code == 200

    daily = client.post("/recommend/daily", json={"user_id": "api-user", "time_of_day": "evening"})
    assert daily.status_code == 200
    assert daily.json()["tracks"]

    search = client.post("/search", json={"user_id": "api-user", "query": "电子 放松"})
    assert search.status_code == 200
    assert "summary" in search.json()
    assert search.json()["agent_trace"]

    chat = client.post("/chat", json={"user_id": "api-user", "message": "推荐一些适合工作时听的音乐"})
    assert chat.status_code == 200
    assert chat.json()["answer"]
    assert chat.json()["agent_trace"]

    enrich = client.post(f"/assets/{asset_id}/enrich", json={"use_network": False})
    assert enrich.status_code == 200
    assert enrich.json()["mode"] == "offline"

    assets = client.get("/assets")
    assert assets.status_code == 200
    assert len(assets.json()["assets"]) >= 1

    refreshed = client.post("/assets/ingest", json={"url": "https://example.com/api-test", "force_refresh": True})
    assert refreshed.status_code == 200
    assert refreshed.json()["asset_id"] == asset_id

    forced_analyze = client.post(f"/assets/{asset_id}/analyze?force_refresh=true")
    assert forced_analyze.status_code == 200
    assert len(forced_analyze.json()["segments"]) == 6

    delete_asset = client.delete(f"/assets/{asset_id}")
    assert delete_asset.status_code == 200
    assert delete_asset.json()["deleted"] is True

    clear_cache = client.delete("/cache?preserve_memory=false")
    assert clear_cache.status_code == 200
    assert "assets" in clear_cache.json()["cleared"]
