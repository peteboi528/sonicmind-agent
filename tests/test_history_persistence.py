from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from app.api.main import app, history_service
from app.models import RecommendationHistoryItem
from app.services.history import HistoryService
from app.storage import JsonStore


def test_chat_history_api_saves_and_lists_turn():
    client = TestClient(app)
    user_id = "history-user-api"
    thread_id = "thread-history-api"
    history_service.clear_chat_threads(user_id)

    resp = client.post("/history/chat/turn", json={
        "user_id": user_id,
        "thread_id": thread_id,
        "user_message": "推荐几首深夜 R&B",
        "assistant_message": "给你 3 首。",
        "cards": [{"title": "Nights", "artist": "Frank Ocean"}],
        "trace_summary": {"intent": "recommend", "final_cards": 1},
    })

    assert resp.status_code == 200
    body = resp.json()
    assert body["saved"] is True
    assert body["thread"]["thread_id"] == thread_id
    assert body["thread"]["messages"][0]["role"] == "user"
    assert body["thread"]["messages"][1]["cards"][0]["title"] == "Nights"

    listed = client.get(f"/history/chat/{user_id}")
    assert listed.status_code == 200
    assert listed.json()["threads"][0]["thread_id"] == thread_id


def test_recommendation_history_api_saves_with_expiration():
    client = TestClient(app)
    user_id = "history-user-recs"
    history_service.store.write_models("recommendation_history_full", user_id, [])

    resp = client.post("/history/recommendations", json={
        "user_id": user_id,
        "thread_id": "thread-recs",
        "query": "推荐几首 Future 类似的歌",
        "answer": "可以从这些开始。",
        "cards": [{"title": "Mask Off", "artist": "Future"}],
        "ttl_days": 7,
    })

    assert resp.status_code == 200
    rec = resp.json()["recommendation"]
    assert rec["query"] == "推荐几首 Future 类似的歌"
    assert rec["cards"][0]["artist"] == "Future"
    assert datetime.fromisoformat(rec["expires_at"]) > datetime.fromisoformat(rec["created_at"])

    listed = client.get(f"/history/recommendations/{user_id}")
    assert listed.status_code == 200
    assert [item["record_id"] for item in listed.json()["recommendations"]] == [rec["record_id"]]


def test_recommendation_history_filters_expired_items(tmp_path):
    store = JsonStore(tmp_path / "store")
    service = HistoryService(store)
    now = datetime.now(UTC)
    expired = RecommendationHistoryItem(
        record_id="expired",
        user_id="u",
        query="old",
        cards=[{"title": "Old"}],
        created_at=(now - timedelta(days=10)).isoformat(),
        expires_at=(now - timedelta(days=1)).isoformat(),
    )
    active = RecommendationHistoryItem(
        record_id="active",
        user_id="u",
        query="new",
        cards=[{"title": "New"}],
        created_at=now.isoformat(),
        expires_at=(now + timedelta(days=1)).isoformat(),
    )
    store.write_models("recommendation_history_full", "u", [expired, active])

    items = service.list_recommendations("u")

    assert [item.record_id for item in items] == ["active"]
    persisted = store.read_models("recommendation_history_full", "u", RecommendationHistoryItem)
    assert [item.record_id for item in persisted] == ["active"]

