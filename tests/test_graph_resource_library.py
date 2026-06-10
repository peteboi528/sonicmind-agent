from fastapi.testclient import TestClient

from app.agent import AudioVisualAgent
from app.api.main import app
from app.graph.nodes import build_agent_plan
from app.models import DislikeRequest
from app.storage import JsonStore


def test_structured_plan_for_playlist_and_journey():
    playlist = build_agent_plan("帮我生成50首chill歌单")
    assert playlist.intent == "playlist"
    assert playlist.target_count == 50
    assert playlist.tools_needed == ["web_music_search", "playlist"]

    journey = build_agent_plan("帮我做跑步热身冲刺放松音乐旅程")
    assert journey.intent == "journey"
    assert journey.tools_needed == ["journey"]


def test_graph_chat_writes_resource_library_and_trace(tmp_path):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))

    answer = agent.chat("u1", "推荐三首chill歌")

    assert answer.answer
    assert any("[plan]" in step for step in answer.agent_trace)
    assert any("[web_music_search]" in step for step in answer.agent_trace)
    assert agent.list_resource_tracks(10)


def test_smalltalk_does_not_require_music_candidates(tmp_path):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))

    answer = agent.chat("u1", "hello")

    # chat 意图不应联网搜索、不应返回可追溯候选
    assert "可追溯的音乐候选" not in answer.answer
    assert "可追溯候选" not in answer.answer
    assert not any("[web_music_search]" in step for step in answer.agent_trace)
    assert answer.pending_goal is None


def test_dislike_filters_online_recommendations(tmp_path):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    agent.record_dislike(DislikeRequest(user_id="u1", title="Online Focus One", artist="Demo Artist"))

    rec = agent.recommend_for_query("u1", "推荐chill R&B", top_k=3)

    titles = [item.asset.title for item in rec.tracks]
    assert "Online Focus One" not in titles
    assert agent.memory.get_memory("u1").dislikes


def test_journey_has_three_phases(tmp_path):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))

    journey = agent.generate_music_journey("u1", "跑步热身冲刺放松")

    assert len(journey["phases"]) == 3
    assert all("tracks" in phase for phase in journey["phases"])


def test_stream_endpoint_emits_sse_events():
    client = TestClient(app)

    with client.stream("POST", "/agent/stream", json={"user_id": "stream-user", "message": "推荐三首chill歌"}) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "data:" in body
    assert '"type": "plan"' in body
    assert '"type": "final"' in body


def test_library_and_dislike_api():
    client = TestClient(app)

    chat = client.post("/agent/run", json={"user_id": "api-lib", "message": "推荐三首chill歌"})
    assert chat.status_code == 200

    library = client.get("/library/tracks?limit=5")
    assert library.status_code == 200
    assert library.json()["tracks"]

    track = library.json()["tracks"][0]
    dislike = client.post("/feedback/dislike", json={
        "user_id": "api-lib",
        "title": track["title"],
        "artist": track["artist"],
        "source": track["source"],
        "source_id": track["source_id"],
        "reason": "test",
    })
    assert dislike.status_code == 200
    assert dislike.json()["updated"] is True
