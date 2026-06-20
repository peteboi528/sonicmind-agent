import asyncio

from fastapi.testclient import TestClient

from app.agent import AudioVisualAgent
from app.api.main import app
from app.graph.nodes import build_agent_plan
from app.models import AgentPlan, DislikeRequest, ExternalTrack
from app.storage import JsonStore


def test_structured_plan_for_playlist_and_journey():
    playlist = build_agent_plan("帮我生成50首chill歌单")
    assert playlist.intent == "playlist"
    assert playlist.target_count == 50
    assert playlist.tools_needed == ["web_music_search", "playlist"]

    journey = build_agent_plan("帮我做跑步热身冲刺放松音乐旅程")
    assert journey.intent == "journey"
    assert journey.tools_needed == ["journey"]


def test_graph_chat_returns_traceable_candidates_without_requiring_library_mutation(tmp_path):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))

    answer = asyncio.run(agent.chat_async("u1", "推荐三首chill歌"))

    assert answer.answer
    assert any("[plan]" in step for step in answer.agent_trace)
    assert any("[web_music_search]" in step for step in answer.agent_trace)
    assert answer.recommended_tracks


def test_smalltalk_does_not_require_music_candidates(tmp_path):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))

    answer = asyncio.run(agent.chat_async("u1", "hello"))

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


def test_journey_has_dynamic_day_arc_and_rotates_tracks(tmp_path, monkeypatch):
    from app.search import netease_playlist

    def fake_extract(query, max_playlists=3, tracks_per_playlist=12):
        from app.models import ExternalTrack

        prefix = query.split()[0]
        return [
            ExternalTrack(
                external_id=f"{prefix}-{i}", title=f"{prefix} Track {i}",
                artist="Verified Artist", source="netease",
            )
            for i in range(1, 6)
        ]

    monkeypatch.setattr(netease_playlist, "search_and_extract", fake_extract)
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))

    journey = agent.generate_music_journey("u1", "做一个清晨到深夜的音乐旅程")

    assert len(journey["phases"]) == 5
    assert [phase["name"] for phase in journey["phases"]] == ["清晨", "上午", "午后", "傍晚", "深夜"]
    assert all(len(phase["tracks"]) == 4 for phase in journey["phases"])
    first_ids = {t["external_id"] for p in journey["phases"] for t in p["tracks"]}
    assert len(first_ids) == 20
    assert [phase["energy"] for phase in journey["phases"]] == [0.28, 0.46, 0.64, 0.76, 0.32]

    second = agent.generate_music_journey("u1", "做一个清晨到深夜的音乐旅程")
    second_ids = {t["external_id"] for p in second["phases"] for t in p["tracks"]}
    assert first_ids.isdisjoint(second_ids)


def test_journey_stream_emits_track_cards(monkeypatch):
    from app.api import main as main_module

    def fake_journey(user_id, instruction):
        return {
            "user_id": user_id,
            "instruction": instruction,
            "phases": [
                {
                    "name": name,
                    "goal": goal,
                    "transition": transition,
                    "tracks": [{
                        "external_id": f"journey-{index}",
                        "title": f"Journey Track {index}",
                        "artist": "Journey Artist",
                        "source": "netease",
                    }],
                }
                for index, (name, goal, transition) in enumerate([
                    ("清晨", "温和唤醒", "开始一天"),
                    ("午后", "保持活力", "增加律动"),
                    ("深夜", "放松收束", "安静落幕"),
                ], start=1)
            ],
        }

    monkeypatch.setattr(main_module.agent, "generate_music_journey", fake_journey)
    client = TestClient(app)
    with client.stream("POST", "/agent/stream", json={
        "user_id": "journey-stream-user",
        "message": "做一个清晨到深夜的音乐旅程",
    }) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert '"type": "candidates"' in body
    assert '"journey_phase": "清晨"' in body
    assert '"final_cards": 3' in body
    assert '"tools": ["journey"]' in body


def test_journey_final_cards_keep_all_phases_without_explicit_count():
    from app.graph.nodes import _select_listed_tracks

    phases = []
    for phase_index, name in enumerate(["清晨", "上午", "午后", "傍晚", "深夜"]):
        phases.append({
            "name": name,
            "tracks": [
                ExternalTrack(
                    external_id=f"{phase_index}-{track_index}",
                    title=f"Track {phase_index}-{track_index}",
                    artist="Artist",
                    source="netease",
                ).model_dump(mode="json")
                for track_index in range(4)
            ],
        })
    results = [{"type": "journey", "journey": {"phases": phases}}]

    tracks = _select_listed_tracks(results, AgentPlan(intent="journey"))

    assert len(tracks) == 20


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
