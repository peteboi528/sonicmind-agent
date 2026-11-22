import pytest
from fastapi.testclient import TestClient

from app.agent import AudioVisualAgent
from app.api import main
from app.models import ExternalTrack
from app.storage import JsonStore


@pytest.fixture
def curation(tmp_path, monkeypatch):
    agent = AudioVisualAgent(JsonStore(tmp_path))
    tracks = [
        ExternalTrack(external_id=str(i), title=t, artist="A", source="netease")
        for i, t in enumerate(["Blue", "Gold", "Green"])
    ]
    monkeypatch.setattr(agent, "_playlist_candidates", lambda *args, **kwargs: tracks)
    monkeypatch.setattr(main, "agent", agent)
    return TestClient(main.app), agent


@pytest.mark.parametrize("reply, mode", [('{"name":"通勤","tracks":[]}', "llm"), ("{}", "fallback")])
def test_select_and_reopen(curation, monkeypatch, reply, mode):
    client, agent = curation
    monkeypatch.setattr(agent.llm, "generate", lambda *args: reply)
    request = {"user_id": "listener", "instruction": "通勤三首"}
    draft = client.post("/playlist/preview", json=request).json()
    assert draft["generated_by"] == mode
    assert len(draft["tracks"]) == 3
    assert client.get("/playlists/listener").json()["playlists"] == []
    selection = {"user_id": "listener", "draft_id": draft["playlist_id"], "indices": [2, 0], "name": "回家路上"}
    for indices in [[], [-1], [3]]:
        assert client.post("/playlist/confirm", json={**selection, "indices": indices}).status_code == 422
    assert client.post("/playlist/confirm", json={**selection, "user_id": "other"}).status_code == 409
    saved = client.post("/playlist/confirm", json=selection).json()
    assert saved["name"] == "回家路上"
    assert [t["external_id"] for t in saved["tracks"]] == ["0", "2"]
    assert client.post("/playlist/confirm", json=selection).json() == saved
    assert client.post("/playlist/confirm", json={**selection, "indices": [1]}).status_code == 409
    assert client.get("/playlists/listener").json()["playlists"] == [saved]
    client.post("/playlist/preview", json=request).raise_for_status()
    assert client.post("/playlist/confirm", json=selection).status_code == 409
    assert client.get("/playlists/listener").json()["playlists"] == [saved]


@pytest.mark.parametrize("reply", ['{"name":"Focus","tracks":[{"title":"Amber","artist":"A"}]}', "{}"])
def test_revisions_keep_tracks_and_exclude_previous_rounds(curation, monkeypatch, reply):
    client, agent = curation
    monkeypatch.setattr(agent.llm, "generate", lambda *args: "{}")
    user = {"user_id": "listener"}
    initial = client.post("/playlist/preview", json={**user, "instruction": "通勤 3 首"}).json()
    assert initial["target_count"] == 3

    new_tracks = [
        ExternalTrack(external_id="other-id", title=" GOLD ", artist="a", source="lastfm"),
        ExternalTrack(external_id="0", title="Another name", artist="A", source="netease"),
        ExternalTrack(external_id="new1", title="Amber", artist="A", source="netease"),
    ]
    calls = []

    def build(instruction, library, seeds, count, *, eligible, recall_hint):
        calls.append((instruction, recall_hint))
        assert count == 2
        assert not eligible(new_tracks[0])
        assert not eligible(new_tracks[1])
        return new_tracks  # Defense in depth if a candidate source returns seen tracks.

    monkeypatch.setattr(agent, "_playlist_candidates", build)
    monkeypatch.setattr(agent.llm, "generate", lambda *args: reply)
    first = {**user, "draft_id": initial["playlist_id"], "kept_indices": [1], "adjustment": "少一点人声"}
    assert client.post("/playlist/revise", json={**first, "kept_indices": [3]}).status_code == 422
    assert client.post("/playlist/revise", json={**first, "user_id": "other"}).status_code == 409

    revised = client.post("/playlist/revise", json=first)
    assert revised.status_code == 200
    result = revised.json()
    assert result["added_count"] == 1
    assert [t["title"] for t in result["draft"]["tracks"]] == ["Gold", "Amber"]
    assert result["draft"]["tracks"][0] == initial["tracks"][1]
    assert result["draft"]["target_count"] == 3
    assert "seen_track_keys" not in result["draft"]
    assert calls[0][1] == "少一点人声"
    assert "通勤 3 首" in calls[0][0]
    assert "少一点人声" in calls[0][0]
    assert client.post("/playlist/revise", json=first).status_code == 409

    second = {**user, "draft_id": result["draft"]["playlist_id"], "kept_indices": [0], "adjustment": "偏爵士"}
    exhausted = client.post("/playlist/revise", json=second).json()
    assert exhausted["added_count"] == 0
    assert exhausted["draft"] == result["draft"]
    assert calls[-1][1] == "偏爵士"
    assert "少一点人声" in calls[-1][0]
    assert client.get("/playlists/listener").json()["playlists"] == []

    # The previous unsaved draft is still confirmable when the candidate pool is exhausted.
    saved = client.post("/playlist/confirm", json={**second, "indices": [0, 1]}).json()
    assert [t["title"] for t in saved["tracks"]] == ["Gold", "Amber"]
    assert client.post("/playlist/revise", json=second).status_code == 409


def test_revision_cannot_overwrite_draft_confirmed_during_recall(curation, monkeypatch):
    client, agent = curation
    initial = client.post("/playlist/preview", json={"user_id": "listener", "instruction": "通勤 3 首"}).json()
    draft_id = initial["playlist_id"]

    def recall(*args, **kwargs):
        agent.confirm_playlist_draft("listener", draft_id, [0], "原草稿")
        return [ExternalTrack(external_id="new1", title="Amber", artist="A", source="netease")]

    monkeypatch.setattr(agent, "_playlist_candidates", recall)
    assert client.post("/playlist/revise", json={"user_id": "listener", "draft_id": draft_id}).status_code == 409
    assert client.get("/playlists/listener").json()["playlists"][0]["name"] == "原草稿"


def test_revision_queries_next_page_when_first_page_is_seen(curation, monkeypatch):
    client, agent = curation
    monkeypatch.setattr(agent.llm, "generate", lambda *args: "{}")
    initial = client.post("/playlist/preview", json={"user_id": "listener", "instruction": "通勤 3 首"}).json()
    seen = [ExternalTrack.model_validate(track) for track in initial["tracks"]]
    service = agent._playlist_service()
    monkeypatch.setattr(agent, "_playlist_candidates", AudioVisualAgent._playlist_candidates.__get__(agent))
    monkeypatch.setattr("app.search.netease_playlist.search_and_extract", lambda *args, **kwargs: seen)
    calls = []

    def search(query, *, top_k, relevance_query, offset=0):
        calls.append((query, offset))
        if offset:
            return [ExternalTrack(external_id="new1", title="Amber", artist="A", source="netease")]
        return seen

    monkeypatch.setattr(service, "_search_web_music", search)
    payload = {
        "user_id": "listener",
        "draft_id": initial["playlist_id"],
        "kept_indices": [0, 1],
        "adjustment": "偏爵士",
    }
    result = client.post("/playlist/revise", json=payload).json()
    assert calls[0] == ("偏爵士", 0)
    assert calls[-1] == ("偏爵士", 25)
    assert [t["title"] for t in result["draft"]["tracks"]] == ["Blue", "Gold", "Amber"]


@pytest.mark.parametrize(
    "reply",
    [
        "{}",
        '{"name":"通勤","tracks":[{"title":"Dreams","artist":"Fleetwood Mac"}]}',
        '{"name":"通勤","tracks":[]}',
    ],
)
def test_revision_excludes_artist_in_selection_fill_and_fallback(curation, monkeypatch, reply):
    client, agent = curation
    monkeypatch.setattr(agent.llm, "generate", lambda *args: "{}")
    user = {"user_id": "listener"}
    draft = client.post("/playlist/preview", json={**user, "instruction": "通勤 2 首"}).json()
    forbidden = ExternalTrack(external_id="2588700", title="Dreams", artist="Fleetwood Mac", source="netease")
    allowed = ExternalTrack(external_id="426194894", title="White Ferrari", artist="Frank Ocean", source="netease")
    monkeypatch.setattr(agent, "_playlist_candidates", lambda *args, **kwargs: [forbidden, allowed])
    monkeypatch.setattr(agent.llm, "generate", lambda *args: reply)
    response = client.post(
        "/playlist/revise",
        json={**user, "draft_id": draft["playlist_id"], "kept_indices": [0], "adjustment": "不要 Fleetwood Mac"},
    )
    response.raise_for_status()
    assert [track["title"] for track in response.json()["draft"]["tracks"]] == ["Blue", "White Ferrari"]


@pytest.mark.parametrize(
    ("instruction", "target"),
    [
        ("推荐 90 年代的 8 首通勤歌", 8),
        ("推荐 Blink-182 的 5 首跑步歌", 5),
        ("给我五首深夜器乐", 5),
        ("集中できる曲を５曲お願いします", 5),
        ("Recomiéndame siete canciones para leer", 7),
        ("Suggest 6 instrumental tracks for writing", 6),
        ("通勤 0 首", 0),
    ],
)
def test_preview_uses_track_quantity(curation, instruction, target):
    client, _ = curation
    result = client.post("/playlist/preview", json={"instruction": instruction})
    result.raise_for_status()
    assert result.json()["target_count"] == target
    if target == 0:
        assert result.json()["tracks"] == []
