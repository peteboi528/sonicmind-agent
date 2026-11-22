import json

from fastapi.testclient import TestClient

from app.api import main
from app.intents import is_continuation
from app.memory import MemoryManager, extract_preference
from app.models import ExternalTrack
from app.rules.track import _valid_external_track


def test_verified_track_can_match_itself_but_placeholder_cannot():
    track = ExternalTrack(external_id="1406633327", title="Blinding Lights", artist="The Weeknd", source="netease")
    assert _valid_external_track(track, "Blinding Lights")
    assert not _valid_external_track(track.model_copy(update={"external_id": "fake-song"}), "Blinding Lights")


def test_tag_related_search_keeps_song_without_scene_words_in_title():
    track = ExternalTrack(
        external_id="1", title="Waltz for Debby", artist="Bill Evans", source="netease", genre=["爵士"], mood=["专注"]
    )
    assert _valid_external_track(track, "工作 专注 爵士")
    assert not _valid_external_track(track, "周杰伦")


def test_zero_count_returns_final_without_music_tools(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Zero-count chat must not run the graph")

    monkeypatch.setattr(main.agent.graph, "astream", fail)
    response = TestClient(main.app).post("/agent/stream", json={"message": "能给我 0 首吗？告诉我原因。"})
    response.raise_for_status()
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert [event["type"] for event in events] == ["final"]
    assert events[0]["payload"]["recommended_tracks"] == []


def test_multilingual_continuations_and_temporary_exclusions():
    assert is_continuation("More like that, but avoid repeats. Please reply in English.")
    assert is_continuation("Dame otras canciones, sin repetir las anteriores. Contesta en español.")
    assert is_continuation("もう少し静かな曲にして。さっきの曲は除いて。日本語で答えて。")
    assert MemoryManager._extract_negative_preference("再来，不要重复。") is None
    assert MemoryManager._extract_negative_preference("再来几首，不要刚才那一批。") is None
    assert MemoryManager._extract_negative_preference("我以后都不想听短视频神曲，请记住。") == "短视频神曲"
    text = "我之前说不喜欢 deep house，现在我重新喜欢了 deep house，请记住。"
    assert MemoryManager._detect_preference_correction(text) is None
    assert extract_preference(text) == "deep house"
    elided = "我之前说不喜欢 deep house，现在我重新喜欢了，请记住。"
    assert MemoryManager._detect_preference_correction(elided) is None
    assert extract_preference(elided) == "deep house"
