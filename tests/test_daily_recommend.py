from app.agent import AudioVisualAgent
from app.models import MemoryUpdateRequest
from app.storage import JsonStore


def test_daily_recommend(tmp_path):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))

    for i in range(3):
        asset = agent.ingest_video(f"https://example.com/track-{i}")
        agent.analyze_media(asset.asset_id)
        agent.record_listen("demo-user", asset.asset_id, duration=180, completed=True)

    agent.update_memory(MemoryUpdateRequest(user_id="demo-user", event="我喜欢电子音乐"))

    rec = agent.daily_recommend("demo-user", time_of_day="evening")
    assert rec.user_id == "demo-user"
    assert rec.tracks
    assert rec.reason_summary
    assert rec.agent_trace
    for track in rec.tracks:
        assert track.reason
        assert track.category in ("familiar", "discovery", "mood_match")


def test_search(tmp_path):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))

    asset = agent.ingest_video("https://example.com/jazz-night")
    agent.analyze_media(asset.asset_id)

    results = agent.search("demo-user", "爵士 放松", include_external=True, top_k=10)
    assert results.summary
    assert isinstance(results.external, list)
    assert results.agent_trace


def test_taste_profile(tmp_path):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))

    for i in range(3):
        asset = agent.ingest_video(f"https://example.com/song-{i}")
        agent.analyze_media(asset.asset_id)
        agent.record_listen("demo-user", asset.asset_id, duration=100, completed=True)

    taste = agent.get_taste_profile("demo-user")
    assert taste.top_genres
    assert taste.preferred_energy > 0


def test_playlist_fallback(tmp_path):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))

    for i in range(3):
        asset = agent.ingest_video(f"https://example.com/playlist-{i}")
        agent.analyze_media(asset.asset_id)

    playlist = agent.generate_playlist("demo-user", "帮我生成一个晚上听的歌单")
    assert playlist.tracks
    assert playlist.generated_by in {"llm", "fallback"}
