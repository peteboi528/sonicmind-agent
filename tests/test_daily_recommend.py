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
    assert "本地" in results.summary
    assert "外部候选" in results.summary


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


def test_playlist_respects_requested_count_with_external_fill(tmp_path):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))

    playlist = agent.generate_playlist("demo-user", "帮我生成50首的chill放松歌单")

    assert len(playlist.tracks) == 50
    assert any(getattr(track, "source", "") != "local" for track in playlist.tracks)


def test_playlist_drops_unverified_llm_tracks(tmp_path):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))

    class FakePlaylistLLM:
        def generate(self, prompt, system=None, temperature=0.7):
            return '{"name":"x","description":"","tracks":[{"title":"不存在的歌","artist":"不存在的歌手","asset_id":null}]}'

        def chat(self, messages, temperature=0.7):
            return ""

        def chat_with_tools(self, messages, tools, temperature=0.3, tool_choice="auto"):
            from app.llm.protocol import LLMResponse
            return LLMResponse(content="ok", finish_reason="stop")

    agent.llm = FakePlaylistLLM()
    playlist = agent.generate_playlist("demo-user", "帮我生成5首chill放松歌单", target_count=5)

    assert len(playlist.tracks) == 5
    assert all(getattr(track, "source", "") != "llm" for track in playlist.tracks)
