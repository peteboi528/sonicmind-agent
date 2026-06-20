from app.agent import AudioVisualAgent
from app.models import Asset, ExternalTrack, MemoryUpdateRequest, RankingBreakdown, TasteProfile
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


def test_recommendation_uses_local_library_and_avoids_recent_round(tmp_path, monkeypatch):
    from app.search import lastfm_discovery, netease_playlist, web_music_discovery

    store = JsonStore(tmp_path / "store")
    agent = AudioVisualAgent(store)
    for i in range(4):
        asset = Asset(
            asset_id=f"local-{i}", source_url=f"https://example.com/{i}",
            title=f"Night R&B {i}", artist=f"Artist {i}", duration_seconds=180,
            status="analyzed", genre=["R&B"], mood=["深夜", "放松"],
        )
        store.write_model("assets", asset.asset_id, asset)
    memory = agent.memory.get_memory("u-local")
    memory.taste_profile = TasteProfile(top_genres=[("R&B", 10)], top_moods=[("放松", 8)])
    store.write_model("memory", "u-local", memory)

    monkeypatch.setattr(netease_playlist, "search_and_extract", lambda *a, **k: [])
    monkeypatch.setattr(web_music_discovery, "discover_from_llm", lambda *a, **k: [])
    monkeypatch.setattr(lastfm_discovery, "discover_from_lastfm", lambda *a, **k: [])
    monkeypatch.setattr(agent, "search_web_music", lambda *a, **k: [])

    first = agent.recommend_for_query("u-local", "深夜 R&B 放松", top_k=2)
    second = agent.recommend_for_query("u-local", "深夜 R&B 放松", top_k=2)
    first_ids = {item.asset.asset_id for item in first.tracks}
    second_ids = {item.asset.asset_id for item in second.tracks}
    assert first_ids
    assert second_ids
    assert first_ids.isdisjoint(second_ids)
    assert all(item.asset.source == "local" for item in first.tracks + second.tracks)


def test_rate_limited_recommend_falls_back_to_resource_pool(tmp_path, monkeypatch):
    """网易云限流（所有在线路由空）时，从 SQLite 已验证候选池召回，而非返回空。"""
    from app.models import ExternalTrack
    from app.search import lastfm_discovery, netease_playlist, web_music_discovery

    store = JsonStore(tmp_path / "store")
    agent = AudioVisualAgent(store)

    # 候选池里预先攒了真实已验证的 netease 歌（模拟历史搜索成果）。
    for i in range(6):
        agent.library.upsert_external(ExternalTrack(
            external_id=f"pool-{i}", title=f"Night Song {i}", artist="Real Artist",
            source="netease", genre=["R&B"], mood=["放松", "深夜"],
            playback_url=f"https://music.163.com/song?id=pool-{i}",
        ))

    # 模拟限流：所有在线路由全空。
    monkeypatch.setattr(netease_playlist, "search_and_extract", lambda *a, **k: [])
    monkeypatch.setattr(web_music_discovery, "discover_from_llm", lambda *a, **k: [])
    monkeypatch.setattr(lastfm_discovery, "discover_from_lastfm", lambda *a, **k: [])
    monkeypatch.setattr(agent, "search_web_music", lambda *a, **k: [])

    rec = agent.recommend_for_query("u-pool", "深夜 R&B 放松", top_k=3)

    # 限流前：返回空。限流后（本修复）：从候选池捞到真实歌。
    assert rec.tracks, "候选池兜底应在限流时仍给出真实候选，而非空结果"
    assert any("resource_pool" in line for line in rec.agent_trace)


def test_source_balance_caps_local_when_online_supply_exists(tmp_path):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    ranked = []
    for i in range(10):
        track = Asset(
            asset_id=f"l-{i}", source_url=f"https://example.com/l-{i}",
            title=f"Local {i}", artist="Local Artist", duration_seconds=180,
            status="analyzed", genre=["R&B"], mood=["放松"],
        )
        ranked.append((track, RankingBreakdown(title=track.title, source="local", score=1 - i * 0.01, reason="test")))
    for i in range(10):
        track = ExternalTrack(
            external_id=f"o-{i}", title=f"Online {i}", artist="Online Artist", source="netease",
        )
        ranked.append((track, RankingBreakdown(title=track.title, source="netease", score=0.5 - i * 0.01, reason="test")))

    selected = agent._balance_recommendation_sources(ranked, top_k=10)

    assert sum(isinstance(track, Asset) for track, _ in selected) == 4
    assert sum(isinstance(track, ExternalTrack) for track, _ in selected) == 6
    assert sum(isinstance(track, Asset) for track, _ in selected[:8]) <= 4
    assert any(isinstance(track, ExternalTrack) for track, _ in selected[:3])


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
