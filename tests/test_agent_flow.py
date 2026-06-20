import asyncio

from app.agent import AudioVisualAgent, CineSonicAgent
from app.media.pipeline import normalize_url, stable_id, title_from_url
from app.models import MemoryUpdateRequest
from app.storage import JsonStore


def test_backward_compat_alias():
    assert CineSonicAgent is AudioVisualAgent


def test_full_flow(tmp_path):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))

    asset = agent.ingest_video("https://example.com/sony-cinematic-demo")
    asset, segments = agent.analyze_media(asset.asset_id)

    assert len(segments) == 6
    assert segments[0].timestamp == "00:00-00:30"
    assert asset.genre
    assert asset.mood
    # 诚实契约：DemoAnalyzer 不做真实音频分析，tempo/energy 在未知时保持 None
    # （下游 score_track 用 or 默认值兜底）。过去这里被随机伪造填充，已移除。
    assert asset.tempo_bpm is None
    assert asset.energy_level is None

    evidences = agent.retrieve_evidence(asset.asset_id, "cinematic trailer climax", top_k=3)
    assert evidences
    assert evidences[0].segment_id

    memory, changed = agent.update_memory(
        MemoryUpdateRequest(
            user_id="demo-user",
            event="我喜欢电子音乐和激昂的节奏。",
            asset_id=asset.asset_id,
        )
    )
    assert changed is True
    assert memory.preferences

    agent.record_listen("demo-user", asset.asset_id, duration=120, completed=True, context="evening")
    memory = agent.memory.get_memory("demo-user")
    assert len(memory.listening_history) == 1

    taste = agent.get_taste_profile("demo-user")
    assert taste.top_genres

    answer = asyncio.run(agent.chat_async("demo-user", "推荐一些适合晚上听的音乐"))
    assert answer.answer
    assert answer.agent_trace
    assert any("recommend" in step for step in answer.agent_trace)


def test_ingest_is_offline_first(tmp_path, monkeypatch):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))

    def fail_fetch(url: str):
        raise AssertionError("network fetch should not run during default ingest")

    monkeypatch.setattr(agent, "_fetch_video_title", fail_fetch)
    asset = agent.ingest_video("https://example.com/offline-first")
    assert asset.title


def test_delete_asset_removes_user_memory_references(tmp_path):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    user_id = "delete-user"

    asset = agent.ingest_video("https://example.com/delete-me")
    agent.analyze_media(asset.asset_id)
    agent.rate_asset(user_id, asset.asset_id, 8.5)
    agent.record_listen(user_id, asset.asset_id, duration=180, completed=True)

    assert agent.delete_asset(asset.asset_id, user_id=user_id) is True

    memory = agent.memory.get_memory(user_id)
    assert not any(r.asset_id == asset.asset_id for r in memory.ratings)
    assert not any(ev.asset_id == asset.asset_id for ev in memory.listening_history)
    assert not agent.media.get_segments(asset.asset_id)


def test_generic_music_recommendation_does_not_auto_bind_recent_asset(tmp_path):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    asset = agent.ingest_video("https://example.com/recent-asset")
    agent.analyze_media(asset.asset_id)
    agent.record_listen("demo-user", asset.asset_id, duration=180, completed=True)

    assert agent._resolve_asset_context("demo-user", "推荐一些 chill 放松歌曲") is None
    assert agent._resolve_asset_context("demo-user", "推荐当前素材里适合做预告片的片段") == asset.asset_id


def test_netease_url_normalization_strips_share_token():
    shared_url = "https://music.163.com/song?id=2700274699&uct2=U2FsdGVkX19mdtaezH9kDfy33am/mSBDHG4wr6X+Ur8="
    canonical = "https://music.163.com/song?id=2700274699"
    hash_url = "https://music.163.com/#/song?id=2700274699&userid=123"

    assert normalize_url(shared_url) == canonical
    assert normalize_url(hash_url) == canonical
    assert stable_id(normalize_url(shared_url)) == stable_id(canonical)
    assert title_from_url(normalize_url(shared_url)) == "网易云歌曲 2700274699"


def test_netease_enrich_uses_title_artist_hint_when_api_falls_back(tmp_path, monkeypatch):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    url = "https://music.163.com/song?id=2700274699&uct2=U2FsdGVkX19mdtaezH9kDfy33am/mSBDHG4wr6X+Ur8="
    asset = agent.ingest_video(url, force_refresh=True)

    monkeypatch.setattr(agent, "_enrich_from_netease", lambda asset, song_id: False)
    monkeypatch.setattr(agent, "_fetch_video_title", lambda url: "浪人的… - 张震岳")

    response = agent.enrich_asset(asset.asset_id, use_network=True)

    assert response.asset.title == "浪人的…"
    assert response.asset.artist == "张震岳"


def test_placeholder_metadata_is_not_marked_found(tmp_path):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    asset = agent.ingest_video("https://music.163.com/song?id=12345", force_refresh=True)

    metadata = agent.fetch_track_metadata(asset_id=asset.asset_id, use_network=False)

    assert metadata["found"] is False
    assert metadata["title"].startswith("网易云歌曲")
