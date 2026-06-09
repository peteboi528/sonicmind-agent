from __future__ import annotations

import os

import pytest


# Tests must be deterministic even when production behavior is online-first and
# a developer has a real LLM key in .env. Runtime behavior is unchanged outside
# pytest.
os.environ["LLM_API_KEY"] = ""
os.environ.setdefault("LLM_TIMEOUT_SECONDS", "1")
os.environ.setdefault("RESOURCE_LIBRARY_PATH", "data/test_resource_library.sqlite")


@pytest.fixture(autouse=True)
def fake_online_music_search(monkeypatch):
    from app.agent import AudioVisualAgent
    from app.models import ExternalTrack

    def _fake_search_web_music(self, query: str, top_k: int = 5):
        seeds = [
            ExternalTrack(
                external_id="netease-real-1",
                title="Online Focus One",
                artist="Demo Artist",
                genre=["R&B"],
                mood=["放松"],
                source="netease",
                playback_url="https://music.163.com/song?id=1",
            ),
            ExternalTrack(
                external_id="bili-real-1",
                title="Online Chill Two",
                artist="Demo Creator",
                genre=["R&B"],
                mood=["放松"],
                source="bilibili",
                playback_url="https://player.bilibili.com/player.html?bvid=BV1",
            ),
            ExternalTrack(
                external_id="yt-real-1",
                title="Online Work Three",
                artist="Demo Channel",
                genre=["电子"],
                mood=["专注"],
                source="youtube",
                playback_url="https://www.youtube.com/embed/demo",
            ),
        ]
        if len(seeds) < top_k:
            seeds.extend(
                ExternalTrack(
                    external_id=f"mock-fallback-{idx}",
                    title=f"Fallback Track {idx}",
                    artist="Mock Artist",
                    genre=["流行"],
                    mood=["放松"],
                    source="mock-fallback",
                )
                for idx in range(1, top_k - len(seeds) + 1)
            )
        return seeds[:top_k]

    monkeypatch.setattr(AudioVisualAgent, "search_web_music", _fake_search_web_music)
