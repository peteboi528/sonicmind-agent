"""Shared deterministic fakes for offline tests and eval runs.

The pytest suite and the eval runner both need the same contract: no real LLM,
no real music/network APIs, no opportunistic embedding model downloads.  Keeping
the stubs here avoids the eval path drifting away from the trusted pytest path.
"""
from __future__ import annotations

import os
import random
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch


def configure_offline_env() -> None:
    """Set process env before importing app.config/settings."""
    os.environ["LLM_API_KEY"] = ""
    os.environ.setdefault("LLM_TIMEOUT_SECONDS", "1")
    os.environ.setdefault("RESOURCE_LIBRARY_PATH", "data/test_resource_library.sqlite")
    os.environ["EXTERNAL_SOURCE"] = "mock"
    os.environ["ENABLE_EMBEDDINGS"] = "false"
    os.environ["ENABLE_MUSICBRAINZ"] = "false"
    os.environ["ENABLE_SPOTIFY"] = "false"
    os.environ["ENABLE_DISCOGS"] = "false"


def seed_random() -> None:
    random.seed(20260626)


def fake_search_web_music(
    self: Any,
    query: str,
    top_k: int = 5,
    relevance_query: str = "",
    offset: int = 0,
    **_: Any,
):
    from app.models import ExternalTrack

    if offset:
        return [
            ExternalTrack(
                external_id=f"netease-paged-{offset}-{idx}",
                title=f"Paged Track {offset}-{idx}",
                artist="Demo Artist",
                genre=["R&B"],
                mood=["放松"],
                source="netease",
                playback_url=f"https://music.163.com/song?id={offset}0{idx}",
            )
            for idx in range(1, top_k + 1)
        ]
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


async def fake_search_web_music_async(
    self: Any,
    query: str,
    top_k: int = 5,
    relevance_query: str = "",
    include_video_sources: bool = False,
    offset: int = 0,
    variants: Any = None,
):
    return self.search_web_music(
        query,
        top_k=top_k,
        relevance_query=relevance_query,
        include_video_sources=include_video_sources,
        offset=offset,
        variants=variants,
    )


async def fake_search_videos_async(self: Any, query: str, top_k: int = 5):
    return self.search_videos(query, top_k=top_k)


async def fake_search_artist_info_async(self: Any, query: str):
    return self.search_artist_info(query)


async def fake_recommend_artist_albums_async(self: Any, user_id: str, artist: str, limit: int = 12):
    return self.recommend_artist_albums(user_id, artist, limit)


def fake_playlist_extract(query: str, max_playlists: int = 3, tracks_per_playlist: int = 15):
    from app.models import ExternalTrack
    from app.sources.mock_source import MockSource

    limit = max_playlists * tracks_per_playlist
    source = MockSource()
    tracks = source.search(query, limit=limit) or source.get_recommendations([], [], limit=limit)
    return [track.model_copy(update={"source": "local"}) for track in tracks[:limit]]


def apply_pytest_monkeypatch(monkeypatch: Any) -> None:
    """Install offline fakes using pytest's monkeypatch fixture."""
    from app.agent import AudioVisualAgent

    monkeypatch.setattr(AudioVisualAgent, "search_web_music", fake_search_web_music)
    monkeypatch.setattr(AudioVisualAgent, "search_web_music_async", fake_search_web_music_async)
    monkeypatch.setattr(AudioVisualAgent, "search_videos_async", fake_search_videos_async)
    monkeypatch.setattr(AudioVisualAgent, "search_artist_info_async", fake_search_artist_info_async)
    monkeypatch.setattr(AudioVisualAgent, "recommend_artist_albums_async", fake_recommend_artist_albums_async)
    monkeypatch.setattr("app.knowledge.web_search_source.search_web_info", lambda *a, **k: [])


def start_offline_patches() -> ExitStack:
    """Install offline fakes outside pytest; caller must close the returned stack."""
    from app.agent import AudioVisualAgent
    from app.retrieval import embeddings

    stack = ExitStack()
    stack.enter_context(patch.object(AudioVisualAgent, "search_web_music", fake_search_web_music))
    stack.enter_context(patch.object(AudioVisualAgent, "search_web_music_async", fake_search_web_music_async))
    stack.enter_context(patch.object(AudioVisualAgent, "search_videos_async", fake_search_videos_async))
    stack.enter_context(patch.object(AudioVisualAgent, "search_artist_info_async", fake_search_artist_info_async))
    stack.enter_context(patch.object(AudioVisualAgent, "recommend_artist_albums_async", fake_recommend_artist_albums_async))
    stack.enter_context(patch("app.search.netease_playlist.search_and_extract", fake_playlist_extract))
    stack.enter_context(patch("app.search.netease_playlist.search_netease_playlists", return_value=[]))
    stack.enter_context(patch("app.search.netease_playlist.get_playlist_tracks", return_value=[]))
    stack.enter_context(patch("app.search.netease_playlist.get_playlist_detail", return_value=None))
    stack.enter_context(patch("app.knowledge.web_search_source.search_web_info", lambda *a, **k: []))
    stack.enter_context(patch.object(embeddings, "embeddings_available", lambda: False))
    stack.enter_context(patch.object(embeddings, "encode", lambda *_a, **_k: None))
    stack.enter_context(patch.object(embeddings, "semantic_scores", lambda *_a, **_k: None))
    for target in (
        "app.recommend.scene_vibe.embeddings_available",
        "app.recommend.hygiene.embeddings_available",
    ):
        try:
            stack.enter_context(patch(target, lambda: False))
        except Exception:
            pass
    for target in (
        "app.recommend.scene_vibe.encode",
        "app.recommend.hygiene.semantic_scores",
    ):
        try:
            stack.enter_context(patch(target, lambda *_a, **_k: None))
        except Exception:
            pass
    embeddings._reset_for_test()
    return stack
