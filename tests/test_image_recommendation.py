"""End-to-end screenshot turns with catalog metadata from real NetEase releases."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from app.agent import AudioVisualAgent
from app.api import main
from app.config import settings
from app.models import Asset, ExternalTrack, Playlist, SavedAlbum
from app.services.cover_recognizer import CoverRecognition
from app.storage import JsonStore
from app.tools.registry import get_tool_spec

TRACKS = (
    ("431610014", "Starboy", "The Weeknd、Daft Punk"),
    ("461544117", "Starboy", "Daft Punk、The Weeknd"),
    ("1406633327", "Blinding Lights", "The Weeknd"),
    ("442867526", "Die For You", "The Weeknd"),
    ("26349642", "Get Lucky", "Daft Punk、Pharrell Williams、Nile Rodgers"),
    ("29558299", "Get Lucky", "Daft Punk、Nile Rodgers、Pharrell Williams"),
    ("17455854", "Harder, Better, Faster, Stronger", "Daft Punk"),
    ("26349639", "Instant Crush", "Daft Punk、Julian Casablancas"),
    ("2046700580", "Giorgio by Moroder", "Daft Punk"),
)


def _netease(track):
    song_id, title, artist = track
    return {"song_id": song_id, "title": title, "artist": artist}


def _local(track):
    song_id, title, artist = track
    return Asset(
        asset_id=f"local-{song_id}",
        title=title,
        artist=artist,
        duration_seconds=240,
        source_url=f"https://music.163.com/song?id={song_id}",
    )


def _events(response):
    response.raise_for_status()
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def _image(client, prompt, *, thread="screenshot", user="alice"):
    return _events(
        client.post(
            "/agent/stream/image",
            files={"file": ("screenshot.png", b"\x89PNG\r\n\x1a\n", "image/png")},
            data={"message": prompt, "thread_id": thread, "user_id": user},
        )
    )


def _message(client, text, *, thread="screenshot", user="alice"):
    return _events(client.post("/agent/stream", json={"message": text, "thread_id": thread, "user_id": user}))


@pytest.fixture
def image_app(tmp_path, monkeypatch):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    inspections = []
    searches = []
    local = [_local(TRACKS[i]) for i in (0, 2, 3)]
    monkeypatch.setattr(main, "agent", agent)
    monkeypatch.setattr(settings, "agent_image_enabled", True)
    monkeypatch.setattr(settings, "auth_enabled", False)
    monkeypatch.setattr(settings, "rate_limit_enabled", False)
    monkeypatch.setattr(agent, "list_assets", lambda: local)

    async def read_image(*args, **kwargs):
        inspections.append(args[1])
        return CoverRecognition(
            method="vision",
            image_kind="player",
            track="Starboy",
            artist="The Weeknd",
            collaborators=["Daft Punk"],
            confidence=0.99,
        )

    async def catalog(query, **kwargs):
        searches.append(query)
        return [_netease(track) for track in TRACKS if query.casefold() in (track[1] + " " + track[2]).casefold()]

    monkeypatch.setattr("app.graph.builder.recognize_music_image", read_image)
    monkeypatch.setattr("app.sources.netease.asearch_netease_many", catalog)
    return TestClient(main.app), agent, local, inspections, searches


@pytest.mark.parametrize(
    ("prompt", "correction", "lang"),
    [
        ("推荐三首这个歌手的歌，别和截图重复", "我指的是合作歌手，收藏过的也去掉", "zh"),
        (
            "Recommend three songs by this artist; skip the screenshot song.",
            "I meant the featured artist. Exclude my saved tracks.",
            "en",
        ),
        ("このアーティストの曲を3曲おすすめ。画像の曲は除いて", "コラボの方。お気に入りは除外して", "ja"),
        (
            "推荐这位歌手 three tracks，别和截图重复。Reply in English",
            "我指的是 featured artist，收藏过的也去掉。Reply in English",
            "en",
        ),
    ],
)
def test_image_recommend_corrects_artist_and_excludes_saved(image_app, monkeypatch, prompt, correction, lang):
    client, agent, _, inspections, calls = image_app
    saved_track = ExternalTrack(external_id=TRACKS[4][0], title=TRACKS[4][1], artist=TRACKS[4][2], source="netease")
    monkeypatch.setattr(
        agent,
        "list_playlists",
        lambda user: [
            Playlist(
                playlist_id="my-music",
                user_id=user,
                name="Saved",
                tracks=[saved_track],
            )
        ],
    )
    monkeypatch.setattr(
        agent,
        "list_saved_albums",
        lambda user: [
            SavedAlbum(
                album_id="26349640",
                name="Random Access Memories",
                artist="Daft Punk",
                user_id=user,
                tracks=[
                    ExternalTrack(external_id=TRACKS[7][0], title=TRACKS[7][1], artist=TRACKS[7][2], source="netease")
                ],
            )
        ],
    )
    first = _image(client, prompt)
    assert first[-1]["type"] == "final"
    assert {card["title"] for card in first[-1]["payload"]["cards"]} == {"Die For You", "Blinding Lights"}
    assert first[-1]["payload"]["trace_summary"]["response_language"] == lang
    corrected = _message(client, correction)
    payload = corrected[-1]["payload"]
    assert payload["trace_summary"]["response_language"] == lang
    assert payload["trace_summary"]["image_reused"]
    assert payload["trace_summary"]["image_vision_calls"] == 0
    assert len(inspections) == 1
    assert "Daft Punk" in calls
    assert {card["title"] for card in payload["cards"]} == {"Harder, Better, Faster, Stronger", "Giorgio by Moroder"}
    assert {track["title"] for track in payload["recommended_tracks"]} == {card["title"] for card in payload["cards"]}



@pytest.mark.parametrize("ordered_credit_line", [True, False])
def test_ocr_starboy_credit_order_with_an_empty_library(image_app, monkeypatch, ordered_credit_line):
    client, _, local, _, _ = image_app
    local.clear()
    credit_text = "The Weeknd, Daft Punk" if ordered_credit_line else "The Weeknd\nDaft Punk"

    async def read_player(*args, **kwargs):
        return CoverRecognition(method="ocr", raw_text=f"NOW PLAYING\nStarboy\n{credit_text}")

    async def catalog(query, **kwargs):
        if query.startswith("Starboy"):
            return [
                _netease(("1804967437", "THE WEEKEND DAFT PUNK - STARBOY(DEMO DROP)BPM95", "DJ Mr Steven、The Weeknd、Daft Punk")),
                *(_netease(TRACKS[i]) for i in (1, 0)),
            ]
        remixes = (
            ("437856703", "Starboy (DVBBS & IZII Remix)", "The Weeknd、Daft Punk、DVBBS、IZII"),
            ("435592093", "Starboy (Vicetone Remix)", "Vicetone、The Weeknd、Daft Punk"),
        )
        return [_netease(track) for track in (*remixes, *TRACKS) if query in track[2]]

    monkeypatch.setattr("app.graph.builder.recognize_music_image", read_player)
    monkeypatch.setattr("app.sources.netease.asearch_netease_many", catalog)
    response = _image(client, "推荐两首这个歌手的歌，别和截图重复")[-1]["payload"]
    if ordered_credit_line:
        assert {card["title"] for card in response["cards"]} == {"Blinding Lights", "Die For You"}
        corrected = _message(client, "我指的是合作歌手")[-1]["payload"]
        assert corrected["trace_summary"]["image_reused"]
        assert len(corrected["cards"]) == 2
        assert all(
            "Daft Punk" in card["artist"] and not card["title"].startswith("Starboy") for card in corrected["cards"]
        )
    else:
        assert response["cards"] == []
        assert set(response["trace_summary"]["image_choices"]) == {"The Weeknd", "Daft Punk"}


def test_filter_only_reuses_candidates_and_thread_isolation(image_app, monkeypatch):
    client, agent, _, inspections, calls = image_app
    monkeypatch.setattr(agent, "list_playlists", lambda user: [])
    monkeypatch.setattr(agent, "list_saved_albums", lambda user: [])
    _image(client, "推荐两首这个歌手的歌")
    _message(client, "我指的是合作歌手")
    calls_before = len(calls)
    for i in range(20):
        patch = "收藏过的去掉" if i % 2 else "收藏过的也可以"
        assert _message(client, patch)[-1]["payload"]["trace_summary"]["image_reused"]
    assert len(calls) == calls_before
    _message(client, "换个话题，打个招呼", thread="other")
    assert agent.memory.get_image_dialogue("alice", "other") is None
    assert agent.memory.get_image_dialogue("bob", "screenshot") is None
    assert agent.memory.get_image_dialogue("alice", "screenshot").plan.target == "Daft Punk"
    assert len(inspections) == 1
    _message(client, "谢谢")
    assert agent.memory.get_image_dialogue("alice", "screenshot").plan.target == "Daft Punk"
    _message(client, "换个话题，打个招呼")
    assert agent.memory.get_image_dialogue("alice", "screenshot") is None


def test_artist_only_and_new_image(image_app, monkeypatch):
    client, agent, _, _, _ = image_app
    _image(client, "推荐三首这个歌手的歌")

    async def artist_only(*args, **kwargs):
        return CoverRecognition(artist="Daft Punk", method="vision", confidence=0.95)

    monkeypatch.setattr("app.graph.builder.recognize_music_image", artist_only)
    payload = _image(client, "推荐两首这个歌手的歌")[-1]["payload"]
    assert len(payload["cards"]) == 2
    state = agent.memory.get_image_dialogue("alice", "screenshot")
    assert state.plan.target == "Daft Punk"
    assert state.screenshot_tracks == []


def test_ambiguous_collaborators_remain_pending_until_selected(image_app, monkeypatch):
    client, agent, local, _, _ = image_app
    local.append(_local(TRACKS[4]))

    async def multiple(*args, **kwargs):
        return CoverRecognition(
            track="Get Lucky",
            artist="Daft Punk",
            collaborators=["Pharrell Williams", "Nile Rodgers"],
            method="vision",
            image_kind="player",
            confidence=0.42,
        )

    monkeypatch.setattr("app.graph.builder.recognize_music_image", multiple)
    _image(client, "推荐三首这个歌手的歌")
    payload = _message(client, "我指的是合作歌手")[-1]["payload"]
    assert set(payload["trace_summary"]["image_choices"]) == {"Pharrell Williams", "Nile Rodgers"}
    assert _message(client, "去掉收藏过的")[-1]["payload"]["trace_summary"]["image_status"] == "needs_artist"
    _message(client, "artist: Nile Rodgers")
    assert agent.memory.get_image_dialogue("alice", "screenshot").plan.target == "Nile Rodgers"


def test_progress_before_timeout_and_no_fabrication(image_app, monkeypatch):
    client, _, _, _, _ = image_app

    async def slow(*args, **kwargs):
        await asyncio.sleep(1)
        return []

    monkeypatch.setattr("app.sources.netease.asearch_netease_many", slow)
    monkeypatch.setattr(get_tool_spec("image_catalog_lookup"), "timeout_seconds", 0.25)
    events = _image(client, "推荐三首这个歌手的歌")
    first = next(i for i, event in enumerate(events) if event["type"] == "candidates")
    timeout = next(i for i, event in enumerate(events) if event["payload"].get("metrics", {}).get("timeout_phase"))
    assert first < timeout
    payload = events[-1]["payload"]
    assert len(payload["cards"]) == 2
    assert payload["trace_summary"]["latency_budget"]["timed_out_tools"] == ["image_catalog_lookup"]
    assert payload["trace_summary"]["tool_candidate_count"] == 3
    assert payload["trace_summary"]["final_cards"] == 2


def test_offline_source_cancelled_when_local_has_enough(image_app, monkeypatch):
    _, agent, _, _, _ = image_app
    cancelled = []

    async def slow(*args, **kwargs):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)

    monkeypatch.setattr("app.sources.netease.asearch_netease_many", slow)

    async def run():
        started = time.monotonic()
        events = [
            e
            async for e in agent.stream_chat_async(
                "alice", "推荐一首这个歌手的歌", thread_id="cancel", image=(b"any", "image/png")
            )
        ]
        return events[-1].payload, time.monotonic() - started

    payload, elapsed = asyncio.run(run())
    assert elapsed < 2
    assert len(payload["cards"]) == 1
    assert cancelled
    assert any(o["status"] == "cancelled" for o in payload["trace_summary"]["tool_outcomes"])


def test_image_switch_and_mime_validation(image_app, monkeypatch):
    client, _, _, inspections, _ = image_app
    assert client.post("/agent/stream/image", files={"file": ("bad.png", b"invalid", "image/png")}).status_code == 415
    monkeypatch.setattr(settings, "agent_image_enabled", False)
    assert (
        client.post("/agent/stream/image", files={"file": ("valid.png", b"\x89PNG\r\n\x1a\n", "image/png")}).status_code
        == 404
    )
    assert not client.get("/health").json()["details"]["image_turn_enabled"]
    assert _message(client, "你好")[-1]["type"] == "final"
    assert inspections == []


def test_artist_only_asks_for_title_if_user_requires_screenshot_exclusion(image_app, monkeypatch):
    client, _, _, inspections, _ = image_app

    async def artist_only(*args, **kwargs):
        inspections.append(args[1])
        return CoverRecognition(artist="The Weeknd", method="vision", confidence=0.92)

    monkeypatch.setattr("app.graph.builder.recognize_music_image", artist_only)
    payload = _image(client, "推荐两首这个歌手的歌，别和截图重复")[-1]["payload"]
    assert payload["trace_summary"]["image_status"] == "needs_title"
    assert payload["cards"] == []
    payload = _message(client, "截图曲名：Starboy")[-1]["payload"]
    assert {card["title"] for card in payload["cards"]} == {"Blinding Lights", "Die For You"}
    assert len(inspections) == 1


def test_source_429_uses_verified_library(image_app, monkeypatch):
    import httpx

    client, _, _, _, _ = image_app

    async def rate_limited(*args, **kwargs):
        request = httpx.Request("GET", "https://music.163.com/api/search/get")
        raise httpx.HTTPStatusError("429 rate limited", request=request, response=httpx.Response(429, request=request))

    monkeypatch.setattr("app.sources.netease.asearch_netease_many", rate_limited)
    payload = _image(client, "推荐三首这个歌手的歌")[-1]["payload"]
    assert {card["source"] for card in payload["cards"]} == {"local"}
    assert payload["fallback_reason"] == "source_degraded"
    assert any(
        o.get("error", {}).get("kind") == "HTTPStatusError"
        for o in payload["trace_summary"]["tool_outcomes"]
        if o.get("error")
    )


def test_client_stops_after_first_card_closes_online_lookup(image_app, monkeypatch):
    _, agent, _, _, _ = image_app

    async def run():
        cancelled = asyncio.Event()

        async def slow(*args, **kwargs):
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.set()

        monkeypatch.setattr("app.sources.netease.asearch_netease_many", slow)
        stream = agent.stream_chat_async(
            "alice", "推荐三首这个歌手的歌", thread_id="disconnect", image=(b"sample", "image/png")
        )
        async for event in stream:
            if event.type == "candidates":
                break
        await stream.aclose()
        await asyncio.wait_for(cancelled.wait(), timeout=1.0)

    asyncio.run(run())


def test_artist_preview_emitted_before_slow_knowledge_sources(image_app, monkeypatch):
    from app.models import AssetStatus
    from app.tools.contracts import ToolResult, ToolStatus

    _, agent, local, _, _ = image_app
    for asset in local:
        asset.status = AssetStatus.ANALYZED

    async def slow(args, context):
        await asyncio.sleep(0.08)
        return ToolResult(tool="web_knowledge_search", status=ToolStatus.EMPTY)

    monkeypatch.setattr(get_tool_spec("web_knowledge_search"), "async_handler", slow)

    async def run():
        timeline = []
        async for event in agent.stream_chat_async(
            "alice", "详细介绍 The Weeknd 的音乐生涯和代表作", thread_id="artist"
        ):
            timeline.append(event)
        return timeline

    timeline = asyncio.run(run())
    glimpse = next(i for i, e in enumerate(timeline) if e.type == "library_preview")
    web = next(
        i for i, e in enumerate(timeline) if e.type == "tool_result" and e.payload["tool"] == "web_knowledge_search"
    )
    assert glimpse < web
    assert len(timeline[glimpse].payload["matches"]) == 3
    assert timeline[-1].type == "final"
