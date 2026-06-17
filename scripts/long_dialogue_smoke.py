from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
RUNTIME_DIR = ARTIFACT_DIR / "long_dialogue_smoke_runtime"
REPORT_MD = ARTIFACT_DIR / "long_dialogue_smoke_report.md"
REPORT_JSON = ARTIFACT_DIR / "long_dialogue_smoke_report.json"
DEPS_DIR = RUNTIME_DIR / "deps"


def configure_environment() -> None:
    """Keep the smoke run deterministic and isolated from the developer store."""
    os.environ["LLM_API_KEY"] = ""
    os.environ["EXTERNAL_SOURCE"] = "mock"
    os.environ["ENABLE_EMBEDDINGS"] = "false"
    os.environ["ENABLE_ONLINE_ENRICH"] = "false"
    os.environ["LASTFM_API_KEY"] = ""
    os.environ["TAVILY_API_KEY"] = ""
    os.environ["LLM_TIMEOUT_SECONDS"] = "1"
    os.environ["STORE_ROOT"] = str(RUNTIME_DIR / "store")
    os.environ["MEDIA_ROOT"] = str(RUNTIME_DIR / "media")
    os.environ["RESOURCE_LIBRARY_PATH"] = str(RUNTIME_DIR / "resource_library.sqlite")


def add_local_deps() -> None:
    """Use dependencies installed with pip --target into the smoke runtime."""
    if DEPS_DIR.exists() and str(DEPS_DIR) not in sys.path:
        sys.path.insert(0, str(DEPS_DIR))


def reset_runtime_dir() -> None:
    """Clear smoke data while preserving optional pip --target dependencies."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    for child in RUNTIME_DIR.iterdir():
        if child.name == "deps":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class CaseResult:
    name: str
    area: str
    ok: bool
    checks: list[CheckResult] = field(default_factory=list)
    error: str = ""
    fix_hint: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DialogueCase:
    name: str
    message: str
    expect: Callable[[Any], list[CheckResult]]
    fix_hint: str


def ok(name: str, condition: bool, detail: str = "") -> CheckResult:
    return CheckResult(name=name, ok=bool(condition), detail=detail)


def trace(answer: Any) -> str:
    return "\n".join(getattr(answer, "agent_trace", []) or [])


def answer_text(answer: Any) -> str:
    return getattr(answer, "answer", "") or ""


def track_count(answer: Any) -> int:
    return len(getattr(answer, "recommended_tracks", []) or [])


def synthetic_tracks():
    from app.models import ExternalTrack

    return [
        ExternalTrack(
            external_id="netease-focus-1",
            title="Midnight Compiler",
            artist="Demo Artist",
            genre=["电子"],
            mood=["专注", "放松"],
            source="netease",
            playback_url="https://music.163.com/song?id=10001",
            cover_url="https://example.com/covers/focus.jpg",
        ),
        ExternalTrack(
            external_id="netease-run-2",
            title="City Run Pulse",
            artist="Demo Runner",
            genre=["流行", "电子"],
            mood=["激昂"],
            source="netease",
            playback_url="https://music.163.com/song?id=10002",
            cover_url="https://example.com/covers/run.jpg",
        ),
        ExternalTrack(
            external_id="bili-live-3",
            title="NewJeans Live Stage",
            artist="Demo Live",
            genre=["流行"],
            mood=["欢快"],
            source="bilibili",
            playback_url="https://player.bilibili.com/player.html?bvid=BVSMOKE01",
        ),
        ExternalTrack(
            external_id="yt-video-4",
            title="Taylor Swift Official Video",
            artist="Demo Channel",
            genre=["流行"],
            mood=["明亮"],
            source="youtube",
            playback_url="https://www.youtube.com/embed/smoke001",
        ),
        ExternalTrack(
            external_id="netease-chill-5",
            title="Soft Neon Rain",
            artist="Demo Chill",
            genre=["R&B"],
            mood=["放松"],
            source="netease",
            playback_url="https://music.163.com/song?id=10005",
        ),
        ExternalTrack(
            external_id="netease-night-6",
            title="Quiet Terminal",
            artist="Demo Night",
            genre=["电子"],
            mood=["专注"],
            source="netease",
            playback_url="https://music.163.com/song?id=10006",
        ),
    ]


def install_fakes() -> None:
    from app.agent import AudioVisualAgent
    from app.models import Asset, ExternalTrack
    from app.sources import netease as netease_module

    base_tracks = synthetic_tracks()

    def fake_search_web_music(
        self: AudioVisualAgent,
        query: str,
        top_k: int = 5,
        relevance_query: str = "",
        offset: int = 0,
        **_: Any,
    ) -> list[ExternalTrack]:
        tracks = []
        for idx in range(max(top_k + offset, top_k)):
            seed = base_tracks[idx % len(base_tracks)]
            clone = seed.model_copy(deep=True)
            if offset:
                clone.external_id = f"{clone.external_id}-page-{offset}-{idx}"
                clone.title = f"{clone.title} #{offset + idx}"
            tracks.append(clone)
        return tracks[offset:offset + top_k] if offset else tracks[:top_k]

    def fake_search_videos(self: AudioVisualAgent, query: str, top_k: int = 5) -> list[ExternalTrack]:
        return [
            ExternalTrack(
                external_id="BVSMOKE01",
                title=f"{query} - Bilibili Live",
                artist="Smoke Video",
                source="bilibili",
                playback_url="https://player.bilibili.com/player.html?bvid=BVSMOKE01",
                candidate_kind="official_mv",
            ),
            ExternalTrack(
                external_id="YTSMOKE01",
                title=f"{query} - Official MV",
                artist="Smoke Video",
                source="youtube",
                playback_url="https://www.youtube.com/embed/YTSMOKE01",
                candidate_kind="official_mv",
            ),
        ][:top_k]

    def fake_search_artist_info(self: AudioVisualAgent, query: str) -> list[dict[str, str]]:
        return [
            {
                "title": f"{query} biography",
                "content": f"{query} 是一个用于 smoke 测试的歌手资料摘要，包含出道、风格和代表作品。",
                "url": "https://example.com/artist-info",
            }
        ]

    def fake_import_playlist(
        self: AudioVisualAgent,
        playlist_ref: str,
        cookie: str = "",
        user_id: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        tracks = []
        for idx, item in enumerate(base_tracks[: min(5, limit)], start=1):
            asset = self.ingest_video(item.playback_url or f"https://music.163.com/song?id={idx}", force_refresh=True)
            asset.title = item.title
            asset.artist = item.artist
            asset.genre = item.genre
            asset.mood = item.mood
            asset.source_url = item.playback_url or asset.source_url
            asset.status = "analyzed"
            self.store.write_model("assets", asset.asset_id, asset)
            tracks.append(item.model_dump(mode="json"))
        return {"name": "Smoke Imported Playlist", "imported": len(tracks), "skipped": 0, "total": len(tracks), "tracks": tracks}

    def fake_get_audio_url(self: AudioVisualAgent, track: Asset | ExternalTrack | Any, netease_cookie: str = "") -> str | None:
        source_url = getattr(track, "source_url", "") or getattr(track, "playback_url", "")
        external_id = getattr(track, "external_id", "") or getattr(track, "source_id", "")
        if "music.163.com" in source_url or "netease" in (getattr(track, "source", "") or ""):
            return f"https://music.163.com/song/media/outer/url?id={external_id or '10001'}.mp3"
        return None

    def fake_get_mv_url(self: AudioVisualAgent, track: Any) -> str | None:
        source = (getattr(track, "source", "") or "").lower()
        external_id = getattr(track, "external_id", "") or "SMOKE"
        if source == "bilibili":
            return f"https://player.bilibili.com/player.html?bvid={external_id}"
        if source == "youtube":
            return f"https://www.youtube.com/embed/{external_id}"
        return "https://www.youtube.com/embed/SMOKE"

    def fake_artist_albums(artist: str, limit: int = 6) -> list[dict[str, Any]]:
        return [
            {"id": "alb-001", "name": "Smoke Era", "artist": artist, "image": "https://example.com/alb1.jpg", "track_count": 3},
            {"id": "alb-002", "name": "Regression Nights", "artist": artist, "image": "https://example.com/alb2.jpg", "track_count": 4},
        ][:limit]

    def fake_album_search(artist: str, album: str) -> dict[str, Any]:
        return {"id": "alb-001", "name": album, "artist": artist, "cover": "https://example.com/alb1.jpg", "track_count": 3}

    def fake_album_tracks(album_id: str, limit: int = 100) -> dict[str, Any]:
        tracks = [
            {"song_id": "10001", "title": "Opening Smoke", "artist": "Smoke Artist", "album": "Smoke Era", "cover": "https://example.com/alb1.jpg"},
            {"song_id": "10002", "title": "Middle Assertion", "artist": "Smoke Artist", "album": "Smoke Era", "cover": "https://example.com/alb1.jpg"},
            {"song_id": "10003", "title": "Final Report", "artist": "Smoke Artist", "album": "Smoke Era", "cover": "https://example.com/alb1.jpg"},
        ][:limit]
        return {"id": album_id, "name": "Smoke Era", "artist": "Smoke Artist", "cover": "https://example.com/alb1.jpg", "track_count": len(tracks), "tracks": tracks}

    AudioVisualAgent.search_web_music = fake_search_web_music
    AudioVisualAgent.search_videos = fake_search_videos
    AudioVisualAgent.search_artist_info = fake_search_artist_info
    AudioVisualAgent.import_netease_playlist = fake_import_playlist
    AudioVisualAgent.get_audio_url = fake_get_audio_url
    AudioVisualAgent.get_mv_url = fake_get_mv_url

    netease_module.search_netease_artist_albums = fake_artist_albums
    netease_module.search_netease_album = fake_album_search
    netease_module.fetch_netease_album_tracks = fake_album_tracks
    netease_module.search_netease_artist_image = lambda artist: "https://example.com/artist.jpg"


def dialogue_cases() -> list[DialogueCase]:
    return [
        DialogueCase(
            name="smalltalk_chat",
            message="你好，先简单介绍一下你能做什么",
            expect=lambda a: [
                ok("answer_non_empty", bool(answer_text(a))),
                ok("trace_has_plan", "[plan]" in trace(a)),
            ],
            fix_hint="检查 chat/smalltalk intent 是否仍被 AgentPlan 接受，以及 compose_answer 是否返回非空。",
        ),
        DialogueCase(
            name="recommend_scene",
            message="推荐几首适合深夜写代码的歌",
            expect=lambda a: [
                ok("answer_non_empty", bool(answer_text(a))),
                ok("has_track_cards", track_count(a) > 0, f"tracks={track_count(a)}"),
                ok("trace_recommend", "[recommend]" in trace(a) or "[web_music_search]" in trace(a)),
            ],
            fix_hint="检查 recommend intent、web_music_search 预取和 recommend_for_query 候选对齐。",
        ),
        DialogueCase(
            name="continue_more",
            message="再来几首，不要上一批",
            expect=lambda a: [
                ok("answer_non_empty", bool(answer_text(a))),
                ok("has_new_tracks", track_count(a) > 0, f"tracks={track_count(a)}"),
                ok("trace_has_offset_search", "获取" in trace(a)),
            ],
            fix_hint="检查多轮 history 继承、excluded_tracks 去重和 offset 翻页逻辑。",
        ),
        DialogueCase(
            name="search_artist",
            message="找 The Weeknd 的歌",
            expect=lambda a: [
                ok("answer_non_empty", bool(answer_text(a))),
                ok("trace_search", "[search]" in trace(a) or "[web_music_search]" in trace(a)),
            ],
            fix_hint="检查 search intent 路由、search_core 抽取和搜索候选输出。",
        ),
        DialogueCase(
            name="playlist_generation",
            message="帮我做 8 首适合跑步冲刺的歌单",
            expect=lambda a: [
                ok("answer_non_empty", bool(answer_text(a))),
                ok("trace_playlist", "[playlist]" in trace(a)),
                ok("has_playlist_tracks", track_count(a) > 0, f"tracks={track_count(a)}"),
            ],
            fix_hint="检查 playlist intent、target_count 解析和 generate_playlist 种子候选。",
        ),
        DialogueCase(
            name="artist_albums",
            message="推荐 Taylor Swift 的专辑",
            expect=lambda a: [
                ok("answer_non_empty", bool(answer_text(a))),
                ok("trace_artist_albums", "[artist_albums]" in trace(a)),
                ok("answer_keeps_album_titles", "专辑《" in answer_text(a), answer_text(a)[:120]),
            ],
            fix_hint="检查 artist_albums intent、album_card 事件和 Answer Guard 专辑白名单。",
        ),
        DialogueCase(
            name="artist_info",
            message="介绍 NewJeans 的背景",
            expect=lambda a: [
                ok("answer_non_empty", bool(answer_text(a))),
                ok("trace_web_info", "[web_info_search]" in trace(a)),
            ],
            fix_hint="检查 artist_info intent 是否被 discuss 抢走，以及 web_info_search 回答合成。",
        ),
        DialogueCase(
            name="video_search",
            message="找 NewJeans 的 MV 视频",
            expect=lambda a: [
                ok("answer_non_empty", bool(answer_text(a))),
                ok("trace_video", "[video_search]" in trace(a)),
                ok("video_tracks", any(t.source in {"bilibili", "youtube"} for t in (a.recommended_tracks or []))),
            ],
            fix_hint="检查 video intent 优先级、search_videos 和 MV 卡片 source 字段。",
        ),
        DialogueCase(
            name="taste_summary",
            message="分析我的音乐品味",
            expect=lambda a: [
                ok("answer_non_empty", bool(answer_text(a))),
                ok("trace_taste", "[taste]" in trace(a)),
            ],
            fix_hint="检查 taste intent、memory 读取和 summarize_taste 输出。",
        ),
        DialogueCase(
            name="taste_experiment",
            message="推荐点不一样的，做个品味实验",
            expect=lambda a: [
                ok("answer_non_empty", bool(answer_text(a))),
                ok("trace_taste_experiment", "[taste_experiment]" in trace(a)),
                ok("mentions_three_buckets", all(word in answer_text(a) for word in ["安全区", "轻微越界", "大胆探索"])),
            ],
            fix_hint="检查 taste_experiment intent、三档候选生成和 Answer Guard 实验曲目白名单。",
        ),
        DialogueCase(
            name="journey",
            message="做一个清晨到深夜的音乐旅程，热身到冲刺再放松",
            expect=lambda a: [
                ok("answer_non_empty", bool(answer_text(a))),
                ok("trace_journey", "[journey]" in trace(a)),
            ],
            fix_hint="检查 journey intent 优先级和 generate_music_journey 阶段生成。",
        ),
        DialogueCase(
            name="discuss_artist",
            message="Asen 牛逼吗，风格是什么水平",
            expect=lambda a: [
                ok("answer_non_empty", bool(answer_text(a))),
                ok("trace_discuss_evidence", "[web_music_search]" in trace(a) or "[search]" in trace(a)),
            ],
            fix_hint="检查 discuss intent 及其联网证据候选，不要输出无来源泛谈。",
        ),
        DialogueCase(
            name="import_playlist",
            message="帮我导入这个网易云歌单 playlist?id=123456",
            expect=lambda a: [
                ok("answer_non_empty", bool(answer_text(a))),
                ok("trace_import", "[import]" in trace(a)),
            ],
            fix_hint="检查 import intent 优先级、网易云歌单链接识别和导入结果合成。",
        ),
    ]


def run_dialogue(agent: Any, user_id: str) -> list[CaseResult]:
    history: list[dict[str, str]] = []
    results: list[CaseResult] = []

    for case in dialogue_cases():
        try:
            answer = agent.chat(user_id, case.message, history=history)
            checks = case.expect(answer)
            passed = all(c.ok for c in checks)
            results.append(
                CaseResult(
                    name=case.name,
                    area="dialogue",
                    ok=passed,
                    checks=checks,
                    fix_hint=case.fix_hint,
                    meta={
                        "message": case.message,
                        "answer_preview": answer_text(answer)[:240],
                        "track_count": track_count(answer),
                        "trace": getattr(answer, "agent_trace", []) or [],
                    },
                )
            )
            history.append({"role": "user", "content": case.message})
            history.append({"role": "assistant", "content": answer_text(answer)})
        except Exception as exc:
            results.append(
                CaseResult(
                    name=case.name,
                    area="dialogue",
                    ok=False,
                    error=repr(exc),
                    fix_hint=case.fix_hint,
                    meta={"message": case.message},
                )
            )
    return results


def api_check(name: str, func: Callable[[], tuple[bool, str, dict[str, Any]]], fix_hint: str) -> CaseResult:
    try:
        passed, detail, meta = func()
        return CaseResult(
            name=name,
            area="api",
            ok=passed,
            checks=[ok("api_contract", passed, detail)],
            fix_hint=fix_hint,
            meta=meta,
        )
    except Exception as exc:
        return CaseResult(name=name, area="api", ok=False, error=repr(exc), fix_hint=fix_hint)


def run_api_checks(app: Any, user_id: str) -> list[CaseResult]:
    from fastapi.testclient import TestClient

    client = TestClient(app)
    state: dict[str, Any] = {}

    def health():
        resp = client.get("/health")
        return resp.status_code == 200 and resp.json().get("status") in {"ok", "degraded"}, str(resp.json()), {}

    def ingest_analyze_listen_rate_delete():
        ingest = client.post("/assets/ingest_full", json={"url": "https://music.163.com/song?id=10001", "force_refresh": True})
        if ingest.status_code != 200:
            return False, f"ingest status={ingest.status_code}", {}
        asset = ingest.json()
        asset_id = asset["asset_id"]
        state["asset_id"] = asset_id
        listen = client.post("/listen", json={"user_id": user_id, "asset_id": asset_id, "duration": 180, "completed": True})
        rate = client.post("/rate", json={"user_id": user_id, "asset_id": asset_id, "score": 8})
        ratings = client.get(f"/ratings/{user_id}")
        return (
            listen.status_code == 200 and rate.status_code == 200 and ratings.status_code == 200 and ratings.json().get("ratings"),
            f"asset={asset_id} listen={listen.status_code} rate={rate.status_code}",
            {"asset_id": asset_id, "ratings": ratings.json().get("ratings", [])[:2]},
        )

    def daily_and_search():
        daily = client.post("/recommend/daily", json={"user_id": user_id, "time_of_day": "evening"})
        search = client.post("/search", json={"user_id": user_id, "query": "深夜 专注", "top_k": 5})
        return (
            daily.status_code == 200 and bool(daily.json().get("tracks")) and search.status_code == 200 and "summary" in search.json(),
            f"daily={daily.status_code} search={search.status_code}",
            {"daily_tracks": len(daily.json().get("tracks", [])), "search_external": len(search.json().get("external", []))},
        )

    def stream_album_events():
        with client.stream("POST", "/agent/stream", json={"user_id": user_id, "message": "推荐 Taylor Swift 的专辑"}) as resp:
            body = resp.read().decode("utf-8")
        event_types = []
        for chunk in body.split("\n\n"):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    event_types.append(json.loads(line[6:]).get("type"))
        return (
            "album_card" in event_types and "final" in event_types,
            f"events={event_types}",
            {"event_types": event_types},
        )

    def stream_taste_experiment_events():
        with client.stream("POST", "/agent/stream", json={"user_id": user_id, "message": "推荐点不一样的，做个品味实验"}) as resp:
            body = resp.read().decode("utf-8")
        event_types = []
        final_payload: dict[str, Any] = {}
        for chunk in body.split("\n\n"):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    event_types.append(event.get("type"))
                    if event.get("type") == "final":
                        final_payload = event.get("payload") or {}
        experiment = final_payload.get("taste_experiment") or {}
        segment_names = [s.get("name") for s in experiment.get("segments", [])]
        trace_summary = final_payload.get("trace_summary") or {}
        return (
            "candidates" in event_types
            and "final" in event_types
            and segment_names == ["safe", "stretch", "bold"]
            and trace_summary.get("intent") == "taste_experiment",
            f"events={event_types} segments={segment_names} trace={trace_summary}",
            {"event_types": event_types, "segment_names": segment_names, "trace_summary": trace_summary},
        )

    def artist_and_album_apis():
        info = client.post("/artist/info", json={"artist": "Taylor Swift"})
        tracks = client.post("/artist/album_tracks", json={"artist": "Taylor Swift", "album": "Smoke Era", "album_id": "alb-001"})
        return (
            info.status_code == 200
            and bool(info.json().get("top_albums"))
            and tracks.status_code == 200
            and len(tracks.json().get("tracks", [])) == 3,
            f"artist={info.status_code} album_tracks={tracks.status_code}",
            {"albums": info.json().get("top_albums", [])[:2], "track_titles": [t["title"] for t in tracks.json().get("tracks", [])]},
        )

    def save_album_cycle():
        payload = {
            "user_id": user_id,
            "album_id": "alb-001",
            "name": "Smoke Era",
            "artist": "Smoke Artist",
            "track_count": 3,
            "tracks": [
                {"external_id": "10001", "title": "Opening Smoke", "artist": "Smoke Artist", "source": "netease", "playback_url": "https://music.163.com/song?id=10001"}
            ],
        }
        save = client.post("/album/save", json=payload)
        status = client.get(f"/album/saved/{user_id}/alb-001")
        listing = client.get(f"/albums/saved/{user_id}")
        delete = client.delete(f"/album/saved/{user_id}/alb-001")
        return (
            save.status_code == 200 and status.json().get("saved") is True and listing.json().get("albums") and delete.json().get("deleted") is True,
            f"save={save.status_code} saved={status.json()} delete={delete.json()}",
            {"listed": listing.json().get("albums", [])[:1]},
        )

    def playback_apis():
        audio = client.post("/api/playback/audio", json={"user_id": user_id, "track": {"title": "Opening Smoke", "artist": "Smoke Artist", "source": "netease", "source_id": "10001", "playback_url": "https://music.163.com/song?id=10001"}})
        mv = client.post("/api/playback/mv", json={"user_id": user_id, "track": {"title": "Live", "artist": "Smoke", "source": "bilibili", "source_id": "BVSMOKE01"}})
        return (
            audio.status_code == 200 and audio.json().get("url") and mv.status_code == 200 and mv.json().get("url"),
            f"audio={audio.json()} mv={mv.json()}",
            {"audio": audio.json(), "mv": mv.json()},
        )

    def exclusions_and_dislike():
        add = client.post(f"/exclusions/{user_id}", json={"rule": "过度商业化"})
        listing = client.get(f"/exclusions/{user_id}")
        dislike = client.post("/feedback/dislike", json={"user_id": user_id, "title": "Commercial Song", "artist": "Demo", "source": "netease", "source_id": "bad-1"})
        remove = client.delete(f"/exclusions/{user_id}/过度商业化")
        return (
            add.status_code == 200
            and "过度商业化" in listing.json().get("rules", [])
            and dislike.status_code == 200
            and remove.status_code == 200,
            f"add={add.status_code} dislike={dislike.status_code} remove={remove.status_code}",
            {"rules_after_add": listing.json().get("rules", [])},
        )

    def playlist_cycle():
        generated = client.post("/playlist/generate", json={"user_id": user_id, "instruction": "做 5 首 smoke 歌单"})
        listing = client.get(f"/playlists/{user_id}")
        playlists = listing.json().get("playlists", [])
        playlist_id = playlists[-1]["playlist_id"] if playlists else generated.json().get("playlist_id")
        deleted = client.delete(f"/playlist/{user_id}/{playlist_id}") if playlist_id else None
        return (
            generated.status_code == 200 and playlists and deleted is not None and deleted.status_code == 200,
            f"generated={generated.status_code} listed={len(playlists)} deleted={getattr(deleted, 'status_code', None)}",
            {"playlist_id": playlist_id, "track_count": len(generated.json().get("tracks", []))},
        )

    def cleanup_asset():
        asset_id = state.get("asset_id")
        if not asset_id:
            return True, "no asset to clean", {}
        deleted = client.delete(f"/assets/{asset_id}?user_id={user_id}")
        return deleted.status_code in {200, 404}, f"delete={deleted.status_code}", {"asset_id": asset_id}

    checks = [
        ("health", health, "检查 /health 依赖探针和 settings 初始化。"),
        ("ingest_listen_rate", ingest_analyze_listen_rate_delete, "检查入库、分析、听歌记录、评分和 TasteProfile 写入。"),
        ("daily_and_search", daily_and_search, "检查每日推荐和 search API 候选输出。"),
        ("stream_album_events", stream_album_events, "检查 SSE 是否发出 album_card/final，前端专辑卡依赖该事件。"),
        ("stream_taste_experiment_events", stream_taste_experiment_events, "检查 SSE final payload 是否带 taste_experiment 三档实验对象。"),
        ("artist_album_apis", artist_and_album_apis, "检查歌手页专辑 id、专辑详情原始曲序和 album_tracks 契约。"),
        ("save_album_cycle", save_album_cycle, "检查收藏专辑保存、查询、列表和删除。"),
        ("playback_apis", playback_apis, "检查音频播放代理和 MV 播放代理。"),
        ("exclusions_and_dislike", exclusions_and_dislike, "检查排除规则和不喜欢反馈写入。"),
        ("playlist_cycle", playlist_cycle, "检查歌单生成、列表和删除。"),
        ("cleanup_asset", cleanup_asset, "检查测试资产删除。"),
    ]
    return [api_check(name, func, hint) for name, func, hint in checks]


def serialize_result(result: CaseResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "area": result.area,
        "ok": result.ok,
        "error": result.error,
        "fix_hint": result.fix_hint,
        "checks": [check.__dict__ for check in result.checks],
        "meta": result.meta,
    }


def render_report(results: list[CaseResult]) -> str:
    passed = sum(1 for item in results if item.ok)
    failed = [item for item in results if not item.ok]
    by_area: dict[str, list[CaseResult]] = {}
    for item in results:
        by_area.setdefault(item.area, []).append(item)

    lines = [
        "# Long Dialogue Smoke Report",
        "",
        f"- Total: {len(results)}",
        f"- Passed: {passed}",
        f"- Failed: {len(failed)}",
        "",
        "## Coverage",
        "",
    ]
    for area, items in by_area.items():
        lines.append(f"- {area}: {sum(1 for i in items if i.ok)}/{len(items)} passed")
    lines.extend(["", "## Results", ""])
    for item in results:
        mark = "PASS" if item.ok else "FAIL"
        lines.append(f"### {mark} {item.area}/{item.name}")
        if item.error:
            lines.append(f"- Error: `{item.error}`")
        for check in item.checks:
            cmark = "PASS" if check.ok else "FAIL"
            detail = f" - {check.detail}" if check.detail else ""
            lines.append(f"- {cmark} `{check.name}`{detail}")
        if item.meta.get("answer_preview"):
            lines.append(f"- Answer preview: {item.meta['answer_preview']}")
        if item.meta.get("event_types"):
            lines.append(f"- Stream events: {item.meta['event_types']}")
        lines.append("")

    lines.extend(["## Modification Plan", ""])
    if not failed:
        lines.append("No blocking failures found in this deterministic smoke run.")
        lines.append("")
        lines.append("Suggested next hardening:")
        lines.append("- Run the same script once with real online sources enabled before release.")
        lines.append("- Add the dialogue section as a CI smoke job after dev dependencies are installed.")
    else:
        journey_poisoned = [
            item for item in failed
            if item.area == "dialogue"
            and any("mock：journey" in line for line in (item.meta.get("trace") or []))
            and "journey" not in item.name
        ]
        if journey_poisoned:
            lines.append("Detected likely root cause: mock query planning is scanning the whole prompt/history,")
            lines.append("so earlier `journey` words poison later turns and many unrelated requests route to `journey`.")
            lines.append("")
            lines.append("Root-cause fix plan:")
            lines.append("1. In `app/llm/mock.py`, make `_query_plan` extract only the current `【本轮输入】` or latest `用户：...` query before keyword routing.")
            lines.append("2. Route mock planning through `match_intent_by_keywords(current_query)` first, then fill `use_web/use_vector/tools` from `INTENT_REGISTRY` so new intents (`artist_albums`, `video`, `artist_info`) stay aligned.")
            lines.append("3. Add a regression where history contains `音乐旅程/热身/冲刺` but current input is `分析我的音乐品味` and must still route to `taste`.")
            lines.append("4. Re-run this script and require `stream_album_events` to include `album_card`.")
            lines.append("")
        lines.append("Prioritize these fixes:")
        for idx, item in enumerate(failed, start=1):
            lines.append(f"{idx}. `{item.area}/{item.name}`: {item.fix_hint}")
            if item.error:
                lines.append(f"   - Runtime error: `{item.error}`")
            bad_checks = [c for c in item.checks if not c.ok]
            if bad_checks:
                lines.append(f"   - Failed checks: {', '.join(c.name for c in bad_checks)}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    configure_environment()
    add_local_deps()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    reset_runtime_dir()

    try:
        install_fakes()
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        result = CaseResult(
            name=f"missing_dependency_{missing}",
            area="environment",
            ok=False,
            error=repr(exc),
            fix_hint=(
                "安装运行依赖到脚本隔离目录后重跑："
                f"`{sys.executable} -m pip install --target {DEPS_DIR} 'python-dotenv>=1.0.0' "
                "'fastapi>=0.115.0' 'httpx>=0.27.0' 'langgraph>=0.2.0' 'qrcode>=7.0'`"
            ),
        )
        REPORT_JSON.write_text(json.dumps([serialize_result(result)], ensure_ascii=False, indent=2), encoding="utf-8")
        REPORT_MD.write_text(render_report([result]), encoding="utf-8")
        print(f"Missing dependency: {missing}")
        print(f"Markdown report: {REPORT_MD}")
        print(f"JSON report: {REPORT_JSON}")
        return 1

    from app.api.main import agent, app
    from app.config import settings

    settings.lastfm_api_key = ""
    settings.tavily_api_key = ""

    user_id = "long-dialogue-smoke"
    results = []
    results.extend(run_dialogue(agent, user_id))
    results.extend(run_api_checks(app, user_id))

    REPORT_JSON.write_text(json.dumps([serialize_result(r) for r in results], ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD.write_text(render_report(results), encoding="utf-8")

    failed = [item for item in results if not item.ok]
    print(f"Long dialogue smoke: {len(results) - len(failed)}/{len(results)} passed")
    print(f"Markdown report: {REPORT_MD}")
    print(f"JSON report: {REPORT_JSON}")
    if failed:
        print("\nFailures:")
        for item in failed:
            print(f"- {item.area}/{item.name}: {item.error or '; '.join(c.name for c in item.checks if not c.ok)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
