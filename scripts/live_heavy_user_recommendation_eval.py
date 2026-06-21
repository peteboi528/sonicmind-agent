from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"


def configure_live_env() -> None:
    os.environ["EXTERNAL_SOURCE"] = "netease"
    os.environ["LLM_API_KEY"] = ""
    os.environ["ENABLE_EMBEDDINGS"] = "false"
    os.environ["ENABLE_ONLINE_ENRICH"] = "false"
    os.environ["ENABLE_EMPTY_RESULT_RECOVERY"] = "true"


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").lower()).strip()


def _metadata_match(track: Any, detail: dict[str, Any] | None) -> bool:
    if not detail:
        return False
    title_ok = _norm(getattr(track, "title", "")) == _norm(str(detail.get("title") or ""))
    artist = _norm(getattr(track, "artist", ""))
    detail_artist = _norm(str(detail.get("artist") or ""))
    if not artist:
        artist_ok = bool(detail_artist)
    else:
        artist_ok = any(part and part in detail_artist for part in re.split(r"[、,/&;；]", artist))
    return bool(title_ok and artist_ok)


def _seed_heavy_user(agent: Any, user_id: str) -> None:
    from app.models import EpisodicMemory, MemoryEntry, TasteProfile, UserMemory

    prefs = [
        ("深夜 R&B / neo-soul，偏空间感、低饱和、松弛但不油腻", 12),
        ("ambient techno / IDM / glitch electronica，用于界面动效和系统设计专注", 10),
        ("bossa nova / city pop，喜欢通勤时轻盈、干净、带一点夏夜感", 9),
        ("post-rock / math rock，偏器乐、层次推进、适合视觉灵感板", 8),
        ("dream pop / shoegaze，喜欢女声、雾化吉他、柔和噪音墙", 8),
        ("jazz hip-hop / lo-fi beats，编码时要稳、温暖、有律动", 10),
        ("国风电子 / future bass 融合，但不要古装剧 OST 感", 7),
        ("Nordic folk / ambient folk / modern classical，偏冷感、留白、自然声场", 6),
    ]
    memory = UserMemory(
        user_id=user_id,
        structured_preferences=[
            MemoryEntry(text=text, frequency=freq, source="live_heavy_eval")
            for text, freq in prefs
        ],
        preferences=[text for text, _ in prefs],
        exclusion_rules=["抖音神曲", "type beat", "AI翻唱", "纯白噪音", "古装剧OST", "低质翻唱", "喊麦"],
        taste_profile=TasteProfile(
            top_genres=[
                ("R&B", 12.0), ("电子", 10.5), ("爵士", 9.5), ("后摇", 8.0),
                ("city pop", 8.0), ("dream pop", 7.5), ("国风电子", 6.5),
            ],
            top_moods=[("深夜", 10.0), ("专注", 9.5), ("空间感", 9.0), ("冷感", 7.0)],
            top_artists=[
                ("Frank Ocean", 9.0), ("SZA", 8.5), ("Four Tet", 8.0),
                ("Nujabes", 8.0), ("toe", 7.5), ("Cocteau Twins", 7.0),
            ],
        ),
        episodic_memory=[
            EpisodicMemory(text="最近连续 3 周都在做设计系统和动效原型，白天要专注，晚上要低饱和。"),
            EpisodicMemory(text="不想要被热搜/短视频神曲污染，宁可少一点也要真实可追溯。"),
        ],
    )
    agent.store.write_model("memory", user_id, memory)


LIVE_CASES = [
    ("night_rnb", "给我 6 首适合深夜做品牌 moodboard 的 R&B neo-soul，类似 SZA Frank Ocean Daniel Caesar，空间感强，不要抖音神曲。"),
    ("idm_design", "给我 6 首适合做界面动效设计时听的 ambient techno IDM glitch electronica，参考 Four Tet Jon Hopkins Aphex Twin。"),
    ("citypop_bossa", "推荐 6 首通勤听的 city pop bossa nova，类似 Lamp 小野リサ Tomoko Aran，轻盈但不俗。"),
    ("post_math_rock", "推荐 6 首适合搭视觉灵感板的 post-rock math rock，参考 toe Explosions in the Sky American Football。"),
    ("dream_shoegaze", "来 6 首女性主唱或柔和人声的 dream pop shoegaze，参考 Cocteau Twins Slowdive Alvvays。"),
    ("jazzhop_lofi", "给我 6 首编码专注用的 jazz hip-hop lo-fi beats，参考 Nujabes J Dilla Uyama Hiroto。"),
    ("guofeng_electronic", "推荐 6 首国风电子 future bass 融合，适合做东方视觉概念，不要古装剧 OST 感。"),
    ("nordic_ambient_folk", "推荐 6 首 Nordic folk ambient folk modern classical，冷感、有留白，适合整理设计文档。"),
]


def run_eval(top_k: int) -> dict[str, Any]:
    configure_live_env()
    from app.agent import (
        AudioVisualAgent,
        _extract_recommendation_anchors,
        _recommendation_anchor_hits,
    )
    from app.sources.netease import fetch_netease_song_detail
    from app.storage import JsonStore

    runtime = Path(tempfile.mkdtemp(prefix="musicagent-live-heavy-"))
    user_id = "heavy-live-online-user"
    agent = AudioVisualAgent(JsonStore(runtime / "store"))
    _seed_heavy_user(agent, user_id)

    results: list[dict[str, Any]] = []
    for case_id, query in LIVE_CASES:
        agent.memory.clear_dialogue_state(user_id)
        anchors = _extract_recommendation_anchors(query)
        started = time.time()
        try:
            recommendation = agent.recommend_for_query(user_id, query, top_k=top_k)
            crashed = False
            error = ""
        except Exception as exc:  # pragma: no cover - live diagnostic path
            recommendation = None
            crashed = True
            error = repr(exc)

        tracks = [item.asset for item in (recommendation.tracks if recommendation else [])]
        checked: list[dict[str, Any]] = []
        for track in tracks:
            detail = fetch_netease_song_detail(getattr(track, "external_id", "")) if getattr(track, "source", "") == "netease" else None
            hits = _recommendation_anchor_hits(track, anchors)
            checked.append({
                "title": getattr(track, "title", ""),
                "artist": getattr(track, "artist", ""),
                "source": getattr(track, "source", ""),
                "source_id": getattr(track, "external_id", ""),
                "detail_found": bool(detail),
                "metadata_match": _metadata_match(track, detail),
                "relevance_hits": hits,
            })
        returned = len(checked)
        relevant = sum(1 for item in checked if item["relevance_hits"])
        metadata_match = sum(1 for item in checked if item["metadata_match"])
        pass_case = (
            not crashed
            and metadata_match == returned
            and (relevant >= min(4, top_k) or (returned < top_k and returned == relevant))
        )
        results.append({
            "id": case_id,
            "query": query,
            "crashed": crashed,
            "error": error,
            "latency_s": round(time.time() - started, 2),
            "returned": returned,
            "metadata_match": metadata_match,
            "relevant": relevant,
            "pass": pass_case,
            "trace_tail": (recommendation.agent_trace[-8:] if recommendation else []),
            "tracks": checked,
        })

    aggregate = {
        "cases": len(results),
        "passed": sum(1 for item in results if item["pass"]),
        "crashes": sum(1 for item in results if item["crashed"]),
        "returned": sum(item["returned"] for item in results),
        "metadata_match": sum(item["metadata_match"] for item in results),
        "relevant": sum(item["relevant"] for item in results),
    }
    return {"runtime": str(runtime), "top_k": top_k, "aggregate": aggregate, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Live heavy-user recommendation accuracy eval")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any live case fails")
    args = parser.parse_args()

    payload = run_eval(max(1, min(args.top_k, 20)))
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.write_report:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = ARTIFACT_DIR / "live_heavy_user_recommendation_eval.json"
        path.write_text(text, encoding="utf-8")
    if args.strict and payload["aggregate"]["passed"] != payload["aggregate"]["cases"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
