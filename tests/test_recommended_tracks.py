from __future__ import annotations

import asyncio
import tempfile
from unittest.mock import patch

from app.agent import AudioVisualAgent
from app.models import AgentAnswer, ExternalTrack, TrackRef
from app.storage import JsonStore
from tests.eval.cases import EvalCase
from tests.eval.metrics import compute_metrics


def test_chat_populates_recommended_tracks():
    agent = AudioVisualAgent(JsonStore(tempfile.mkdtemp()))

    answer = asyncio.run(agent.chat_async("u-rec", "推荐几首适合跑步的歌"))

    assert isinstance(answer, AgentAnswer)
    assert answer.recommended_tracks
    first = answer.recommended_tracks[0]
    assert first.title
    assert first.source
    assert answer.prompt_versions.get("query_plan", "").startswith("v")
    assert any("[prompt]" in line for line in answer.agent_trace)
    assert answer.runtime_metrics["llm_calls"] >= 1
    assert "total_tokens" in answer.runtime_metrics
    assert any("[meta]" in line for line in answer.agent_trace)


def test_compute_metrics_includes_diversity_from_recommended_tracks():
    case = EvalCase(
        case_id="metric_diversity",
        description="test",
        user_id="u",
        query="推荐几首歌",
    )
    answer = AgentAnswer(
        answer="给你两首歌。",
        evidences=[],
        recommended_tracks=[
            TrackRef(title="Song A", source="netease", genre=["rock"], mood=["energetic"]),
            TrackRef(title="Song B", source="netease", genre=["classical"], mood=["calm"]),
        ],
    )

    metrics = compute_metrics(case, answer)

    assert metrics["recommended_tracks"] == 2
    assert metrics["diversity"] is not None
    assert 0.0 <= metrics["diversity"] <= 1.0


def test_afternoon_recommendation_admits_quality_tracks_not_functional_audio():
    agent = AudioVisualAgent(JsonStore(tempfile.mkdtemp()))

    playlist_hits = [
        ExternalTrack(external_id="bad-0", title="舒适的下午", artist="咖啡厅音乐", source="netease"),
        ExternalTrack(external_id="bad-1", title="放松chill（轻音乐）", artist="轻松治愈", source="netease"),
        ExternalTrack(external_id="bad-2", title="下午茶音乐 放松身心", artist="音眠治愈所", source="netease"),
        ExternalTrack(external_id="ok-1", title="Sunset Lover", artist="Petit Biscuit", source="netease"),
        ExternalTrack(external_id="ok-2", title="Sweet Disposition", artist="The Temper Trap", source="netease"),
    ]
    web_hits = [
        ExternalTrack(external_id="bad-3", title="午后放松时光（优美旋律）", artist="治愈音乐集", source="netease"),
        ExternalTrack(external_id="ok-3", title="踊り子", artist="Vaundy", source="netease"),
    ]

    with (
        patch("app.search.netease_playlist.search_and_extract", return_value=playlist_hits),
        patch.object(agent, "search_web_music", return_value=web_hits),
    ):
        rec = agent.recommend_for_query("u-afternoon", "推荐适合下午的歌", top_k=3)

    titles = [item.asset.title for item in rec.tracks]
    assert titles
    assert "Sunset Lover" in titles
    assert "Sweet Disposition" in titles
    assert "舒适的下午" not in titles
    assert "放松chill（轻音乐）" not in titles
    assert "下午茶音乐 放松身心" not in titles
    assert "午后放松时光（优美旋律）" not in titles
