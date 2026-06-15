from __future__ import annotations

import tempfile

from app.agent import AudioVisualAgent
from app.models import AgentAnswer, TrackRef
from app.storage import JsonStore
from tests.eval.cases import EvalCase
from tests.eval.metrics import compute_metrics


def test_chat_populates_recommended_tracks():
    agent = AudioVisualAgent(JsonStore(tempfile.mkdtemp()))

    answer = agent.chat("u-rec", "推荐几首适合跑步的歌")

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
