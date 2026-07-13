from __future__ import annotations

from app.models import AgentAnswer, AgentPlan, StreamEvent
from app.tools.checkpoint_serde import SanitizingCheckpointSerializer, sanitize_checkpoint_value


def test_checkpoint_sanitizer_removes_lyrics_credentials_binary_and_old_history():
    history = [{"role": "user", "content": f"message-{index}"} for index in range(30)]
    payload = {
        "plan": AgentPlan(intent="recommend", tools_needed=["recommend"]),
        "history": history,
        "results": [
            {
                "type": "lyrics",
                "title": "Song",
                "lines": ["secret lyric line one", "secret lyric line two"],
            }
        ],
        "context": {"cookie": "MUSIC_U=secret-cookie", "api_key": "secret-key"},
        "raw_audio": b"audio-secret-bytes",
        "answer": AgentAnswer(answer="complete lyric response", evidences=[]),
        "events": [StreamEvent(type="final", content="complete lyric response", payload={"answer": "complete"})],
    }

    sanitized = sanitize_checkpoint_value(payload)

    assert isinstance(sanitized["plan"], AgentPlan)
    assert sanitized["results"][0]["line_count"] == 2
    assert "lines" not in sanitized["results"][0]
    assert sanitized["context"] == {"cookie": "[redacted]", "api_key": "[redacted]"}
    assert sanitized["raw_audio"] == "[omitted]"
    assert len(sanitized["history"]) == 20
    assert sanitized["history"][0]["content"] == "message-10"
    assert isinstance(sanitized["answer"], AgentAnswer)
    assert sanitized["answer"].answer == "[lyrics response omitted from checkpoint]"
    assert sanitized["events"][0].content == "[lyrics omitted]"


def test_checkpoint_serializer_round_trip_keeps_models_without_secret_text():
    serializer = SanitizingCheckpointSerializer()
    kind, data = serializer.dumps_typed(
        {
            "plan": AgentPlan(intent="recommend", tools_needed=["recommend"]),
            "results": [{"type": "lyrics", "lines": ["do-not-persist-this-lyric"]}],
            "authorization": "Bearer do-not-persist-this-token",
        }
    )
    loaded = serializer.loads_typed((kind, data))

    assert isinstance(loaded["plan"], AgentPlan)
    assert loaded["results"][0] == {"type": "lyrics", "line_count": 1}
    assert loaded["authorization"] == "[redacted]"
    assert b"do-not-persist-this-lyric" not in data
    assert b"do-not-persist-this-token" not in data
