import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api import main
from app.graph.finalize import compose_answer_stream_async
from app.intents import match_intent_by_keywords
from app.knowledge import resolve_music_entities
from app.models import AgentPlan
from app.reply_language import split_reply_request


@pytest.mark.parametrize(
    ("message", "expected", "lang"),
    [
        ("Recommend Spanish songs. Reply in English.", "Recommend Spanish songs", "en"),
        ("Dame canciones en español", "Dame canciones en español", None),
        ("Recommend rock tracks in English", "Recommend rock tracks in English", None),
        ("日本語で歌う曲を探して", "日本語で歌う曲を探して", None),
        ("日本語で、集中できるインストをおすすめして。", "集中できるインストをおすすめして", "ja"),
        ("Find an artist. Responde en español.", "Find an artist", "es"),
    ],
)
def test_reply_request_is_not_a_music_language_filter(message, expected, lang):
    assert split_reply_request(message) == (expected, lang)


@pytest.mark.parametrize(
    ("intent", "query", "text"),
    [
        ("recommend", "Reply in English.", "I could not find any verified tracks"),
        ("video", "日本語で答えて。", "条件に合う動画は見つかりませんでした"),
        ("artist_info", "Responde en español.", "No encontré información fiable"),
    ],
)
def test_localized_empty_replies_skip_the_model(monkeypatch, intent, query, text):
    def forbidden(*args, **kwargs):
        raise AssertionError("Empty reply should not depend on the configured model")

    monkeypatch.setattr(
        "app.graph.nodes.select_llm",
        lambda *_args: SimpleNamespace(
            agenerate_stream=forbidden,
        ),
    )

    async def run():
        return "".join(
            [part async for part in compose_answer_stream_async(query, [], AgentPlan(intent=intent), agent=object())]
        )

    assert text in asyncio.run(run())


def test_configured_reply_model_receives_the_language_instruction(monkeypatch):
    prompts = []

    async def generate(prompt, **kwargs):
        prompts.append(prompt)
        yield "This is a short answer in English."

    monkeypatch.setattr("app.graph.nodes.select_llm", lambda *_args: SimpleNamespace(agenerate_stream=generate))

    # Provide an agent-like value to select the stand-in for the configured model.
    async def run_with_model():
        plan = AgentPlan(intent="chat", response_language="en")
        return "".join(
            [
                part
                async for part in compose_answer_stream_async(
                    "Explain rhythm.", [], plan, agent=SimpleNamespace(memory=None)
                )
            ]
        )

    reply = asyncio.run(run_with_model())
    assert reply.startswith("This is a short answer")
    assert "Write your reply in English." in prompts[0]
    assert "用中文" not in prompts[0]


def test_named_release_and_followup_resolve_the_same_album():
    query = "讲讲 Frank Ocean 的 Blonde，特别是它的风格。"
    assert match_intent_by_keywords(query) == "album_deep_dive"
    first = resolve_music_entities(query, "album_deep_dive")[0]
    followup = resolve_music_entities(
        "这张专辑有什么代表作？",
        "album_deep_dive",
        {"retrieval_plan": {"entities": [first.name, first.artist]}},
    )[0]
    assert (first.type, first.name, first.artist) == ("album", "Blonde", "Frank Ocean")
    assert (followup.type, followup.name, followup.artist) == (first.type, first.name, first.artist)


def test_playlist_definition_does_not_run_music_tools():
    message = "Please explain what a playlist is. Reply in English."
    response = TestClient(main.app).post("/agent/stream", json={"message": message, "user_id": "definition-test"})
    response.raise_for_status()
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert not any(event["type"] == "tool_start" for event in events)
    final = events[-1]
    assert final["type"] == "final"
    assert final["payload"]["trace_summary"]["response_language"] == "en"
    assert final["payload"]["trace_summary"]["intent"] == "chat"
    assert "A playlist is a collection of tracks" in final["content"]
