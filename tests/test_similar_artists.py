from __future__ import annotations

import asyncio

from app.graph.nodes import _apply_dialogue_continuation, _planned_arguments, build_agent_plan
from app.intents import match_intent_by_keywords
from app.models import AgentPlan, ResourceTrack, RetrievalPlan
from app.services.tools import tool_runtime
from app.tools.contracts import ToolCall, ToolContext, ToolStatus


class _Agent:
    def list_resource_tracks(self, limit=100):
        return [
            ResourceTrack(title="Often", artist="The Weeknd", genre=["R&B"], mood=["放松"]),
            ResourceTrack(title="Starboy", artist="The Weeknd", genre=["R&B", "电子"], mood=["热血"]),
            ResourceTrack(title="Snooze", artist="SZA", genre=["R&B"], mood=["放松"]),
            ResourceTrack(title="Nights", artist="Frank Ocean", genre=["R&B"], mood=["放松"]),
            ResourceTrack(title="Faded", artist="Alan Walker", genre=["电子"], mood=["热血"]),
            ResourceTrack(title="Take Five", artist="Dave Brubeck", genre=["爵士"], mood=["放松"]),
        ]

    @staticmethod
    def artist_name_matches(query, artist):
        return query.strip().lower() == artist.strip().lower()


def test_similar_artist_keyword_has_dedicated_intent():
    assert match_intent_by_keywords("推荐同类型的歌手") == "similar_artists"
    plan = build_agent_plan("推荐同类型的歌手")
    assert plan.intent == "similar_artists"
    assert plan.tools_needed == ["similar_artists"]


def test_similar_artist_continuation_inherits_previous_artist():
    previous = {
        "entities": ["The Weeknd"], "last_intent": "search",
        "last_query": "找 The Weeknd 的歌", "genre_tags": [], "mood_tags": [], "scenario_tags": [],
        "shown_artists": [{"name": "SZA", "source": "local_library"}],
    }
    plan = AgentPlan(
        intent="similar_artists", tools_needed=["similar_artists"], online_required=False,
        # Mock/小模型可能把关系词误抽成实体；确定性层必须忽略它并继承真实歌手。
        retrieval_plan=RetrievalPlan(entities=["同类型"], search_query="同类型"),
    )
    inherited, _ = _apply_dialogue_continuation(plan, "推荐同类型的歌手", previous)
    assert inherited.intent == "similar_artists"
    assert inherited.retrieval_plan.entities == ["The Weeknd"]
    assert _planned_arguments("similar_artists", "推荐同类型的歌手", inherited, 5)["artist"] == "The Weeknd"
    assert inherited._excluded_artists[0]["name"] == "SZA"


def test_similar_artist_runtime_returns_traceable_local_artists():
    result = asyncio.run(tool_runtime.execute(
        ToolCall(name="similar_artists", arguments={"artist": "The Weeknd", "top_k": 4}),
        ToolContext(thread_id="t", user_id="u", query="推荐同类型的歌手", agent=_Agent()),
    ))
    assert result.status == ToolStatus.OK
    names = [artist["name"] for artist in result.data["artists"]]
    assert "The Weeknd" not in names
    assert names[:2] == ["Frank Ocean", "SZA"]
    assert all(artist["source"] == "local_library" for artist in result.data["artists"])


def test_similar_artist_runtime_excludes_previously_shown_artists():
    result = asyncio.run(tool_runtime.execute(
        ToolCall(name="similar_artists", arguments={"artist": "The Weeknd", "top_k": 4}),
        ToolContext(
            thread_id="t",
            user_id="u",
            query="再来一点",
            agent=_Agent(),
            plan={"_excluded_artists": [{"name": "Frank Ocean", "source": "local_library"}]},
        ),
    ))
    assert result.status == ToolStatus.OK
    names = [artist["name"] for artist in result.data["artists"]]
    assert "Frank Ocean" not in names
    assert names[0] == "SZA"
