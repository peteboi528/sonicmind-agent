from __future__ import annotations

from app.llm.tools import AGENT_TOOLS, TOOL_PLAYLIST, TOOL_RECOMMEND, TOOL_SEARCH
from app.tools.registry import ALL_TOOL_NAMES, get_handler, normalize_tool_name, to_openai_tools


def test_registry_generates_agent_tools():
    built = to_openai_tools()
    assert AGENT_TOOLS == built
    names = {tool["function"]["name"] for tool in built}
    assert TOOL_RECOMMEND in names
    assert TOOL_SEARCH in names
    assert TOOL_PLAYLIST in names


def test_aliases_normalize_to_same_handler():
    assert normalize_tool_name("recommend_music") == "recommend"
    assert normalize_tool_name("search_music") == "search"
    assert normalize_tool_name("generate_playlist") == "playlist"
    assert get_handler("recommend_music") == get_handler("recommend")
    assert get_handler("import") == "import_netease_playlist"
    assert "recommend_music" in ALL_TOOL_NAMES
