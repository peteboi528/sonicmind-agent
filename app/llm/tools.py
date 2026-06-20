"""Agent 工具集定义（OpenAI tools/function calling 格式）。

统一从 app.tools.registry 读取，避免图侧和 tool schema 漂移。
"""

from __future__ import annotations

from app.tools.registry import ALL_TOOL_NAMES, to_openai_tools

TOOL_RECOMMEND = "recommend"
TOOL_SEARCH = "search"
TOOL_PLAYLIST = "playlist"
TOOL_TASTE = "taste"
TOOL_SIMILAR_CROSS = "similar_cross"
TOOL_SIMILAR_INTRA = "similar_intra"
TOOL_RETRIEVE = "retrieve"
TOOL_ANALYZE = "analyze"
TOOL_REPORT = "report"
TOOL_MEMORY_UPDATE = "memory_update"
TOOL_WEB_MUSIC_SEARCH = "web_music_search"
TOOL_FETCH_METADATA = "fetch_metadata"
TOOL_IMPORT_NETEASE_PLAYLIST = "import_netease_playlist"
TOOL_FEEDBACK = "feedback"
TOOL_LISTENING_HISTORY = "listening_history"
TOOL_LIST_MY_PLAYLISTS = "list_my_playlists"
TOOL_VIDEO_SEARCH = "video_search"
TOOL_WEB_INFO_SEARCH = "web_info_search"
TOOL_FIND_ON_PLATFORM = "find_on_platform"
TOOL_LYRICS = "lyrics"
TOOL_AUDIO_FEATURES = "audio_features"
TOOL_SAVE_TO_PLAYLIST = "save_to_playlist"
TOOL_FAVORITE_TRACK = "favorite_track"
TOOL_CONCERT_EVENTS = "concert_events"

AGENT_TOOLS = to_openai_tools()

__all__ = [
    "AGENT_TOOLS",
    "ALL_TOOL_NAMES",
    "TOOL_ANALYZE",
    "TOOL_FETCH_METADATA",
    "TOOL_IMPORT_NETEASE_PLAYLIST",
    "TOOL_FEEDBACK",
    "TOOL_LISTENING_HISTORY",
    "TOOL_LIST_MY_PLAYLISTS",
    "TOOL_VIDEO_SEARCH",
    "TOOL_WEB_INFO_SEARCH",
    "TOOL_FIND_ON_PLATFORM",
    "TOOL_LYRICS",
    "TOOL_AUDIO_FEATURES",
    "TOOL_SAVE_TO_PLAYLIST",
    "TOOL_FAVORITE_TRACK",
    "TOOL_CONCERT_EVENTS",
    "TOOL_MEMORY_UPDATE",
    "TOOL_PLAYLIST",
    "TOOL_RECOMMEND",
    "TOOL_REPORT",
    "TOOL_RETRIEVE",
    "TOOL_SEARCH",
    "TOOL_SIMILAR_CROSS",
    "TOOL_SIMILAR_INTRA",
    "TOOL_TASTE",
    "TOOL_WEB_MUSIC_SEARCH",
]
