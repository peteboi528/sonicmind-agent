"""搜索相关 prompt。"""

LLM_SEARCH_VERSION = "v1-2026-06-05"


def LLM_SEARCH_TEMPLATE(*, query: str, limit: int) -> str:
    return (
        f"根据搜索词「{query}」推荐{limit}首真实存在的歌曲。\n"
        f"请严格按 JSON 格式输出：\n"
        f'[{{"title":"歌名","artist":"歌手","genre":"风格","mood":"情绪"}}]\n'
        f"只输出 JSON，不要其他内容。"
    )
