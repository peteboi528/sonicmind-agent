"""URL 歌曲识别 prompt。"""

IDENTIFY_VERSION = "v1-2026-06-05"


def IDENTIFY_FROM_URL_TEMPLATE(*, url: str, parsed_title: str, video_title: str | None) -> str:
    title_info = f"视频标题: {video_title}\n" if video_title else ""
    return (
        f"从以下信息中识别这是哪首歌曲，以及歌手是谁。\n"
        f"URL: {url}\n"
        f"{title_info}"
        f"解析标题: {parsed_title}\n\n"
        f"请严格按以下格式回复（如果无法识别就填'未知'）：\n"
        f"歌名: xxx\n歌手: xxx\n风格: xxx\n情绪: xxx"
    )
