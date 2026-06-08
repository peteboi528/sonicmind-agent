"""每日推荐和理由生成相关 prompt。"""

DAILY_RECOMMEND_VERSION = "v2-2026-06-05"


def DAILY_RECOMMEND_USER_TEMPLATE(
    *,
    count: int,
    prefs_desc: str,
    taste_desc: str,
    library_desc: str,
    bucket: str,
) -> str:
    return (
        f"你是一个音乐推荐引擎。根据用户的品味为其推荐{count}首歌曲。\n\n"
        f"用户偏好: {prefs_desc}\n"
        f"品味档案: {taste_desc}\n"
        f"用户库中的歌曲: {library_desc}\n"
        f"当前时段: {bucket}\n\n"
        f"要求:\n"
        f"1. 推荐与用户库中歌曲风格相似的其他歌曲（同类型歌手、同风格）\n"
        f"2. 70%推荐同风格的歌，30%推荐可能喜欢的新风格\n"
        f"3. 推荐真实存在的歌曲，包含歌名和歌手\n"
        f"4. 不要重复推荐用户库中已有的歌\n\n"
        f"请严格按以下JSON格式输出（不要输出其他内容）:\n"
        f'[{{"title":"歌名","artist":"歌手","genre":"风格","mood":"情绪","reason":"推荐理由(中文,20字内)"}}]\n'
    )


def DAILY_SUMMARY_TEMPLATE(*, count: int, genre_str: str, bucket: str) -> str:
    return (
        f"用一句中文总结今日推荐（不超过30字）。"
        f"共{count}首，主要风格：{genre_str}，时段：{bucket}。"
    )
