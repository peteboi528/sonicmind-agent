"""歌单生成相关 prompt。"""

PLAYLIST_VERSION = "v2-2026-06-07"


def GENERATE_PLAYLIST_TEMPLATE(*, instruction: str, library_size: int, lib_desc: str) -> str:
    return (
        f"用户指令：{instruction}\n\n"
        f"用户音乐库（{library_size}首）：\n{lib_desc}\n\n"
        f"规则（务必遵守）：\n"
        f"1. 优先从上面的音乐库里挑选匹配的歌曲，并填写其真实的 asset_id。\n"
        f"2. 只有库内歌曲明显不足时，才补充库外歌曲；补充的必须是真实存在、"
        f"广为人知的作品，歌名和歌手要准确，绝不可虚构。拿不准就不要写。\n"
        f"3. 库外歌曲的 asset_id 必须填 null。\n"
        f"4. 至少一半曲目应来自用户音乐库。\n"
        f"输出JSON（不要输出其他内容）：\n"
        f'{{"name":"歌单名","description":"一句话描述","tracks":['
        f'{{"title":"歌名","artist":"歌手","asset_id":"库中的id或null"}}]}}\n'
    )


def AUTO_PLAYLIST_TEMPLATE(*, library_size: int, lib_desc: str) -> str:
    return (
        f"以下是用户的音乐库（{library_size}首）：\n{lib_desc}\n\n"
        f"请按风格/情绪/能量等维度自动分成 3-5 个歌单。\n"
        f"输出JSON数组（不要输出其他内容）：\n"
        f'[{{"name":"歌单名","description":"描述","track_ids":["asset_id1","asset_id2"]}}]\n'
    )
