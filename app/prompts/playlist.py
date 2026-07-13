"""歌单生成相关 prompt。"""

PLAYLIST_VERSION = "v3-2026-06-11"


def GENERATE_PLAYLIST_TEMPLATE(
    *,
    instruction: str,
    library_size: int,
    lib_desc: str,
    target_count: int,
    candidate_desc: str = "",
    taste_summary: str = "",
    exclusion_rules: list[str] | None = None,
) -> str:
    candidate_block = f"\n联网/外部/上游工具候选：\n{candidate_desc}\n" if candidate_desc else ""
    taste_block = ""
    if taste_summary:
        taste_block = f"\n用户品味档案：{taste_summary}\n"
    exclusion_block = ""
    if exclusion_rules:
        exclusion_block = f"\n用户明确排除：{'、'.join(exclusion_rules)}\n（上述风格/类型绝对不要出现在歌单里。）\n"
    return (
        f"用户指令：{instruction}\n\n"
        f"目标曲目数：{target_count}首\n\n"
        f"{taste_block}"
        f"{exclusion_block}"
        f"用户音乐库（{library_size}首）：\n{lib_desc}\n\n"
        f"{candidate_block}"
        f"规则（务必遵守）：\n"
        f"1. 最终 tracks 尽量接近目标曲目数，不要只给 5-10 首。\n"
        f"2. 如果用户要求联网、新作品、大数量歌单，优先使用联网/外部候选，再用本地库补充。\n"
        f"3. 本地库歌曲要填写真实 asset_id；库外歌曲的 asset_id 必须填 null。\n"
        f"4. 库外歌曲必须是真实存在、"
        f"广为人知的作品，歌名和歌手要准确，绝不可虚构。拿不准就不要写。\n"
        f"5. 优先选符合用户品味档案的曲目；如果品味偏 R&B 就不要全选摇滚。\n"
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
