"""曲风词表的单一事实来源（Single Source of Truth）。

历史上 _VALID_GENRES / 分类 prompt / netease tag 映射 / Last.fm 英文映射散落在 8 个文件里，
各自维护一份 10 个一级标签的字面量，扩充时极易漂移。这里集中维护，其余模块一律 import。

设计：一级词表从 10 扩到 ~30，覆盖主流可辨听感差异（中文/英文说唱分流、独立、英伦摇滚、
盯鞋、氛围、lo-fi、City Pop 等）。一首歌最多挂 3 个标签（在分类层裁剪）。扩词表后：
- 库匹配（knowledge._match_library_to_entity）按更细的 genre 交集命中，更准；
- 品味画像 top_genres 粒度变细，推荐 rerank 的曲风信号更聪明；
- netease tag / Last.fm 映射同步扩，导入与发现都吃得到细分。
"""

from __future__ import annotations

# ── 一级曲风词表（标准名）────────────────────────────────────────────────────
# 顺序无意义，集合判定。新增务必是「听感可辨、有人会说自己喜欢」的粒度，避免过细噪音。
VALID_GENRES: list[str] = [
    # 原 10 个一级（保持兼容，老数据/老测试不受影响）
    "流行",
    "摇滚",
    "电子",
    "古典",
    "R&B",
    "说唱",
    "爵士",
    "民谣",
    "国风",
    "金属",
    # 流行细分
    "华语流行",
    "欧美流行",
    "韩流",
    "日系流行",
    "City Pop",
    "合成器流行",
    # 摇滚细分
    "英伦摇滚",
    "独立摇滚",
    "朋克",
    "后摇",
    "盯鞋",
    "硬摇滚",
    # 电子细分
    "House",
    "Techno",
    "氛围电子",
    "未来贝斯",
    "synthwave",
    # 说唱细分（你最关心的中英文分流）
    "中文说唱",
    "欧美说唱",
    "Trap",
    "Drill",
    # R&B / soul 细分
    "新灵魂",
    "另类R&B",
    "放克",
    # 其他常见
    "独立",
    "氛围",
    "lo-fi",
    "原声",
    "世界音乐",
    "拉丁",
    "雷鬼",
    "蓝调",
    "乡村",
    "古风",
    "后朋克",
    "实验",
]

VALID_GENRE_SET: set[str] = set(VALID_GENRES)

# ── 子风格 → 一级父类的归并（库匹配/品味聚合时可上卷到父类做粗粒度命中）─────────
# 让「中文说唱」既能精确命中，又能在需要粗粒度时回退到「说唱」。
GENRE_PARENT: dict[str, str] = {
    "华语流行": "流行",
    "欧美流行": "流行",
    "韩流": "流行",
    "日系流行": "流行",
    "City Pop": "流行",
    "合成器流行": "流行",
    "英伦摇滚": "摇滚",
    "独立摇滚": "摇滚",
    "朋克": "摇滚",
    "后摇": "摇滚",
    "盯鞋": "摇滚",
    "硬摇滚": "摇滚",
    "后朋克": "摇滚",
    "House": "电子",
    "Techno": "电子",
    "氛围电子": "电子",
    "未来贝斯": "电子",
    "synthwave": "电子",
    "中文说唱": "说唱",
    "欧美说唱": "说唱",
    "Trap": "说唱",
    "Drill": "说唱",
    "新灵魂": "R&B",
    "另类R&B": "R&B",
    "放克": "R&B",
    "古风": "国风",
    "蓝调": "爵士",
    "乡村": "民谣",
    "原声": "民谣",
}


def parent_genre(genre: str) -> str:
    """取一级父类；本身就是一级或未知则原样返回。"""
    return GENRE_PARENT.get(genre, genre)


# ── 网易云歌单 tags → 本系统曲风（导入时唯一可靠的曲风线索，扩到细分）──────────
NETEASE_TAG_TO_GENRE: dict[str, str] = {
    # 流行（注意：欧美/华语/粤语/日语/韩语是地区/语种标签，不是曲风，不映射——
    # 一个"欧美"歌单可能是摇滚/说唱/任何风格，硬映射成流行会污染。语种加权另有专门逻辑。）
    "流行": "流行",
    "Pop": "流行",
    "City Pop": "City Pop",
    "都市流行": "City Pop",
    "K-Pop": "韩流",
    "电子流行": "合成器流行",
    # 摇滚
    "摇滚": "摇滚",
    "Rock": "摇滚",
    "朋克": "朋克",
    "Punk": "朋克",
    "英伦": "英伦摇滚",
    "Britpop": "英伦摇滚",
    "独立摇滚": "独立摇滚",
    "后摇": "后摇",
    "Post-Rock": "后摇",
    "盯鞋": "盯鞋",
    "Shoegaze": "盯鞋",
    # 金属
    "金属": "金属",
    "Metal": "金属",
    # 电子
    "电子": "电子",
    "Electronic": "电子",
    "House": "House",
    "Techno": "Techno",
    "EDM": "电子",
    "Ambient": "氛围电子",
    "氛围": "氛围",
    # 说唱
    "说唱": "说唱",
    "Rap": "说唱",
    "Hip-Hop": "欧美说唱",
    "嘻哈": "说唱",
    "中文说唱": "中文说唱",
    "华语说唱": "中文说唱",
    "Trap": "Trap",
    # R&B / soul
    "R&B/Soul": "R&B",
    "R&B": "R&B",
    "Soul": "新灵魂",
    "蓝调": "蓝调",
    "Funk": "放克",
    # 爵士
    "爵士": "爵士",
    "Jazz": "爵士",
    "布鲁斯": "蓝调",
    # 古典
    "古典": "古典",
    "Classical": "古典",
    "纯音乐": "古典",
    # 民谣 / 原声
    "民谣": "民谣",
    "Folk": "民谣",
    "乡村": "乡村",
    "Country": "乡村",
    "原声": "原声",
    # 国风
    "国风": "国风",
    "古风": "古风",
    "中国风": "国风",
    # lo-fi / 世界
    "Lo-Fi": "lo-fi",
    "拉丁": "拉丁",
    "Latin": "拉丁",
    "雷鬼": "雷鬼",
    "Reggae": "雷鬼",
    "世界音乐": "世界音乐",
}

# ── 一级/细分曲风 → Last.fm 英文 tag（发现页用英文召回）──────────────────────
GENRE_TO_LASTFM_EN: dict[str, str] = {
    "流行": "pop",
    "华语流行": "mandopop",
    "欧美流行": "pop",
    "韩流": "k-pop",
    "日系流行": "j-pop",
    "City Pop": "city pop",
    "合成器流行": "synthpop",
    "摇滚": "rock",
    "英伦摇滚": "britpop",
    "独立摇滚": "indie rock",
    "朋克": "punk",
    "后摇": "post-rock",
    "盯鞋": "shoegaze",
    "硬摇滚": "hard rock",
    "后朋克": "post-punk",
    "电子": "electronic",
    "House": "house",
    "Techno": "techno",
    "氛围电子": "ambient",
    "未来贝斯": "future bass",
    "synthwave": "synthwave",
    "说唱": "hip-hop",
    "中文说唱": "chinese hip-hop",
    "欧美说唱": "hip-hop",
    "Trap": "trap",
    "Drill": "drill",
    "R&B": "r&b",
    "新灵魂": "neo-soul",
    "另类R&B": "alternative r&b",
    "放克": "funk",
    "爵士": "jazz",
    "蓝调": "blues",
    "古典": "classical",
    "民谣": "folk",
    "乡村": "country",
    "原声": "acoustic",
    "国风": "chinese",
    "古风": "chinese traditional",
    "金属": "metal",
    "独立": "indie",
    "氛围": "ambient",
    "lo-fi": "lo-fi",
    "世界音乐": "world",
    "拉丁": "latin",
    "雷鬼": "reggae",
    "实验": "experimental",
}
