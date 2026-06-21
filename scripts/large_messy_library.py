"""大而杂候选库：用于压测推荐在杂乱大库下的稳定性与用户画像是否被污染。

复用 long_dialogue_smoke.install_fakes 的基础 patch（视频/专辑/导入/播放——
别让那些路径炸真实网络），只把 search_web_music 换成"大而杂"版本：
150+ 首横跨 9 流派，故意掺入 SEO 垃圾曲名、流派标签错乱、缺标签、同名不同人。
search 按查询关键词（流派/情绪/场景）命中打分返回，并混入 ~20% 噪声——
测推荐 rerank 能否把噪声压下去、从杂库里挑准，以及记忆层在复杂对话下不被污染。
"""
from __future__ import annotations

from typing import Any

# 流派 → 典型情绪/场景词（同时作为 search 命中与噪声生成的依据）
_GENRE_PROFILES: dict[str, dict[str, list[str]]] = {
    "电子": {"mood": ["专注", "放松", "律动", "空灵"], "scene": ["深夜", "跑步", "工作"]},
    "民谣": {"mood": ["治愈", "怀旧", "安静", "温暖"], "scene": ["旅行", "深夜", "散步"]},
    "R&B": {"mood": ["性感", "慵懒", "放松", "暗黑"], "scene": ["深夜", "约会", "开车"]},
    "古典": {"mood": ["宁静", "庄严", "专注", "宏大"], "scene": ["学习", "睡眠", "冥想"]},
    "说唱": {"mood": ["激昂", "自信", "硬核", "律动"], "scene": ["健身", "派对", "通勤"]},
    "国风": {"mood": ["悠远", "雅致", "怀旧", "宁静"], "scene": ["品茶", "阅读", "深夜"]},
    "摇滚": {"mood": ["激昂", "叛逆", "热血", "粗犷"], "scene": ["演唱会", "公路", "派对"]},
    "爵士": {"mood": ["慵懒", "优雅", "即兴", "微醺"], "scene": ["深夜", "酒吧", "约会"]},
    "流行": {"mood": ["欢快", "明亮", "伤感", "治愈"], "scene": ["通勤", "派对", "约会"]},
}

# 关键词索引：query 命中这些词时给对应流派/情绪加分
_GENRE_KEYWORDS = {
    "电子": ["电子", "电音", "edm", "synth", "techno", "house"],
    "民谣": ["民谣", "folk", "木吉他"],
    "R&B": ["r&b", "rnb", "节奏布鲁斯", "soul", "灵魂"],
    "古典": ["古典", "classic", "钢琴曲", "交响"],
    "说唱": ["说唱", "rap", "嘻哈", "hiphop", "hip-hop"],
    "国风": ["国风", "古风", "中国风"],
    "摇滚": ["摇滚", "rock", "金属", "metal"],
    "爵士": ["爵士", "jazz", "慵懒", "bossa", "巴萨诺瓦"],
    "流行": ["流行", "pop"],
}
_MOOD_KEYWORDS = {
    "放松": ["放松", "chill", "舒缓", "解压"],
    "深夜": ["深夜", "夜晚", "night", "失眠"],
    "专注": ["专注", "工作", "学习", "写代码", "focus"],
    "激昂": ["激昂", "高能量", "热血", "跑步", "健身", "upbeat"],
    "安静": ["安静", "宁静", "peaceful"],
    "慵懒": ["慵懒", "lazy", "慵"],
    "暗黑": ["暗黑", "dark", "氛围"],
    "治愈": ["治愈", "温暖", "温柔"],
    "怀旧": ["怀旧", "nostalgic", "经典"],
}

# SEO 垃圾片段（拼进曲名测推荐能否不被相关性差的热词带偏）
_SEO_NOISE = ["合集", "经典回顾", "热门推荐", "必听精选", "无损高音质", "抖音同款", "热门翻唱"]


def _build_messy_pool() -> list[Any]:
    from app.models import ExternalTrack

    pool: list[ExternalTrack] = []
    counter = 0
    for genre, profile in _GENRE_PROFILES.items():
        moods = profile["mood"]
        for i in range(16):  # 每流派 16 首 → ~144 首
            counter += 1
            mood = [moods[i % len(moods)]]
            # 流派交叉：少数歌多挂一个不相关流派（噪声）
            cross = []
            if i % 5 == 0:
                other = next(g for g in _GENRE_PROFILES if g != genre)
                cross = [other]
            title = f"{genre} Track {i:02d}"
            artist = f"Demo {genre} {i // 4 + 1}"
            pool.append(ExternalTrack(
                external_id=f"messy-{genre}-{i}",
                title=title,
                artist=artist,
                genre=[*cross, genre],
                mood=mood,
                source="netease",
                playback_url=f"https://music.163.com/song?id=900{counter:03d}",
            ))

    # SEO 垃圾曲名：相关性差、靠热词堆砌，推荐不该把它排到前面
    for j in range(18):
        pool.append(ExternalTrack(
            external_id=f"messy-seo-{j}",
            title=f"{_SEO_NOISE[j % len(_SEO_NOISE)]} #{j}",
            artist="SEO Collector",
            genre=["流行"],
            mood=["欢快"],
            source="netease",
            playback_url=f"https://music.163.com/song?id=910{j:02d}",
        ))

    # 标签错乱：R&B 的歌标成"古典"，测推荐是否被错误标签误导
    for k in range(10):
        pool.append(ExternalTrack(
            external_id=f"messy-mislabel-{k}",
            title=f"Mislabeled R&B {k}",
            artist=f"Confused Artist {k}",
            genre=["古典"],   # 实际是 R&B 但被错标
            mood=["性感"],
            source="netease",
            playback_url=f"https://music.163.com/song?id=920{k:02d}",
        ))

    # 缺标签：部分歌无流派/情绪，测推荐在信息不全时的鲁棒性
    for m in range(8):
        pool.append(ExternalTrack(
            external_id=f"messy-bare-{m}",
            title=f"Bare Title {m}",
            artist=f"Unknown {m}",
            genre=[],
            mood=[],
            source="netease",
            playback_url=f"https://music.163.com/song?id=930{m:02d}",
        ))

    # 同名不同人：测去重别误杀
    pool.append(ExternalTrack(external_id="dup-a", title="同名曲", artist="Singer A", genre=["流行"], mood=["伤感"], source="netease", playback_url="https://music.163.com/song?id=94001"))
    pool.append(ExternalTrack(external_id="dup-b", title="同名曲", artist="Singer B", genre=["摇滚"], mood=["激昂"], source="netease", playback_url="https://music.163.com/song?id=94002"))

    return pool


def _query_keywords(query: str) -> set[str]:
    """从 query 抽流派/情绪/场景关键词，用于命中打分。"""
    lowered = (query or "").lower()
    hits: set[str] = set()
    for canon, words in {**_GENRE_KEYWORDS, **_MOOD_KEYWORDS}.items():
        if any(w in lowered for w in words):
            hits.add(canon)
    return hits


def _score_track(track: Any, wanted: set[str]) -> int:
    text = " ".join([
        getattr(track, "title", "") or "",
        getattr(track, "artist", "") or "",
        *(getattr(track, "genre", []) or []),
        *(getattr(track, "mood", []) or []),
    ]).lower()
    return sum(1 for w in wanted if w.lower() in text)


def install_large_messy() -> None:
    """在 install_fakes 基础上，把 search_web_music 换成大而杂库版本。"""
    from app.agent import AudioVisualAgent
    from app.models import ExternalTrack
    from scripts.long_dialogue_smoke import install_fakes

    install_fakes()  # 基础 patch：视频/专辑/导入/播放，避免炸真实网络
    pool = _build_messy_pool()

    def fake_search_web_music(
        self: AudioVisualAgent,
        query: str,
        top_k: int = 5,
        relevance_query: str = "",
        offset: int = 0,
        **_: Any,
    ) -> list[ExternalTrack]:
        rel = relevance_query or query
        wanted = _query_keywords(rel)
        scored = sorted(((_score_track(t, wanted), t) for t in pool), key=lambda x: -x[0])
        # 取命中最高的若干 + 混入 ~20% 噪声（低分项），测 rerank 能否压住噪声。
        take = max(top_k + offset, top_k)
        top = [t for _, t in scored[:take]]
        if len(top) < take:
            return top[offset:offset + top_k] if offset else top[:top_k]
        # 末位替换 1 个为低分噪声项（若有的话）
        noise_pool = [t for _, t in scored[len(top):]]
        if noise_pool and top:
            top[-1] = noise_pool[0]
        return top[offset:offset + top_k] if offset else top[:top_k]

    AudioVisualAgent.search_web_music = fake_search_web_music
