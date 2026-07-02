from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.media.pipeline import netease_song_id
from app.models import Asset, ExternalTrack
from app.rules.discover import _normalize_match_text, _scene_playlist_queries


def _netease_song_id(url: str) -> str | None:
    """从各种网易云 URL 格式中提取 song id。"""
    return netease_song_id(url)


def _infer_playlist_count(text: str) -> int | None:
    match = re.search(r"(\d{1,3})\s*(?:首|个|tracks?|songs?)?", text, re.IGNORECASE)
    if not match:
        return None
    return max(1, min(int(match.group(1)), 100))


def get_time_bucket_name() -> str:
    from app.recommend.daily import get_time_bucket

    names = {
        "morning": "早上",
        "focus": "工作学习",
        "afternoon": "下午",
        "evening": "晚上",
        "night": "深夜",
    }
    return names.get(get_time_bucket(), "今天")


@dataclass(frozen=True)
class RecommendationAnchors:
    artists: tuple[str, ...] = ()
    styles: tuple[str, ...] = ()
    negatives: tuple[str, ...] = ()

    @property
    def explicit(self) -> bool:
        return bool(self.artists or self.styles)


_KNOWN_RECOMMENDATION_ARTISTS = (
    "SZA", "Frank Ocean", "Daniel Caesar",
    "Four Tet", "Jon Hopkins", "Aphex Twin",
    "Lamp", "小野リサ", "Lisa Ono", "Tomoko Aran", "亜蘭知子",
    "toe", "Explosions in the Sky", "American Football",
    "Cocteau Twins", "Slowdive", "Alvvays",
    "Nujabes", "J Dilla", "Uyama Hiroto",
)

# LLM entities 里偶尔混进的非艺人词（场景/情绪/请求词）——这些不能当艺人硬锚点，
# 否则会用一个泛词去 anchor_filter 把所有真实候选清空。genre/mood 走 styles 分支。
_NON_ANCHOR_ENTITY_WORDS = {
    "", "推荐", "音乐", "歌曲", "歌", "song", "songs", "music", "track", "tracks",
    "chill", "vibe", "vibes", "playlist", "mix", "放松", "深夜", "学习", "工作", "跑步",
    "随便", "好听", "来点", "来几首", "更多", "similar", "more",
}

_RECOMMENDATION_STYLE_ALIASES: dict[str, tuple[str, ...]] = {
    "R&B": ("r&b", "rnb", "neo-soul", "neo soul", "soul", "节奏布鲁斯"),
    "ambient techno": ("ambient techno", "ambient", "techno"),
    "IDM": ("idm", "glitch electronica", "electronica", "glitch"),
    "city pop": ("city pop", "city-pop", "シティポップ"),
    "bossa nova": ("bossa nova", "bossa", "巴萨诺瓦"),
    "post-rock": ("post-rock", "post rock", "后摇"),
    "math rock": ("math rock", "数学摇滚"),
    "dream pop": ("dream pop", "dream-pop"),
    "shoegaze": ("shoegaze", "盯鞋"),
    "jazz hip-hop": ("jazz hip-hop", "jazz hip hop", "jazzhop", "jazz-hop"),
    "lo-fi": ("lo-fi", "lofi", "lo fi", "lo-fi beats"),
    "国风电子": ("国风电子", "国风 电", "中国风 电子", "国风", "中国风"),
    "future bass": ("future bass", "futurebass"),
    "Nordic folk": ("nordic folk", "北欧民谣", "北欧 folk"),
    "ambient folk": ("ambient folk",),
    "modern classical": ("modern classical", "modern-classical", "neoclassical", "neo classical", "新古典"),
}

_RECOMMENDATION_STYLE_SEARCH_VARIANTS: dict[str, tuple[str, ...]] = {
    "R&B": ("R&B", "neo soul", "Frank Ocean", "Daniel Caesar", "SZA"),
    "ambient techno": ("ambient techno", "Jon Hopkins", "Four Tet"),
    "IDM": ("IDM", "Aphex Twin", "glitch electronica"),
    "city pop": ("city pop", "Lamp", "Tomoko Aran", "小野リサ"),
    "bossa nova": ("bossa nova", "小野リサ", "Lisa Ono"),
    "post-rock": ("post-rock", "Explosions in the Sky", "toe"),
    "math rock": ("math rock", "toe", "American Football"),
    "dream pop": ("dream pop", "Cocteau Twins", "Alvvays"),
    "shoegaze": ("shoegaze", "Slowdive", "Cocteau Twins"),
    "jazz hip-hop": ("jazz hip-hop", "Nujabes", "J Dilla"),
    "lo-fi": ("lo-fi beats", "Nujabes", "Uyama Hiroto"),
    "国风电子": ("国风电子", "中国风 电子", "徐梦圆"),
    "future bass": ("future bass", "徐梦圆"),
    "Nordic folk": ("Nordic folk", "北欧民谣"),
    "ambient folk": ("ambient folk",),
    "modern classical": ("modern classical", "neoclassical"),
}

_RUNNING_PLAYLIST_TOKENS = ("跑步", "运动", "健身", "慢跑", "夜跑", "running", "workout", "gym", "run ")

_RUNNING_PLAYLIST_ANTI_CONTEXT_PATTERNS = (
    r"白噪音", r"粉噪音", r"褐噪音", r"助眠", r"催眠", r"睡前", r"睡眠", r"入睡",
    r"工作学习", r"学习工作", r"学习(?:时|用|专注|必备|音乐)?", r"专注力", r"提高专注",
    r"放松(?:音乐|催眠|冥想|疗愈)?", r"冥想", r"spa", r"asmr",
    r"雨声", r"雨水声", r"下雨声", r"雷声", r"雨林", r"自然(?:声|白噪音)",
    r"高音质.*/.*白噪音", r"睡觉", r"安眠",
    r"white\s*noise", r"brown\s*noise", r"pink\s*noise", r"sleep(?:ing)?",
    r"study(?:ing)?", r"focus", r"concentrat(?:e|ion)", r"meditation", r"relax(?:ing|ation)?",
    r"rain\s*sounds?", r"thunder\s*sounds?", r"lullaby",
)

_LONG_VIDEO_RECOMMENDATION_MARKERS = (
    "推荐歌曲", "日推", "一定要带上耳机", "带上耳机", "不能错过", "歌单推荐",
    "热门推荐", "音乐分享", "高音质", "完整版", "合集", "纯享", "一小时", "1小时",
)

_FUNCTIONAL_AUDIO_PATTERNS = (
    r"白噪音", r"粉噪音", r"褐噪音", r"助眠", r"催眠", r"睡前", r"睡眠", r"入睡",
    r"工作学习", r"学习工作", r"学习(?:时|用|专注|必备|音乐)?", r"专注力", r"提高专注",
    r"放松(?:音乐|身心|催眠|冥想|疗愈|解压)?", r"舒缓解压", r"解压放松",
    r"轻音乐", r"纯音乐", r"背景音乐", r"咖啡厅音乐", r"下午茶音乐", r"午后放松时光",
    r"舒适的下午", r"冥想", r"spa", r"asmr", r"雨声", r"雨水声", r"下雨声", r"雷声",
    r"雨林", r"自然(?:声|白噪音)", r"睡觉", r"安眠",
    r"white\s*noise", r"brown\s*noise", r"pink\s*noise", r"sleep(?:ing)?",
    r"study(?:ing)?", r"focus", r"concentrat(?:e|ion)", r"meditation", r"relax(?:ing|ation)?",
    r"coffee\s*(?:shop|house|cafe)\s*music", r"background\s*music",
    r"rain\s*sounds?", r"thunder\s*sounds?", r"lullaby",
)

_FUNCTIONAL_AUDIO_ARTIST_PATTERNS = (
    r"纯音乐馆", r"轻松治愈", r"音眠治愈所", r"治愈音乐集", r"解压放松治愈",
    r"休闲音乐", r"睡眠音乐", r"助眠音乐", r"白噪音", r"背景音乐", r"轻音乐",
    r"咖啡厅音乐", r"咖啡馆音乐", r"咖啡音乐", r"放松治愈", r"催眠", r"冥想",
    r"sleep\s*music", r"relax(?:ing|ation)?\s*music", r"study\s*music",
    r"coffee\s*(?:shop|house|cafe)\s*music",
)

_GENERIC_SCENE_TITLE_COMPACTS = {
    "放松chill轻音乐",
    "放松chill",
    "舒适的下午",
    "下午茶音乐放松身心",
    "午后放松时光优美旋律",
    "轻松放松舒缓解压",
    "工作学习时听的音乐提高专注力",
    "工作学习时听的音乐",
    "提高专注力",
    "雨林下雨声自然白噪音工作学习睡觉",
    "高音质白噪音雨水声雷声放松催眠学习睡前音乐工作学习必备",
}

_GENERIC_SCENE_ARTIST_COMPACTS = {
    "轻松治愈", "音眠治愈所", "治愈音乐集", "解压放松治愈",
    "纯音乐馆", "休闲音乐", "睡眠音乐", "助眠音乐", "背景音乐",
    "咖啡厅音乐", "咖啡馆音乐", "咖啡音乐",
}


def _extract_recommendation_anchors(query: str, entities: list[str] | None = None) -> RecommendationAnchors:
    lowered = (query or "").lower()
    artists: list[str] = []
    for artist in _KNOWN_RECOMMENDATION_ARTISTS:
        if artist.lower() in lowered or artist in query:
            artists.append(artist)

    # LLM 抽取的实体（歌手/歌名/专辑）——这是硬编码 _KNOWN_RECOMMENDATION_ARTISTS 名单
    # 覆盖不到的长尾艺人（如 The Weeknd）的锚点来源。有它们才让 anchors.explicit=True，
    # 从而在 recommend_for_query 里触发艺人过滤（否则本地库里用户爱听的其他歌手会漏进来）。
    # 只接受出现在本轮 query 里的实体——延续指令继承的旧实体不在本句时不作硬锚点，
    # 避免"再来几首"把上一轮艺人当成本轮强约束。
    for entity in entities or []:
        value = str(entity or "").strip()
        if not value or value.lower() in _NON_ANCHOR_ENTITY_WORDS:
            continue
        if value.lower() in lowered or value in query:
            artists.append(value)

    styles: list[str] = []
    for style, aliases in _RECOMMENDATION_STYLE_ALIASES.items():
        if any(alias.lower() in lowered for alias in aliases):
            styles.append(style)

    negatives: list[str] = []
    for match in re.finditer(r"(?:不要|别推|不想听|排除|避开)\s*([^，。,.!?！？]{1,24})", query or ""):
        value = match.group(1).strip()
        value = re.sub(r"(?:的)?(?:歌曲|音乐|风格|歌)?[吧呀啊啦了呢]*$", "", value).strip()
        if value:
            negatives.append(value)
    return RecommendationAnchors(
        artists=tuple(dict.fromkeys(artists)),
        styles=tuple(dict.fromkeys(styles)),
        negatives=tuple(dict.fromkeys(negatives)),
    )


def _recommendation_search_seeds(
    search_goal: str,
    original_query: str,
    anchors: RecommendationAnchors,
    upstream_variants: list[str] | None = None,
) -> list[str]:
    seeds: list[str] = []

    def add(value: str) -> None:
        value = (value or "").strip()
        if value and value.lower() not in {item.lower() for item in seeds}:
            seeds.append(value)

    for artist in anchors.artists:
        add(artist)
    for style in anchors.styles:
        for variant in _RECOMMENDATION_STYLE_SEARCH_VARIANTS.get(style, (style,)):
            add(variant)
    if anchors.artists and anchors.styles:
        for artist in anchors.artists[:4]:
            for style in anchors.styles[:2]:
                add(f"{artist} {style}")
    for item in upstream_variants or []:
        add(item)
    add(search_goal)
    if not seeds:
        add(original_query)
    return seeds[: max(1, settings.max_search_variants + 8)]


def _recommendation_anchor_hits(track: Asset | ExternalTrack, anchors: RecommendationAnchors) -> list[str]:
    if not anchors.explicit:
        return []
    raw = " ".join([
        getattr(track, "title", "") or "",
        getattr(track, "artist", "") or "",
        " ".join(getattr(track, "genre", []) or []),
        " ".join(getattr(track, "mood", []) or []),
    ]).lower()
    normalized = _normalize_match_text(raw)
    hits: list[str] = []
    for artist in anchors.artists:
        key = _normalize_match_text(artist)
        if key and key in normalized:
            hits.append(artist)
    for style in anchors.styles:
        aliases = _RECOMMENDATION_STYLE_ALIASES.get(style, (style,))
        if any(_normalize_match_text(alias) and _normalize_match_text(alias) in normalized for alias in aliases):
            hits.append(style)
    return list(dict.fromkeys(hits))


def _track_matches_recommendation_anchors(track: Asset | ExternalTrack, anchors: RecommendationAnchors) -> bool:
    return bool(_recommendation_anchor_hits(track, anchors))


def _is_recommendation_quality_track(track: Any, *, allow_variants: bool = False) -> bool:
    title = (getattr(track, "title", "") or "").strip()
    artist = (getattr(track, "artist", "") or "").strip()
    if not title or not artist:
        return False
    if allow_variants:
        return True

    lowered_title = title.lower()
    lowered_artist = artist.lower()
    compact_title = re.sub(r"[^a-z0-9一-鿿㐀-䶿]+", "", lowered_title)
    compact_artist = re.sub(r"[^a-z0-9一-鿿㐀-䶿]+", "", lowered_artist)

    noise_patterns = (
        r"\btype\s*beat\b", r"\bfree\s*beat\b", r"\binstrumental\b",
        r"\bbpm\s*\d+\b", r"\bprod\.?(?:\s|$)", r"\bdemo\b",
        r"\bai\s*(?:翻唱|cover|歌曲|音乐)", r"\blive\s*stage\b",
    )
    if any(re.search(pattern, lowered_title) for pattern in noise_patterns):
        return False
    if any(token in lowered_title for token in (
        "伴奏", "翻自", "动态歌词", "歌词版", "无损音质", "完整版试听",
        "步频", "卡点",
    )):
        return False
    if "#" in title or title.startswith(("【free】", "[free]", "（free）", "(free)")):
        return False
    if re.search(r"[（(].{0,8}(?:版|翻唱|cover).{0,4}[)）]\s*$", lowered_title):
        return False

    generic_names = {
        "热门歌曲", "热门音乐", "流行音乐", "独立流行", "另类流行", "新灵魂",
        "更新灵魂", "氛围说唱", "律动rnb", "律动rb", "另类rnb", "另类rb",
        "小众rnb", "小众rb", "neosoulbeat", "neosoul", "纯音乐", "伴奏",
        "typebeat", "unknown", "佚名", "群星", "rnb", "rb", "律动", "说唱",
        "学习专注", "专注学习", "学习能量", "专注力读书音乐", "学习专注力读书音乐",
    }
    if compact_title in generic_names or compact_artist in generic_names:
        return False
    if compact_title == compact_artist:
        return False

    genre_words = (
        "r&b", "rnb", "soul", "说唱", "嘻哈", "trap", "chill", "beat",
        "另类", "流行", "爵士", "电子", "梦核", "雷鬼",
    )
    descriptor_count = sum(1 for word in genre_words if word in lowered_title)
    if descriptor_count >= 3:
        return False
    if len(title) > 38 and descriptor_count >= 2:
        return False
    return True


def _is_scene_recommendation_instruction(instruction: str) -> bool:
    lowered = (instruction or "").lower()
    strict_scene_tokens = (
        *_RUNNING_PLAYLIST_TOKENS,
        "通勤", "开车", "派对", "聚会", "散步", "旅行", "约会",
    )
    return bool(_scene_playlist_queries(instruction)) or any(token in lowered for token in strict_scene_tokens)


def _looks_like_functional_audio(track: Any) -> bool:
    title = (getattr(track, "title", "") or "").strip()
    artist = (getattr(track, "artist", "") or "").strip()
    lowered_title = title.lower()
    lowered_artist = artist.lower()
    combined = f"{lowered_title} {lowered_artist}"
    compact = re.sub(r"\s+", "", combined)
    title_compact = re.sub(r"[^a-z0-9一-鿿㐀-䶿]+", "", lowered_title)
    artist_compact = re.sub(r"[^a-z0-9一-鿿㐀-䶿]+", "", lowered_artist)

    if title_compact in _GENERIC_SCENE_TITLE_COMPACTS or artist_compact in _GENERIC_SCENE_ARTIST_COMPACTS:
        return True
    if any(re.search(pattern, combined) or re.search(pattern, compact) for pattern in _FUNCTIONAL_AUDIO_PATTERNS):
        return True
    if any(re.search(pattern, lowered_artist) or re.search(pattern, artist_compact) for pattern in _FUNCTIONAL_AUDIO_ARTIST_PATTERNS):
        return True
    scene_words = ("下午", "午后", "放松", "治愈", "舒缓", "解压", "chill", "睡眠", "学习", "专注", "咖啡")
    title_scene_hits = sum(1 for word in scene_words if word in lowered_title)
    if title_scene_hits >= 2 and (len(title) > 12 or artist_compact in _GENERIC_SCENE_ARTIST_COMPACTS):
        return True
    return False


def _is_playlist_context_compatible(instruction: str, track: Any) -> bool:
    lowered_instruction = (instruction or "").lower()
    if not _is_scene_recommendation_instruction(instruction):
        return True

    if _looks_like_functional_audio(track):
        return False

    title = (getattr(track, "title", "") or "").strip()
    artist = (getattr(track, "artist", "") or "").strip()
    source = (getattr(track, "source", "") or "").lower()
    candidate_kind = (getattr(track, "candidate_kind", "") or "").lower()
    lowered_title = title.lower()
    lowered_artist = artist.lower()
    combined = f"{lowered_title} {lowered_artist}"
    compact = re.sub(r"\s+", "", combined)

    if any(token in lowered_instruction for token in _RUNNING_PLAYLIST_TOKENS) and any(
        re.search(pattern, combined) or re.search(pattern, compact)
        for pattern in _RUNNING_PLAYLIST_ANTI_CONTEXT_PATTERNS
    ):
        return False

    if source in {"bilibili", "youtube"} or candidate_kind in {"video", "compilation"}:
        prose_like = len(title) > 24 or any(mark in title for mark in _LONG_VIDEO_RECOMMENDATION_MARKERS)
        if prose_like and any(mark in title for mark in _LONG_VIDEO_RECOMMENDATION_MARKERS):
            return False

    title_compact = re.sub(r"[^a-z0-9一-鿿㐀-䶿]+", "", lowered_title)
    artist_compact = re.sub(r"[^a-z0-9一-鿿㐀-䶿]+", "", lowered_artist)
    generic_scene_names = {
        "工作学习时听的音乐提高专注力",
        "工作学习时听的音乐",
        "提高专注力",
        "雨林下雨声自然白噪音工作学习睡觉",
        "高音质白噪音雨水声雷声放松催眠学习睡前音乐工作学习必备",
        "纯音乐馆",
        "休闲音乐",
    }
    if title_compact in generic_scene_names or artist_compact in generic_scene_names:
        return False
    return True
