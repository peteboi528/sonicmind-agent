from __future__ import annotations

import re
from typing import Any

from app.config import settings
from app.models import Asset, ExternalTrack, TasteProfile

# 搜索噪声过滤：中文停用词不作为相关性判据（如"的""歌""我"等）
_QUERY_NOISE = {
    "的",
    "了",
    "在",
    "是",
    "我",
    "你",
    "他",
    "她",
    "它",
    "和",
    "与",
    "或",
    "歌",
    "曲",
    "音乐",
    "首",
    "些",
    "几",
    "个",
    "找",
    "要",
    "想",
    "帮",
    "给",
    "推荐",
    "适合",
    "播放",
    "听",
    "下",
    "不",
    "也",
    "都",
    "就",
    "还",
    "又",
    "几首",
    "一些",
    "几个",
    "来几",
    "来首",
    "我想",
    "帮我",
    "给我",
    "来点",
    "好听",
    "推一",
    "推几",
    "推些",
    "介绍",
    "分享",
    "列举",
    "一下",
    "生成",
    "只要",
    "其他",
    "不要",
    "别的",
    "还有",
    "有没有",
    "可以",
    "能",
    "会",
    "让",
    "从",
    "到",
    "去",
    "来",
    "上",
    "这",
    "那",
    "什么",
    "怎么",
    "哪些",
    "如何",
    "为什么",
    "多少",
    "很多",
    "比较",
    "一点",
    "稍微",
    "偏",
    "微",
    "更",
    "帮我搜索",
    "帮我找",
    "给我推荐",
    "帮我推荐",
    "来几首",
    "弄几首",
    "做",
    "弄",
    "搞",
    "弄个",
    "做个",
    "生成个",
    "songs",
    "song",
    "music",
    "me",
    "some",
    "please",
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "is",
    "to",
    "for",
    "my",
    "and",
    "or",
    "it",
    "s",
    "t",
    "m",
}

_SCENARIO_PLAYLIST_SIGNALS = {
    "跑步",
    "运动",
    "健身",
    "workout",
    "running",
    "通勤",
    "开车",
    "学习",
    "专注",
    "工作",
    "睡眠",
    "助眠",
    "派对",
    "聚会",
    "散步",
    "旅行",
    "约会",
    "泡澡",
    "下午",
    "午后",
    "下午茶",
    "深夜",
    "夜晚",
    "早晨",
    "上午",
    "傍晚",
    "周末",
}

_ARTIST_ALIAS_STOPWORDS = {
    "the",
    "and",
    "band",
    "music",
    "official",
    "feat",
    "featuring",
    "with",
    "from",
    "west",
}


def _extract_search_query(goal: str) -> str:
    cleaned = goal
    for noise in sorted(_QUERY_NOISE, key=len, reverse=True):
        if not noise or not all("一" <= c <= "鿿" or "㐀" <= c <= "䶿" for c in noise):
            continue
        if noise in cleaned:
            cleaned = cleaned.replace(noise, " ")
    for noise in sorted(_QUERY_NOISE, key=len, reverse=True):
        if not noise or not noise.isascii():
            continue
        cleaned = re.sub(r"\b" + re.escape(noise) + r"\b", " ", cleaned, flags=re.IGNORECASE)

    english_tokens = re.findall(r"[A-Za-z][A-Za-z0-9'&\-]*", cleaned)
    english_tokens = [t for t in english_tokens if t.lower() not in _QUERY_NOISE and len(t) > 1]
    cjk_tokens = re.findall(r"[一-鿿㐀-䶿豈-﫿]{2,}", cleaned)
    cjk_tokens = [t for t in cjk_tokens if t not in _QUERY_NOISE]
    candidates = english_tokens + cjk_tokens
    if candidates:
        return " ".join(candidates)
    return goal


def _playlist_search_terms(instruction: str) -> str:
    terms = [instruction]
    lowered = instruction.lower()
    if "chill" in lowered or "lofi" in lowered or any(token in instruction for token in ("下午", "午后", "下午茶")):
        terms.extend(["轻快", "律动", "indie pop", "city pop", "R&B"])
    if "跑步" in instruction or "运动" in instruction:
        terms.extend(["激昂", "热血", "电子", "摇滚"])
    if "工作" in instruction or "专注" in instruction:
        terms.extend(["放松", "宁静", "电子", "爵士"])
    return " ".join(terms)


def _is_scenario_playlist_instruction(instruction: str) -> bool:
    lowered = instruction.lower()
    return any(signal in lowered for signal in _SCENARIO_PLAYLIST_SIGNALS)


def _curated_playlist_query(instruction: str) -> str:
    lowered = instruction.lower()
    if any(token in lowered for token in ("跑步", "运动", "健身", "running", "workout")):
        return "跑步 动感 节奏"
    if any(token in lowered or token in instruction for token in ("下午", "午后", "afternoon")):
        return "午后 chill indie pop"
    if any(token in lowered or token in instruction for token in ("深夜", "夜晚", "凌晨", "late night")):
        return "深夜 chill R&B"
    if any(token in lowered or token in instruction for token in ("早晨", "早上", "清晨", "morning")):
        return "清晨 轻快 indie pop"
    if any(token in lowered for token in ("学习", "专注", "工作")):
        return "学习 专注 工作 纯音乐"
    if any(token in lowered for token in ("睡眠", "助眠", "泡澡")):
        return "睡眠 放松 舒缓"
    if any(token in lowered for token in ("开车", "通勤")):
        return "开车 通勤 节奏"
    if any(token in lowered for token in ("派对", "聚会")):
        return "派对 高能 热门"
    return _extract_search_query(instruction) or instruction


def _scene_playlist_queries(instruction: str) -> list[str]:
    lowered = (instruction or "").lower()
    queries: list[str] = []

    def add_many(values: tuple[str, ...]) -> None:
        for value in values:
            if value and value not in queries:
                queries.append(value)

    if any(token in lowered or token in instruction for token in ("下午", "午后", "下午茶", "afternoon")):
        add_many(
            (
                "午后 chill indie pop",
                "afternoon mellow pop",
                "city pop chill",
                "chill R&B neo soul",
                "轻快 放松 华语流行",
            )
        )
    if any(token in lowered or token in instruction for token in ("深夜", "夜晚", "凌晨", "late night")):
        add_many(
            (
                "深夜 chill R&B",
                "late night neo soul",
                "dream pop 夜晚",
                "ambient pop 深夜",
            )
        )
    if any(token in lowered or token in instruction for token in ("早晨", "早上", "清晨", "morning")):
        add_many(
            (
                "清晨 轻快 indie pop",
                "morning acoustic pop",
                "sunny city pop",
                "起床 清新 流行",
            )
        )
    if any(token in lowered for token in ("chill", "lofi", "lo-fi")):
        add_many(
            (
                "chill R&B",
                "lo-fi hip hop",
                "mellow indie pop",
            )
        )
    return queries[:6]


def _playlist_online_queries(search_terms: str, instruction: str | None = None) -> list[str]:
    queries = [search_terms]
    lowered = search_terms.lower()
    for scene_query in _scene_playlist_queries(instruction or search_terms):
        queries.append(scene_query)
    if "chill" in lowered or "放松" in search_terms:
        queries.extend(
            [
                "chill R&B neo soul",
                "华语 chill 流行",
                "mellow indie pop",
            ]
        )
    if "跑步" in search_terms or "运动" in search_terms:
        queries.extend(
            [
                "跑步 高能 歌曲推荐",
                "运动 电子 摇滚 歌单",
            ]
        )
    unique: list[str] = []
    for query in queries:
        normalized = query.strip()
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique[:4]


def _query_requests_variant_content(query: str) -> bool:
    lowered = (query or "").lower()
    return any(
        token in lowered
        for token in (
            "type beat",
            "free beat",
            "伴奏",
            "instrumental",
            "demo",
            "翻唱",
            "cover",
            "remix",
            "纯音乐",
            "beat版",
            "beat 版",
        )
    )


def _local_ratio_from_query(query: str, default: float) -> float:
    q = (query or "").lower()
    if any(
        k in q
        for k in (
            "不要local",
            "不要本地",
            "不要曲库",
            "不要我库",
            "不要推本地",
            "不要本地歌",
            "不要本地库里",
            "不要我库里",
            "不要库里的",
            "别用我库",
            "别用本地",
            "全要线上",
            "全是线上",
            "只推线上",
            "只要线上",
            "别推本地",
            "no local",
            "no local tracks",
            "without local",
        )
    ):
        return 0.0
    if any(
        k in q
        for k in (
            "减少local",
            "少点local",
            "少一点local",
            "少推本地",
            "少来点本地",
            "少点本地",
            "少一些本地",
            "本地少一点",
            "多用线上",
            "优先线上",
            "线上结果多一点",
            "多来点线上",
            "reduce local",
            "less local",
            "fewer local",
            "prefer online",
        )
    ):
        return 0.15
    return default


def _journey_phases(instruction: str, taste: TasteProfile | None = None) -> list[dict[str, Any]]:
    lowered = instruction.lower()
    taste = taste or TasteProfile()
    genres = [name for name, _ in taste.top_genres[:3]]
    anchor = " ".join(genres[:2])

    def phase(name: str, goal: str, energy: float, intents: list[str], transition: str) -> dict[str, Any]:
        queries = [f"{intent} {anchor}".strip() for intent in intents]
        return {
            "name": name,
            "goal": goal,
            "energy": energy,
            "query": queries[0],
            "queries": queries,
            "transition": transition,
        }

    if "清晨" in instruction and "深夜" in instruction:
        return [
            phase(
                "清晨",
                "温和唤醒",
                0.28,
                ["清晨 治愈 轻快", "晨光 温柔 放松", "早起 清新 节奏"],
                "从低密度、明亮的声音开始。",
            ),
            phase(
                "上午",
                "进入状态",
                0.46,
                ["上午 专注 律动", "工作学习 稳定节奏", "通勤 清醒 groove"],
                "保持清醒感，逐步建立稳定拍点。",
            ),
            phase(
                "午后",
                "维持活力",
                0.64,
                ["午后 律动 活力", "下午 groove 欢快", "白天 节奏 能量"],
                "中段抬高律动与辨识度。",
            ),
            phase(
                "傍晚",
                "释放张力",
                0.76,
                ["傍晚 热血 节奏", "黄昏 高能 律动", "下班 释放 活力"],
                "在日落前达到整条旅程的能量峰值。",
            ),
            phase(
                "深夜",
                "放松收束",
                0.32,
                ["深夜 放松 氛围", "夜晚 慵懒 舒缓", "午夜 安静 治愈"],
                "逐步降低能量与声音密度，安静落幕。",
            ),
        ]
    if any(token in lowered or token in instruction for token in ["跑步", "运动", "running", "workout"]):
        return [
            phase(
                "热身",
                "轻快进入状态",
                0.45,
                ["热身 轻快 节奏", "运动 开场 groove", "慢跑 活力"],
                "从低强度节奏进入身体状态。",
            ),
            phase(
                "推进", "稳定耐力", 0.72, ["跑步 稳定 高能", "运动 律动 节奏", "训练 动感"], "稳定拍点，保持持续推进。"
            ),
            phase(
                "冲刺",
                "高能量峰值",
                0.92,
                ["冲刺 高能 快节奏", "跑步 爆发 热血", "训练 峰值"],
                "把 BPM 和能量推到峰值。",
            ),
            phase("放松", "降速恢复", 0.30, ["运动后 放松 舒缓", "拉伸 治愈", "恢复 安静"], "尾段降低强度，帮助恢复。"),
        ]
    return [
        phase("开场", "建立氛围", 0.35, [f"{instruction} 开场 氛围", "温和 开场", "稳定 情绪"], "先用稳定情绪铺底。"),
        phase(
            "推进",
            "提升记忆点",
            0.68,
            [f"{instruction} 推进 律动", "中段 能量", "节奏 提升"],
            "中段提高辨识度和情绪张力。",
        ),
        phase(
            "收束",
            "留下余韵",
            0.30,
            [f"{instruction} 收束 放松", "结尾 舒缓", "余韵 安静"],
            "最后降低声音密度，留下余韵。",
        ),
    ]


def _format_search_summary(query: str, local: list[Asset], external: list[ExternalTrack], memory_query: str) -> str:
    parts = [f"搜索「{query}」完成：本地 {len(local)} 首，外部候选 {len(external)} 首。"]
    if memory_query:
        parts.append(f"已结合记忆扩展：{memory_query[:80]}。")
    if local:
        parts.append("本地命中：" + "、".join(track.title for track in local[:5]) + "。")
    if external:
        verified = [track for track in external if track.source != "llm"]
        unverified = len(external) - len(verified)
        parts.append("外部候选：" + "、".join(track.title for track in external[:5]) + "。")
        if unverified:
            parts.append(f"其中 {unverified} 首为 LLM 补充候选，尚未真实回查。")
    return " ".join(parts)


def _normalize_match_text(text: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKC", text or "").lower()
    normalized = normalized.replace("r&b", "rnb").replace("r and b", "rnb").replace("hip-hop", "hiphop")
    return re.sub(r"[^a-z0-9一-鿿㐀-䶿]+", "", normalized)


def _artist_credit_parts(artist: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"[、,/;&]|\b(?:feat\.?|featuring|with)\b", artist or "", flags=re.IGNORECASE)
        if part.strip()
    ]


def _artist_alias_keys(artist: str) -> set[str]:
    aliases: set[str] = set()
    for part in _artist_credit_parts(artist):
        full_key = _normalize_match_text(part)
        if full_key:
            aliases.add(full_key)
        cleaned_key = _normalize_match_text(_extract_search_query(part))
        if cleaned_key:
            aliases.add(cleaned_key)
        if re.search(r"[A-Za-z]", part):
            for token in re.findall(r"[A-Za-z0-9]+", part.lower()):
                if len(token) >= 3 and token not in _ARTIST_ALIAS_STOPWORDS:
                    aliases.add(_normalize_match_text(token))
    return aliases


def _string_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    try:
        from rapidfuzz import fuzz

        return float(fuzz.ratio(a, b))
    except Exception:
        from difflib import SequenceMatcher

        return SequenceMatcher(None, a, b).ratio() * 100.0


def _artist_query_matches(query: str, artist: str, *, allow_fuzzy: bool = False) -> bool:
    query_key = _normalize_match_text(_extract_search_query(query) or query)
    if not query_key or query_key in _ARTIST_ALIAS_STOPWORDS:
        return False
    aliases = _artist_alias_keys(artist)
    if query_key in aliases:
        return True
    return bool(
        allow_fuzzy
        and len(query_key) >= 6
        and max((_string_similarity(query_key, alias) for alias in aliases), default=0.0) >= 88.0
    )


def _looks_like_bare_artist_query(raw: str, normalized: str, tags: dict[str, list[str]] | None = None) -> bool:
    if not normalized:
        return False
    tags = tags or {}
    if any(tags.get(key) for key in ("genre", "mood", "scenario")):
        return False

    lowered = normalized.lower().strip()
    if not lowered:
        return False

    blocked_fragments = (
        "歌词",
        "歌单",
        "专辑",
        "歌曲",
        "电台",
        "推荐",
        "播放",
        "找歌",
        "搜歌",
        "album",
        "albums",
        "discography",
        "playlist",
        "lyrics",
        "radio",
        "mix",
        "remix",
        "ost",
        "soundtrack",
        "live",
        "karaoke",
    )
    if any(fragment in lowered for fragment in blocked_fragments):
        return False
    if re.search(r"[0-9《》“”\"'()\[\]{}]", normalized):
        return False

    cjk = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
    if cjk:
        if 2 <= len(cjk) <= 8 and len(normalized.split()) <= 2:
            blocked_cjk = {
                "专注",
                "放松",
                "伤感",
                "治愈",
                "清晨",
                "深夜",
                "通勤",
                "派对",
                "运动",
                "跑步",
                "学习",
                "睡前",
                "睡觉",
                "工作",
                "白噪音",
            }
            common_surnames = {
                "赵",
                "钱",
                "孙",
                "李",
                "周",
                "吴",
                "郑",
                "王",
                "冯",
                "陈",
                "褚",
                "卫",
                "蒋",
                "沈",
                "韩",
                "杨",
                "朱",
                "秦",
                "尤",
                "许",
                "何",
                "吕",
                "施",
                "张",
                "孔",
                "曹",
                "严",
                "华",
                "金",
                "魏",
                "陶",
                "姜",
                "戚",
                "谢",
                "邹",
                "喻",
                "柏",
                "水",
                "窦",
                "章",
                "云",
                "苏",
                "潘",
                "葛",
                "奚",
                "范",
                "彭",
                "郎",
                "鲁",
                "韦",
                "昌",
                "马",
                "苗",
                "凤",
                "花",
                "方",
                "俞",
                "任",
                "袁",
                "柳",
                "唐",
                "罗",
                "薛",
                "伍",
                "余",
                "米",
                "贝",
                "姚",
                "孟",
                "顾",
                "尹",
                "江",
                "钟",
                "黎",
                "龚",
                "邓",
                "侯",
                "邱",
                "邵",
                "蔡",
                "田",
                "樊",
                "胡",
                "凌",
                "霍",
                "虞",
                "万",
                "支",
                "柯",
                "昝",
                "管",
                "卢",
                "莫",
                "经",
                "房",
                "裘",
                "缪",
                "干",
                "解",
                "应",
                "宗",
                "丁",
                "宣",
                "贲",
                "郁",
                "单",
                "杭",
                "洪",
                "包",
                "诸",
                "左",
                "石",
                "崔",
                "吉",
                "钮",
                "程",
                "嵇",
                "邢",
                "滑",
                "裴",
                "陆",
                "荣",
                "翁",
                "荀",
                "羊",
                "於",
                "惠",
                "甄",
                "麹",
                "家",
                "封",
                "芮",
                "羿",
                "储",
                "靳",
                "汲",
                "邴",
                "糜",
                "松",
                "井",
                "段",
                "富",
                "巫",
                "乌",
                "焦",
                "巴",
                "弓",
                "牧",
                "隗",
                "山",
                "谷",
                "车",
                "宓",
                "蓬",
                "全",
                "郗",
                "班",
                "仰",
                "秋",
                "仲",
                "伊",
                "宫",
                "宁",
                "仇",
                "栾",
                "暴",
                "甘",
                "钭",
                "厉",
                "戎",
                "祖",
                "武",
                "符",
                "刘",
                "景",
                "詹",
                "束",
                "龙",
                "叶",
                "幸",
                "司",
                "韶",
                "郜",
                "蓟",
                "薄",
                "印",
                "宿",
                "白",
                "怀",
                "蒲",
                "台",
                "从",
                "鄂",
                "索",
                "咸",
                "籍",
                "赖",
                "卓",
                "蔺",
                "屠",
                "蒙",
                "池",
                "乔",
                "阴",
                "胥",
                "能",
                "苍",
                "双",
                "闻",
                "莘",
                "党",
                "翟",
                "谭",
                "贡",
                "劳",
                "逄",
                "姬",
                "申",
                "扶",
                "堵",
                "冉",
                "宰",
                "郦",
                "雍",
                "郤",
                "璩",
                "桑",
                "桂",
                "濮",
                "牛",
                "寿",
                "通",
                "边",
                "扈",
                "燕",
                "冀",
                "郏",
                "浦",
                "尚",
                "农",
                "温",
                "别",
                "庄",
                "晏",
                "柴",
                "瞿",
                "阎",
                "充",
                "慕",
                "连",
                "茹",
                "习",
                "宦",
                "艾",
                "鱼",
                "容",
                "向",
                "古",
                "易",
                "慎",
                "戈",
                "廖",
                "庾",
                "终",
                "暨",
                "居",
                "衡",
                "步",
                "都",
                "耿",
                "满",
                "弘",
                "匡",
                "国",
                "文",
                "寇",
                "广",
                "禄",
                "阙",
                "东",
                "欧",
                "殳",
                "沃",
                "利",
                "蔚",
                "越",
                "夔",
                "隆",
                "师",
                "巩",
                "厍",
                "聂",
                "晁",
                "勾",
                "敖",
                "融",
                "冷",
                "訾",
                "辛",
                "阚",
                "那",
                "简",
                "饶",
                "空",
                "曾",
                "毋",
                "沙",
                "乜",
                "养",
                "鞠",
                "须",
                "丰",
                "巢",
                "关",
                "蒯",
                "相",
                "查",
                "後",
                "荆",
                "红",
                "游",
                "竺",
                "权",
                "逯",
                "盖",
                "益",
                "桓",
                "公",
                "仉",
                "督",
                "岳",
                "帅",
                "缑",
                "亢",
                "况",
                "后",
                "有",
                "琴",
                "归",
                "海",
                "晋",
                "楚",
                "闫",
                "法",
                "汝",
                "鄢",
                "涂",
                "钦",
                "商",
            }
            if cjk in blocked_cjk:
                return False
            return 2 <= len(cjk) <= 4 and cjk[0] in common_surnames
        return False

    words = [part for part in re.split(r"\s+", normalized.strip()) if part]
    if not words or len(words) > 4:
        return False
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z'.-]*", part) for part in words):
        return False

    if len(words) == 1:
        single = words[0].lower()
        blocked_single = {
            "focus",
            "study",
            "sleep",
            "party",
            "chill",
            "relax",
            "workout",
            "running",
            "romance",
            "sad",
            "happy",
            "healing",
            "morning",
            "night",
        }
        return len(single) >= 5 and single not in blocked_single

    return all(part[0].isupper() or part.isupper() for part in words)


def _fuzzy_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    try:
        from rapidfuzz import fuzz

        return float(fuzz.partial_ratio(a, b))
    except Exception:
        from difflib import SequenceMatcher

        return SequenceMatcher(None, a, b).ratio() * 100.0


def _match_token(token: str, text: str, *, fuzzy: bool = False) -> bool:
    token_norm = _normalize_match_text(token)
    text_norm = _normalize_match_text(text)
    if not token_norm:
        return True
    if token_norm in text_norm:
        return True
    if fuzzy and len(token_norm) >= 3:
        return _fuzzy_ratio(token_norm, text_norm) >= settings.fuzzy_threshold
    return False


def _query_matches_track(query: str, track: ExternalTrack) -> bool:
    def split_tokens(text: str) -> list[str]:
        raw = re.split(r"[\s,，、·\-|/\\]+", text.strip())
        result = []
        for seg in raw:
            sub_tokens = re.findall(r"[A-Za-z]+&[A-Za-z]+|[A-Za-z0-9]+|[一-鿿㐀-䶿豈-﫿]+", seg)
            result.extend(sub_tokens if sub_tokens else [seg])
        return result

    tokens = split_tokens(query)
    tokens = [t for t in tokens if t and t.lower() not in _QUERY_NOISE and len(t) > 1]
    if not tokens:
        return True

    searchable = f"{(track.title or '')} {(track.artist or '')}"
    searchable_parts = [track.title or "", track.artist or "", searchable]
    entity_tokens: list[str] = []
    general_tokens: list[str] = []
    for token in tokens:
        lowered = token.lower()
        is_ascii = bool(re.fullmatch(r"[A-Za-z0-9&]+", token))
        if is_ascii and len(token) > 2:
            entity_tokens.append(lowered)
        else:
            general_tokens.append(token)

    entity_hit = any(_match_token(entity, part, fuzzy=True) for entity in entity_tokens for part in searchable_parts)
    if entity_hit:
        return True
    if not general_tokens:
        return False

    required = max(1, len(general_tokens) - 1)
    hits = 0
    for token in general_tokens:
        token_is_ascii = bool(re.fullmatch(r"[A-Za-z0-9&]+", token))
        if any(_match_token(token, part, fuzzy=token_is_ascii) for part in searchable_parts):
            hits += 1
    return hits >= required
