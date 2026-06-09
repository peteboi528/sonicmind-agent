"""确定性标签提取规则（对齐 SoulTuner "LLM 判意图、规则抽标签" 的分工）。

LLM 只判意图 + 抽实体；genre/mood/scenario 标签由这里的关键词映射确定性填充，
避免 LLM 凭空编造标签，降低幻觉与 token 成本。词表与项目既有词汇保持一致。
"""

from __future__ import annotations

# genre 关键词 → 标准 genre 名
_GENRE_RULES: dict[str, list[str]] = {
    "流行": ["流行", "pop", "popular"],
    "摇滚": ["摇滚", "rock", "metal", "金属"],
    "电子": ["电子", "electronic", "edm", "techno", "house", "dj"],
    "古典": ["古典", "classical", "钢琴曲", "交响"],
    "爵士": ["爵士", "jazz", "blues", "蓝调"],
    "民谣": ["民谣", "folk", "弹唱", "indie", "独立"],
    "说唱": ["说唱", "rap", "hip-hop", "hiphop", "嘻哈", "trap"],
    "R&B": ["r&b", "rnb", "节奏蓝调", "soul", "灵魂乐"],
    "国风": ["国风", "古风", "中国风", "戏腔"],
}

# mood 关键词 → 标准 mood 名
_MOOD_RULES: dict[str, list[str]] = {
    "放松": ["放松", "chill", "轻松", "舒缓", "lofi", "lo-fi", "慵懒"],
    "治愈": ["治愈", "温暖", "暖", "healing", "疗愈"],
    "欢快": ["欢快", "快乐", "开心", "happy", "活力", "元气"],
    "伤感": ["伤感", "悲伤", "难过", "sad", "失恋", "emo", "孤独"],
    "浪漫": ["浪漫", "romantic", "甜", "告白", "love"],
    "激昂": ["激昂", "热血", "燃", "high", "嗨", "energetic", "激情"],
    "宁静": ["宁静", "安静", "平静", "calm", "冥想", "睡前"],
    "梦幻": ["梦幻", "dreamy", "空灵", "ambient", "氛围"],
}

# scenario 关键词 → 标准 scenario 名
_SCENARIO_RULES: dict[str, list[str]] = {
    "运动": ["运动", "跑步", "健身", "workout", "running", "gym"],
    "学习": ["学习", "工作", "专注", "focus", "study", "coding", "写代码", "效率"],
    "睡眠": ["睡眠", "助眠", "睡前", "入睡", "sleep", "晚安"],
    "开车": ["开车", "驾驶", "公路", "driving", "兜风"],
    "通勤": ["通勤", "地铁", "路上", "commute"],
    "派对": ["派对", "聚会", "party", "蹦迪", "club"],
    "咖啡": ["咖啡", "下午茶", "cafe", "咖啡馆", "惬意"],
}


def _match(text: str, rules: dict[str, list[str]]) -> list[str]:
    lowered = text.lower()
    matched: list[str] = []
    for label, keywords in rules.items():
        if any(kw in lowered for kw in keywords) and label not in matched:
            matched.append(label)
    return matched


def extract_genre(text: str) -> list[str]:
    return _match(text, _GENRE_RULES)


def extract_mood(text: str) -> list[str]:
    return _match(text, _MOOD_RULES)


def extract_scenario(text: str) -> list[str]:
    return _match(text, _SCENARIO_RULES)


def extract_tags(text: str) -> dict[str, list[str]]:
    """一次性抽取三类标签，供 RetrievalPlan 填充。"""
    return {
        "genre": extract_genre(text),
        "mood": extract_mood(text),
        "scenario": extract_scenario(text),
    }
