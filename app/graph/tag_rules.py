"""确定性标签提取规则（对齐 SoulTuner "LLM 判意图、规则抽标签" 的分工）。

LLM 只判意图 + 抽实体；genre/mood/scenario 标签由这里的关键词映射确定性填充，
避免 LLM 凭空编造标签，降低幻觉与 token 成本。词表与项目既有词汇保持一致。
"""

from __future__ import annotations

# genre 关键词 → 标准 genre 名
# 扩充版：覆盖主流子风格关键词，让 tag_rules 能从歌名/歌手/歌单标签中精准提取
_GENRE_RULES: dict[str, list[str]] = {
    "流行": [
        "流行", "pop", "popular", "华语流行", "欧美流行", "c-pop", "k-pop", "j-pop",
        "idol", "偶像", "打榜", "billboard", "tiktok",
    ],
    "摇滚": [
        "摇滚", "rock", "金属", "metal", "punk", "朋克", "grunge", "垃圾摇滚",
        "britpop", "英伦", "british rock", "indie rock", "后朋", "post-punk",
        "new wave", "新浪潮", "硬摇", "hard rock", "另类摇", "alt rock",
        "numetal", "新金属", "death", "black", "thrash", "doom",
        "shoegaze", "盯鞋", "post-rock", "后摇",
    ],
    "电子": [
        "电子", "electronic", "edm", "techno", "house", "dj", "trance",
        "dubstep", "drum", "dnb", "ambient", "氛围电子", "synth", "合成器",
        "electro", "deep house", "progressive", "future bass", "bass",
        "chillwave", "vaporwave", "retrowave", "合成波",
        "idm", "glitch", "lo-fi electronic",
    ],
    "古典": [
        "古典", "classical", "钢琴曲", "交响", "弦乐", "管弦",
        "sonata", "concerto", "symphony", "baroque", "巴洛克",
        "orchestral", "chamber", "室内乐", "奏鸣曲", "协奏曲",
        "莫扎特", "贝多芬", "巴赫", "肖邦", "柴可夫斯基",
    ],
    "爵士": [
        "爵士", "jazz", "blues", "蓝调", "swing", "摇摆",
        "bebop", "fusion", "smooth jazz", "soul jazz",
        "bossa nova", "波萨诺瓦", "latin jazz",
        "萨克斯", "saxophone", "即兴", "improvisation",
    ],
    "民谣": [
        "民谣", "folk", "弹唱", "indie", "独立", "acoustic", "原声",
        "singer-songwriter", "唱作", "吉他弹唱", "乡村民谣",
        "country", "乡村", "蓝草", "bluegrass",
    ],
    "说唱": [
        "说唱", "rap", "hip-hop", "hiphop", "嘻哈", "trap",
        "boom bap", "drill", "grime", "mumble rap",
        "gangsta", "gangster", "old school", "new school",
        "freestyle", "battle", "diss", "beats",
        "flow", "verse", "hook", "bars",
        "underground hip", "conscious hip", "conscious rap",
        "中文说唱", "华语说唱", "中国嘻哈", "c-hiphop",
        "melo rap", "旋律说唱", "emo rap",
    ],
    "R&B": [
        "r&b", "rnb", "r and b", "rhythm and blues", "节奏蓝调",
        "soul", "灵魂乐", "neo-soul", "新灵魂", "neo soul",
        "funk", "放克", "g-funk", "pfunk",
        "motown", "gospel",
        "contemporary r&b", "alternative r&b", "alt r&b",
        "quiet storm", "new jack swing",
        "taiwan r&b", "korean r&b", "k-r&b",
        "urban", "groove", "律动",
    ],
    "国风": [
        "国风", "古风", "中国风", "戏腔", "古韵", "仙侠",
        "民乐", "国乐", "传统", "民族", "琵琶", "古筝", "二胡", "笛子",
        "汉服", "武侠", "江湖",
    ],
    "金属": [
        "金属", "metal", "heavy metal", "黑金", "black metal",
        "死金", "death metal", "碾核", "grindcore",
        "power metal", "力量金属", "speed metal", "速度金属",
        "prog metal", "progressive metal", "前卫金属",
        "gothic metal", "哥特金属", "symphonic metal", "交响金属",
        "folk metal", "民谣金属", "industrial metal", "工业金属",
        "sludge", "doom metal", "厄运金属",
    ],
}

# mood 关键词 → 标准 mood 名
# 扩充版：覆盖更多情绪维度
_MOOD_RULES: dict[str, list[str]] = {
    "放松": [
        "放松", "chill", "轻松", "舒缓", "lofi", "lo-fi", "慵懒",
        "laidback", "laid back", "mellow", "柔", "慢",
        "downtempo", "easy", "惬意", "散漫",
    ],
    "治愈": [
        "治愈", "温暖", "暖", "healing", "疗愈", "温馨",
        "comfort", "comforting", "soothing", "抚慰",
        "阳光", "暖心", "安心",
    ],
    "欢快": [
        "欢快", "快乐", "开心", "happy", "活力", "元气",
        "upbeat", "cheerful", "joy", "joyful",
        "趣味", "俏皮", "playful",
    ],
    "伤感": [
        "伤感", "悲伤", "难过", "sad", "失恋", "emo",
        "孤独", "lonely", "loneliness", "寂寞", "落寞",
        "heartbreak", "心碎", "遗憾", "惋惜", "怀念", "思念",
        "忧郁", "melancholy", "depressed", "抑郁", "消沉",
    ],
    "浪漫": [
        "浪漫", "romantic", "甜", "告白", "love", "恋爱",
        "暧昧", "甜蜜", "温柔", "tender", "affection",
        "情歌", "love song", "情人节",
    ],
    "激昂": [
        "激昂", "热血", "燃", "high", "嗨", "energetic", "激情",
        "powerful", "力量感", "强", "猛烈", "aggressive",
        "epic", "史诗", "壮阔", "震撼",
    ],
    "宁静": [
        "宁静", "安静", "平静", "calm", "冥想", "睡前",
        "peaceful", "serene", "tranquil", "安详",
        "纯净", "清澈", "空寂",
    ],
    "梦幻": [
        "梦幻", "dreamy", "空灵", "ambient", "氛围",
        "ethereal", "仙", "迷幻", "psychedelic", "psyc",
        "幻觉", "超现实", "surreal", "haze", "朦胧",
    ],
    "律动": [
        "律动", "groove", "groovy", "funky", "节奏感",
        "bounce", "swing", "body", "舞动",
        "rhythmic", "syncopation", "切分",
        "bop", "head bop", "nod",
    ],
    "暗黑": [
        "暗黑", "dark", "黑暗", "阴郁", "压抑",
        "gloomy", "sinister", "ominous", "gothic",
        "恐怖", "惊悚", "诡异", "morbid",
        "深渊", "深渊感", "despair", "绝望",
    ],
    "性感": [
        "性感", "sexy", "sensual", "sultry", "诱惑",
        "bedroom", "缠绵", "暧昧感", "intimate",
        "provocative", "摩登",
    ],
    "励志": [
        "励志", "inspiring", "inspiration", "motivation",
        "奋斗", "坚持", "勇气", "勇气", "信念",
        "hustle", "grind", "fight", "never give up",
        "uplifting", "empowering",
    ],
}

# scenario 关键词 → 标准 scenario 名
_SCENARIO_RULES: dict[str, list[str]] = {
    "运动": ["运动", "跑步", "健身", "workout", "running", "gym", "举铁", "有氧"],
    "学习": ["学习", "工作", "专注", "focus", "study", "coding", "写代码", "效率", "考研"],
    "睡眠": ["睡眠", "助眠", "睡前", "入睡", "sleep", "晚安", "白噪音"],
    "开车": ["开车", "驾驶", "公路", "driving", "兜风", "road trip", "自驾"],
    "通勤": ["通勤", "地铁", "路上", "commute", "公交", "上班"],
    "派对": ["派对", "聚会", "party", "蹦迪", "club", "夜店", "饮酒"],
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


# ─── 已知歌手 → 风格映射 ───────────────────────────────────────────────
# 当关键词规则匹配不到时，用歌手名查表兜底。
# 只收录知名度高、风格稳定的艺人，避免误判。
# key 全小写，匹配时也用小写。

_ARTIST_GENRE_HINTS: dict[str, list[str]] = {
    # ── 说唱 / Hip-Hop ──
    "drake": ["说唱", "R&B"],
    "kendrick lamar": ["说唱"],
    "j. cole": ["说唱"],
    "travis scott": ["说唱"],
    "post malone": ["说唱", "流行"],
    "kanye west": ["说唱"],
    "eminem": ["说唱"],
    "jay-z": ["说唱"],
    "nicki minaj": ["说唱"],
    "cardi b": ["说唱"],
    "lil nas x": ["说唱", "流行"],
    "21 savage": ["说唱"],
    "future": ["说唱"],
    "migos": ["说唱"],
    "asap rocky": ["说唱"],
    "tyler the creator": ["说唱"],
    "snoop dogg": ["说唱"],
    "wiz khalifa": ["说唱"],
    "logic": ["说唱"],
    "joji": ["R&B", "电子"],
    # ── 中文说唱 ──
    "宝石gem": ["说唱"],
    "宝石": ["说唱"],
    "gai": ["说唱"],
    "pg one": ["说唱"],
    "vava": ["说唱"],
    "ty.": ["说唱"],
    "谢帝": ["说唱"],
    "那吾克热": ["说唱"],
    "艾热": ["说唱"],
    "jony j": ["说唱"],
    "满舒克": ["说唱"],
    "tizzy t": ["说唱"],
    "布瑞吉": ["说唱"],
    "bridge": ["说唱"],
    "街道办gdc": ["说唱"],
    "街道办": ["说唱"],
    "梨冻紧": ["说唱"],
    "王以太": ["说唱"],
    "ice paper": ["说唱"],
    "沙一汀": ["说唱"],
    "skyler sky": ["说唱"],
    # ── R&B / Soul ──
    "the weeknd": ["R&B", "流行"],
    "sza": ["R&B"],
    "frank ocean": ["R&B"],
    "keshi": ["R&B", "流行"],
    "brent faiyaz": ["R&B"],
    "6lack": ["R&B", "说唱"],
    "partynextdoor": ["R&B"],
    "bryson tiller": ["R&B", "说唱"],
    "daniel caesar": ["R&B"],
    "h.e.r.": ["R&B"],
    "summer walker": ["R&B"],
    "jhené aiko": ["R&B"],
    "alina baraz": ["R&B"],
    "khalid": ["R&B", "流行"],
    "dualipa": ["R&B", "流行"],
    "dua lipa": ["流行", "R&B"],
    "chris brown": ["R&B", "说唱"],
    "usher": ["R&B"],
    "beyoncé": ["R&B", "流行"],
    "rihanna": ["R&B", "流行"],
    "adele": ["流行", "R&B"],
    "tinashe": ["R&B"],
    "tony rich": ["R&B"],
    "dean": ["R&B"],
    "zion.t": ["R&B"],
    "crush": ["R&B"],
    "sunwoo jung-a": ["R&B"],
    "jay park": ["说唱", "R&B"],
    # ── 电子 ──
    "calvin harris": ["电子"],
    "marshmello": ["电子"],
    "martin garrix": ["电子"],
    "avicci": ["电子"],
    "david guetta": ["电子"],
    "skrillex": ["电子"],
    "deadmau5": ["电子"],
    "kygo": ["电子"],
    "zedd": ["电子"],
    "the chainsmokers": ["电子", "流行"],
    "odesza": ["电子"],
    "flume": ["电子"],
    "bonobo": ["电子"],
    "jamie xx": ["电子"],
    # ── 摇滚 ──
    "coldplay": ["摇滚", "流行"],
    "imagine dragons": ["摇滚", "流行"],
    "radiohead": ["摇滚"],
    "arctic monkeys": ["摇滚"],
    "muse": ["摇滚"],
    "foo fighters": ["摇滚"],
    "linkin park": ["摇滚", "金属"],
    "metallica": ["金属"],
    "nirvana": ["摇滚"],
    "oasis": ["摇滚"],
    "the killers": ["摇滚"],
    " PARAMORE": ["摇滚"],
    "paramore": ["摇滚"],
    "green day": ["摇滚"],
    "red hot chili peppers": ["摇滚"],
    "the strokes": ["摇滚"],
    "tame impala": ["摇滚", "电子"],
    "mgmt": ["摇滚", "电子"],
    # ── 流行 ──
    "taylor swift": ["流行"],
    "ed sheeran": ["流行"],
    "billie eilish": ["流行", "电子"],
    "harry styles": ["流行", "摇滚"],
    "ariana grande": ["流行", "R&B"],
    "bruno mars": ["流行", "R&B"],
    "justin bieber": ["流行", "R&B"],
    "selena gomez": ["流行"],
    # 注意："dua lipa" 在第 259 行已映射为 ["流行","R&B"]（信息更全），此处勿重复定义，否则后者静默覆盖。
    "lady gaga": ["流行", "电子"],
    "olivia rodrigo": ["流行", "摇滚"],
    "shawn mendes": ["流行"],
    "sam smith": ["流行", "R&B"],
    "lana del rey": ["流行"],
    "lizzo": ["流行", "R&B"],
    # ── 爵士 ──
    "miles davis": ["爵士"],
    "john coltrane": ["爵士"],
    "bill evans": ["爵士"],
    "norah jones": ["爵士", "流行"],
    "kamasi washington": ["爵士"],
    "robert glasper": ["爵士", "R&B"],
    # ── 民谣/独立 ──
    "bon iver": ["民谣", "电子"],
    "fleet foxes": ["民谣"],
    "iron & wine": ["民谣"],
    "sufjan stevens": ["民谣"],
    "phoebe bridgers": ["民谣", "摇滚"],
    "mitski": ["独立", "摇滚"],
    "the national": ["摇滚"],
}


def extract_genre_from_artist(artist: str) -> list[str]:
    """从歌手名查风格映射表。优先精确匹配，再尝试包含匹配。

    返回风格列表，找不到返回空列表。
    """
    if not artist:
        return []
    lowered = artist.lower().strip()

    # 精确匹配
    if lowered in _ARTIST_GENRE_HINTS:
        return _ARTIST_GENRE_HINTS[lowered]

    # 包含匹配：歌手名可能是 "XX Ft. YY" 或 "XX、YY"
    for hint_artist, genres in _ARTIST_GENRE_HINTS.items():
        if hint_artist in lowered or lowered in hint_artist:
            return genres

    # 拆分多歌手（Ft. / feat. / × / 、/ /）
    import re
    parts = re.split(r'\s*[Ff]t\.?\s*|\s*[Ff]eat\.?\s*|\s*[×xX]\s*|\s*[、/]\s*|\s+feat\s+', lowered)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part in _ARTIST_GENRE_HINTS:
            return _ARTIST_GENRE_HINTS[part]
        for hint_artist, genres in _ARTIST_GENRE_HINTS.items():
            if hint_artist in part or part in hint_artist:
                return genres

    return []
