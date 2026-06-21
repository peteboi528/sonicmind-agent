"""统一 Intent 注册表：集中声明每个意图的工具链、执行策略、联网需求、
默认摘要和关键词 fallback 信号。

设计动机：意图相关的逻辑过去散落在 graph/nodes.py（_tools_for_intent /
_strategy_for / _default_summary / _VALID_INTENTS）、prompts/query_plan.py
（硬编码意图清单）和 build_agent_plan（关键词分支）三处，新增一个意图要改
四个地方还容易漏，漏了就触发 Pydantic Literal 500（如历史上的 discuss）。

现在所有意图元数据在此一处声明，其余模块只读 registry：
- AgentPlan.intent 用 field_validator 对照 registry 校验，未知意图降级为 chat
- nodes.plan_with_llm 读 registry 取 tools/strategy/summary
- query_plan prompt 的意图清单由 registry 动态生成
- build_agent_plan 关键词 fallback 遍历 keyword_signals
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class IntentDef:
    name: str
    summary: str                      # 默认 reasoning_summary
    prompt_desc: str                  # query_plan prompt 里这个意图的说明行
    base_tools: tuple[str, ...] = ()  # 该意图始终执行的工具
    prepend_web_search: bool = False  # use_web 时是否在最前面插 web_music_search
    strategy_web: str = "online_first"     # use_web=True 时的策略
    strategy_no_web: str = "library_first"  # use_web=False 时的策略
    online_default: bool = True       # 未显式给 use_web 时的默认联网需求
    keyword_signals: tuple[str, ...] = ()  # 关键词 fallback 命中信号

    def tools_for(self, use_web: bool) -> list[str]:
        prefix = ["web_music_search"] if (use_web and self.prepend_web_search) else []
        return prefix + list(self.base_tools)

    def strategy_for(self, use_web: bool) -> str:
        return self.strategy_web if use_web else self.strategy_no_web


# 关键词 fallback 的匹配优先级（从具体到泛化）：journey/import 要早于
# playlist（"导入歌单""音乐旅程"含子串），discuss 作为音乐知识类兜底放在
# recommend 之后、chat 之前。
_INTENT_PRIORITY = (
    "feedback", "listening_history", "my_playlists", "lyrics", "audio_features",
    "find_on_platform", "concert_events", "journey", "import", "playlist", "video",
    "sample_lookup",
    "music_compare", "review_summary", "album_deep_dive", "artist_deep_dive",
    "search", "taste_experiment", "taste", "artist_albums", "similar_artists", "recommend", "artist_info", "discuss",
)

_DISCUSS_KEYWORDS = (
    "牛逼", "怎么样", "评价", "风格是", "什么水平", "好听吗",
    "厉害", "经典", "代表", "值得听", "有什么歌", "有哪些歌", "成名曲",
    "特色", "曲风", "地位", "影响", "如何看", "聊聊",
    "和 ", " vs ", "对比", "谁的", "谁更", "更牛", "专辑", "出道", "代表作", "区别",
    "同一", "一样", "是不是", "think", "opinion", "feel about",
)

INTENT_REGISTRY: dict[str, IntentDef] = {
    "feedback": IntentDef(
        name="feedback",
        summary="用户在评价刚刚展示或曲库中的真实歌曲，记录结构化反馈并更新排序信号。",
        prompt_desc="feedback：喜欢/不喜欢/跳过/听过某首歌或某位歌手",
        base_tools=("feedback",),
        strategy_web="memory_only", strategy_no_web="memory_only", online_default=False,
        keyword_signals=("不喜欢", "讨厌", "别再放", "这首不错", "喜欢这首", "跳过", "切歌", "听过这首"),
    ),
    "listening_history": IntentDef(
        name="listening_history",
        summary="只读用户听歌历史并做聚合统计。",
        prompt_desc="listening_history：最近听了什么 / 循环最多的歌或歌手",
        base_tools=("listening_history",),
        strategy_web="memory_only", strategy_no_web="memory_only", online_default=False,
        keyword_signals=("听歌历史", "最近在听", "最近听了", "循环最多", "上周听", "本月听"),
    ),
    "my_playlists": IntentDef(
        name="my_playlists",
        summary="读取当前登录用户的网易云歌单列表。",
        prompt_desc="my_playlists：看看我的网易云歌单 / 我的收藏歌单",
        base_tools=("list_my_playlists",),
        prepend_web_search=False,
        keyword_signals=("我的歌单", "我收藏的歌单", "网易云歌单列表", "看看歌单"),
    ),
    "lyrics": IntentDef(
        name="lyrics",
        summary="解析真实 song_id 后获取网易云歌词，失败时不编造。",
        prompt_desc="lyrics：查询某首歌的歌词",
        base_tools=("lyrics",), prepend_web_search=False,
        keyword_signals=("歌词", "lyrics", "唱的什么"),
    ),
    "audio_features": IntentDef(
        name="audio_features",
        summary="读取已有真实 BPM、能量等音频特征，缺失时保持未知。",
        prompt_desc="audio_features：查询歌曲 BPM、调性、能量或舞蹈性",
        base_tools=("audio_features",), prepend_web_search=False, online_default=False,
        keyword_signals=("BPM", "bpm", "调性", "音频特征", "能量值", "danceability"),
    ),
    "find_on_platform": IntentDef(
        name="find_on_platform",
        summary="在用户指定的平台定位真实歌曲或视频。",
        prompt_desc="find_on_platform：把某首歌定位到网易云、YouTube 或 B站",
        base_tools=("find_on_platform",), prepend_web_search=False,
        keyword_signals=("在youtube", "在 youtube", "在b站", "在 b站", "在网易云", "换个平台"),
    ),
    "concert_events": IntentDef(
        name="concert_events",
        summary="查找带网页来源的公开演出或巡演信息。",
        prompt_desc="concert_events：查某歌手近期演出或巡演信息",
        base_tools=("concert_events",), prepend_web_search=False,
        keyword_signals=("巡演信息", "近期演出", "演出日期", "演出安排", "tour dates"),
    ),
    "recommend": IntentDef(
        name="recommend",
        summary="用户要推荐音乐，优先获取真实线上候选，再结合记忆排序。",
        prompt_desc="recommend：推荐音乐 / 每日推荐 / 按心情或场景推荐",
        base_tools=("recommend",),
        prepend_web_search=True,
        keyword_signals=("推荐", "适合", "recommend", "chill", "来点", "来几首", "给我来", "放松", "类似", "像"),
    ),
    "album_deep_dive": IntentDef(
        name="album_deep_dive",
        summary="用户想理解一张专辑，按严格延迟预算生成可引用的专辑档案、乐评共识和聆听路线。",
        prompt_desc="album_deep_dive：解读/讲讲/分析某张专辑，为什么经典，专辑背景或听法",
        base_tools=("resolve_music_entity", "music_metadata_lookup", "review_search", "build_music_dossier"),
        prepend_web_search=False,
        strategy_web="online_first",
        strategy_no_web="online_first",
        online_default=True,
        keyword_signals=("这张专辑", "专辑解读", "解读专辑", "为什么经典", "为什么这么经典", "album deep dive", "分析这张"),
    ),
    "artist_deep_dive": IntentDef(
        name="artist_deep_dive",
        summary="用户想系统理解一位艺人/乐队，生成生涯脉络、代表作品和个性化聆听路线。",
        prompt_desc="artist_deep_dive：系统讲解艺人/乐队、音乐路线、生涯地图、风格变化",
        base_tools=("resolve_music_entity", "music_metadata_lookup", "review_search", "build_music_dossier"),
        prepend_web_search=False,
        strategy_web="online_first",
        strategy_no_web="online_first",
        online_default=True,
        keyword_signals=("音乐路线", "生涯", "风格变化", "系统讲讲", "系统介绍", "艺人档案", "artist deep dive"),
    ),
    "review_summary": IntentDef(
        name="review_summary",
        summary="用户想看乐评/评价聚合，在预算内检索可引用乐评并保守总结。",
        prompt_desc="review_summary：乐评怎么说、专业评价、听众评价、争议点、被高估/低估",
        base_tools=("resolve_music_entity", "music_metadata_lookup", "review_search", "build_music_dossier"),
        prepend_web_search=False,
        strategy_web="online_first",
        strategy_no_web="online_first",
        online_default=True,
        keyword_signals=("乐评", "专业评价", "评价怎么说", "评价如何", "争议点", "被高估", "被低估", "review summary"),
    ),
    "music_compare": IntentDef(
        name="music_compare",
        summary="用户想比较两个音乐实体，在预算内并行查证并给出风格、评价和聆听路径差异。",
        prompt_desc="music_compare：比较两个艺人/专辑/歌曲的区别、相似点和入门路径",
        base_tools=("resolve_music_entity", "music_metadata_lookup", "review_search", "build_music_dossier"),
        prepend_web_search=False,
        strategy_web="online_first",
        strategy_no_web="online_first",
        online_default=True,
        keyword_signals=("区别在哪", "有什么区别", "怎么比", "对比", "比较", " vs ", " VS "),
    ),
    "sample_lookup": IntentDef(
        name="sample_lookup",
        summary="用户想查歌曲采样、插值、翻唱或源曲关系，并把源曲定位成可播放卡片。",
        prompt_desc="sample_lookup：查询歌曲采样/插值/源曲/翻唱来源，并调出源曲播放卡",
        base_tools=("resolve_music_entity", "sample_relation_search", "locate_sample_sources", "build_sample_dossier"),
        prepend_web_search=False,
        strategy_web="online_first",
        strategy_no_web="online_first",
        online_default=True,
        keyword_signals=("采样", "源曲", "sample", "sampled", "interpolation", "插值", "翻唱", "用了哪首", "调出来"),
    ),
    "similar_artists": IntentDef(
        name="similar_artists",
        summary="基于上一轮或当前歌手，在可追溯曲库中寻找风格相近的其他艺人。",
        prompt_desc="similar_artists：推荐与某歌手同类型、同风格或相似的其他歌手",
        base_tools=("similar_artists",),
        prepend_web_search=False,
        strategy_web="library_first",
        strategy_no_web="library_first",
        online_default=False,
        keyword_signals=("同类型的歌手", "同类型歌手", "同风格歌手", "相似歌手", "类似的歌手", "类似歌手"),
    ),
    "artist_albums": IntentDef(
        name="artist_albums",
        summary="用户要某歌手的专辑，直接取网易云真实专辑清单（带 album_id，可整张播放）。",
        prompt_desc="artist_albums：推荐/列举某歌手的专辑、唱片、大碟、discography（直接取真实专辑清单，不走单曲搜索）",
        base_tools=("artist_albums",),
        prepend_web_search=False,  # 不走单曲搜索（避开限流假候选）；专辑端点独立且带缓存
        strategy_web="online_first",
        strategy_no_web="no_search",
        keyword_signals=("专辑", "唱片", "大碟", "discography", "album", "哪几张", "几张专辑"),
    ),
    "taste_experiment": IntentDef(
        name="taste_experiment",
        summary="用户要探索/扩展自己的音乐口味，生成 safe/stretch/bold 三档品味实验并收集反馈。",
        prompt_desc="taste_experiment：探索口味 / 推荐点不一样 / 听腻了 / 发现新风格 / 做品味实验",
        base_tools=("taste_experiment",),
        prepend_web_search=False,
        strategy_web="online_first",
        strategy_no_web="online_first",
        online_default=True,
        keyword_signals=(
            "探索我的口味", "探索口味", "口味实验", "品味实验", "taste lab",
            "推荐点不一样", "推荐一点不一样", "听腻了", "发现新风格",
            "帮我发现新风格", "做个品味实验", "探索边界", "大胆一点",
        ),
    ),
    "search": IntentDef(
        name="search",
        summary="用户要找歌，优先搜索真实平台候选，再补充本地库命中。",
        prompt_desc="search：搜索特定歌曲或歌手",
        base_tools=("search",),
        prepend_web_search=True,
        keyword_signals=("搜索", "找歌", "找一首", "找几首", "帮我找", "搜歌", "搜一下", "找到", "search", "联网", "真实", "最新"),
    ),
    "playlist": IntentDef(
        name="playlist",
        summary="用户要生成歌单，先联网扩展真实候选，再生成可追溯歌单。",
        prompt_desc="playlist：生成歌单 / 播放列表 / 合集",
        base_tools=("playlist",),
        prepend_web_search=True,
        keyword_signals=("歌单", "playlist", "合集"),
    ),
    "taste": IntentDef(
        name="taste",
        summary="用户要分析品味，只读取记忆和行为画像。",
        prompt_desc="taste：分析用户品味、查看偏好档案（只读记忆，无需联网）",
        base_tools=("taste",),
        prepend_web_search=False,
        strategy_web="memory_only",
        strategy_no_web="memory_only",
        online_default=False,
        keyword_signals=("品味", "分析我", "taste", "听听什么"),
    ),
    "import": IntentDef(
        name="import",
        summary="用户要导入网易云歌单作为推荐输入。",
        prompt_desc="import：导入网易云歌单",
        base_tools=("import",),
        prepend_web_search=False,
        keyword_signals=("导入", "import"),
    ),
    "journey": IntentDef(
        name="journey",
        summary="用户需要多阶段音乐编排，使用音乐旅程节点分段检索和解释。",
        prompt_desc='journey：多阶段音乐旅程（如"热身→冲刺→放松"，有明显情绪曲线）',
        base_tools=("journey",),
        prepend_web_search=False,
        keyword_signals=("旅程", "热身", "冲刺", "journey"),
    ),
    "discuss": IntentDef(
        name="discuss",
        summary="音乐讨论，联网搜歌手真实曲目作为讨论论据。",
        prompt_desc="discuss：讨论歌手/乐队风格、专辑背景、音乐评价、创作故事等音乐知识话题（需联网搜真实曲目作论据）",
        base_tools=(),
        prepend_web_search=True,
        strategy_web="online_first",
        strategy_no_web="no_search",
        keyword_signals=_DISCUSS_KEYWORDS,
    ),
    "video": IntentDef(
        name="video",
        summary="用户要找MV/现场/演唱会视频，直接搜索B站/YouTube。",
        prompt_desc="video：找MV/现场版/演唱会/Live视频（直接搜B站/YouTube，不走网易云）",
        base_tools=("video_search",),
        prepend_web_search=False,
        strategy_web="online_first",
        strategy_no_web="no_search",
        keyword_signals=("MV", "mv", "现场", "live", "演唱会", "concert", "视频", "video", "演出", "tour", "Live"),
    ),
    "artist_info": IntentDef(
        name="artist_info",
        summary="用户要了解歌手/乐队信息，用搜索引擎获取真实百科资料。",
        prompt_desc="artist_info：了解歌手/乐队背景资料、成员介绍、出道经历、音乐风格介绍等（用搜索引擎查百科）",
        base_tools=("web_info_search",),
        prepend_web_search=False,
        strategy_web="online_first",
        strategy_no_web="no_search",
        keyword_signals=(
            "介绍", "背景", "成员", "出道", "简介", "资料", "百科", "是谁",
            "什么团", "这个团", "哪个团", "谁啊", "来头", "讲讲", "查查",
            "biography",
        ),
    ),
    "chat": IntentDef(
        name="chat",
        summary="普通对话，不需要检索音乐候选。",
        prompt_desc="chat：普通寒暄或与音乐无关的对话",
        base_tools=(),
        prepend_web_search=False,
        strategy_web="no_search",
        strategy_no_web="no_search",
        online_default=False,
        keyword_signals=(),
    ),
}


def is_valid_intent(intent: str) -> bool:
    return intent in INTENT_REGISTRY


def get_intent(intent: str) -> IntentDef:
    """返回意图定义；未知意图降级为 chat（绝不抛错，避免 Pydantic 500）。"""
    return INTENT_REGISTRY.get(intent, INTENT_REGISTRY["chat"])


def valid_intents() -> set[str]:
    return set(INTENT_REGISTRY)


def match_intent_by_keywords(query: str) -> str | None:
    """关键词 fallback：按优先级遍历 keyword_signals，命中返回意图名，否则 None。"""
    lowered = query.lower()
    for name in _INTENT_PRIORITY:
        signals = INTENT_REGISTRY[name].keyword_signals
        if any(sig in lowered or sig in query for sig in signals):
            return name
    return None


# 延续指令信号：本轮省略实体、依赖上一轮上下文的"接着上文"类输入。
# 注意：跨轮去重的排除集只在 is_continuation 为真时挂载（_apply_dialogue_continuation），
# 所以"不要重复/换新的"这类反重复信号必须在这里出现，否则去重永不触发。
_CONTINUATION_SIGNALS = (
    "再来", "再推", "再给", "换一批", "换一首", "换几首", "还要", "还想", "还有",
    "多来", "多给", "类似", "差不多", "类似这个", "像这样", "像这个", "继续",
    "更多", "再来点", "再来些", "同类型", "同风格", "相似歌手", "类似的歌手",
    # 反重复信号：用户明确表示上一轮给重了，要换没看过的。
    "不要重复", "别重复", "不重复", "重复了", "又重复", "换新的", "来点新的", "来些新的",
    "more", "another", "similar", "same vibe", "keep going",
)

# 指代信号：本轮包含代词或省略实体，需要从前文继承实体。
# 当 query 匹配到这些模式时，应该从 DialogueState.entities 继承上一轮实体。
_COREFERENCE_PATTERNS = [
    re.compile(r"他(的|的歌|的歌)"),
    re.compile(r"她(的|的歌|的歌)"),
    re.compile(r"只要(他|她|的|这个|那个|这些|那些)"),
    re.compile(r"不要(其他|别的|其他的|别的的)"),
    re.compile(r"只要.*的(歌|曲|音乐)"),
    re.compile(r"(只要|仅|就)要(他|她|这个|那个)"),
    re.compile(r"排除(其他|别的|别的歌)"),
    re.compile(r"只要$"),
    # 反重复的否定表达（"不要重复/别一样/不要相同"），substring 信号兜不全时这层兜底。
    re.compile(r"(不|别)(?:要|用|会)?(?:重复|重样|一样|相同)"),
]

_CONTENT_NEGATION = re.compile(
    r"(?:不要|别放|别推|别推荐|不想听|排除|避开|去掉|no\s+|without\s+|exclude\s+|avoid\s+)"
    r"\s*([^，。,.!?！？]{1,24})",
    re.IGNORECASE,
)
_NON_CONTENT_NEGATIONS = {
    "", "了", "啦", "吧", "重复", "重样", "一样", "相同", "其他", "其他的",
    "别的", "别的歌", "这个", "那个",
}

_LANGUAGE_NEGATION_GROUPS: dict[str, tuple[str, ...]] = {
    "中文": ("中文", "华语", "国语", "汉语", "chinese", "mandarin"),
    "英文": ("英文", "英语", "欧美", "english"),
    "日语": ("日语", "日文", "日本语", "日本", "japanese"),
    "韩语": ("韩语", "韩文", "韩国语", "韩国", "korean"),
    "越南": ("越南", "越南语", "越南文", "vietnamese"),
    "粤语": ("粤语", "广东话", "cantonese"),
    "法语": ("法语", "法文", "french"),
    "西班牙语": ("西班牙语", "西语", "西班牙文", "spanish"),
    "泰语": ("泰语", "泰文", "thai"),
}


def normalize_content_negation(value: str) -> str:
    """Normalize language/country aliases while leaving arbitrary content exclusions intact."""
    key = value.strip().lower()
    for canonical, aliases in _LANGUAGE_NEGATION_GROUPS.items():
        if any(alias.lower() == key for alias in aliases):
            return canonical
    return value.strip()


def expand_content_negation(value: str) -> tuple[str, ...]:
    """Return all removable aliases for a normalized exclusion."""
    canonical = normalize_content_negation(value)
    return _LANGUAGE_NEGATION_GROUPS.get(canonical, (value,))


def extract_content_negations(query: str) -> list[str]:
    """抽取“不要越南/别放中文歌”这类依赖前文的内容排除条件。"""
    constraints: list[str] = []
    for match in _CONTENT_NEGATION.finditer(query.strip()):
        value = match.group(1).strip()
        value = re.sub(r"^(?:推荐|播放|放|听)", "", value).strip()
        value = re.sub(r"(?:的)?(?:歌曲|音乐|风格|语种|语言|歌)?[吧呀啊啦了呢]*$", "", value).strip()
        value = re.sub(r"\s+(?:songs?|music|tracks?)$", "", value, flags=re.IGNORECASE).strip()
        value = normalize_content_negation(value)
        if value in _NON_CONTENT_NEGATIONS or not value:
            continue
        if any(token in value for token in ("重复", "重样", "一样", "相同")):
            continue
        constraints.append(value)
    return list(dict.fromkeys(constraints))

# 纯数量延续："我需要12首""再来几首""要多首"——只给数字+量词、省略实体，
# 依赖上一轮的歌手/歌单上下文。必须确认去掉语气/动词/数字量词后基本无实体残留，
# 否则会把"周杰伦12首"（自带新实体）误判成延续。
_COUNT_QUANTIFIER = re.compile(r"\d+\s*首|[几两]\s*首|十\s*[几]?\s*首|多\s*首")
_PURE_COUNT_STRIP = re.compile(
    r"[\s我你他她它要给来多再需想希望看听听下点号、，。!！?？首\d几两十百歌曲个吧的了]"
)


def _is_count_continuation(query: str) -> bool:
    """纯数量请求是否算延续（"我需要12首"）。无实体残留才成立。"""
    q = query.strip()
    if len(q) > 12 or not _COUNT_QUANTIFIER.search(q):
        return False
    residual = _PURE_COUNT_STRIP.sub("", q)
    return len(residual) <= 1


def is_continuation(query: str) -> bool:
    """判断是否为延续指令（省略实体、依赖上一轮上下文）。

    包括两种情况：
    1. 明确的延续信号（再来/换一批/继续等）
    2. 指代消解（"他的歌"/"只要他的"/"不要其他的"），需要继承前文实体
    仅在输入较短时才认定为延续，避免长查询（自带新实体）误判。
    注意：如果 query 包含明确的新意图信号（MV/视频/介绍/搜索等），
    即使含指代也不判为延续——用户是在切换意图。
    """
    q = query.strip().lower()
    if len(query.strip()) > 30:
        return False

    # 检查是否包含明确的新意图信号——有则不是延续，而是意图切换
    _NEW_INTENT_SIGNALS = (
        "MV", "mv", "现场", "live", "演唱会", "concert", "视频", "video", "演出", "tour",
        "介绍", "背景", "成员", "出道", "简介", "百科", "biography", "about",
        "歌单", "playlist", "合集",
        "导入", "import", "旅程",
        "品味", "taste",
        "搜索", "搜", "search",
    )
    if any(sig in q or sig in query for sig in _NEW_INTENT_SIGNALS):
        return False

    # 明确延续信号 / 指代 / 纯数量：哪怕本轮带了英文实体（用户重提歌手名+"再来几首"），
    # 也是延续——否则英文实体守卫会提前 return，去重永不触发。
    has_continuation_signal = (
        any(sig in q or sig in query for sig in _CONTINUATION_SIGNALS)
        or any(pat.search(query) for pat in _COREFERENCE_PATTERNS)
        or _is_count_continuation(query)
        or bool(extract_content_negations(query))
    )
    if has_continuation_signal:
        return True

    # 自带英文实体（看起来像歌手名）且无任何延续信号 → 不是延续，是新意图。
    # 排除延续信号里的英文词（more/another/similar/keep 等）。
    _ENG_CONTINUATION_WORDS = {"more", "another", "similar", "same", "keep", "going", "vibe", "please", "give"}
    eng_words = set(w.lower() for w in re.findall(r"[A-Za-z]{3,}", query))
    if eng_words and not eng_words.issubset(_ENG_CONTINUATION_WORDS):
        return False
    return False


def intent_prompt_block() -> str:
    """生成 query_plan prompt 里的「意图类型」清单（动态，避免与代码漂移）。"""
    n = len(INTENT_REGISTRY)
    lines = [f"意图类型 intent（{_cn_count(n)}选一）："]
    lines.extend(f"- {d.prompt_desc}" for d in INTENT_REGISTRY.values())
    return "\n".join(lines)


_CN_NUM = "零一二三四五六七八九十"


def _cn_count(n: int) -> str:
    return _CN_NUM[n] if 0 <= n <= 10 else str(n)
