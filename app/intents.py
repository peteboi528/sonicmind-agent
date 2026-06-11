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
from dataclasses import dataclass, field


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
_INTENT_PRIORITY = ("journey", "import", "playlist", "search", "taste", "recommend", "discuss")

_DISCUSS_KEYWORDS = (
    "牛逼", "怎么样", "评价", "介绍", "背景", "风格是", "什么水平", "好听吗",
    "厉害", "经典", "代表", "值得听", "有什么歌", "有哪些歌", "成名曲",
    "特色", "曲风", "地位", "影响", "如何看", "聊聊",
    "和 ", " vs ", "对比", "谁的", "专辑", "出道", "代表作",
)

INTENT_REGISTRY: dict[str, IntentDef] = {
    "recommend": IntentDef(
        name="recommend",
        summary="用户要推荐音乐，优先获取真实线上候选，再结合记忆排序。",
        prompt_desc="recommend：推荐音乐 / 每日推荐 / 按心情或场景推荐",
        base_tools=("recommend",),
        prepend_web_search=True,
        keyword_signals=("推荐", "适合", "recommend", "chill"),
    ),
    "search": IntentDef(
        name="search",
        summary="用户要找歌，优先搜索真实平台候选，再补充本地库命中。",
        prompt_desc="search：搜索特定歌曲或歌手",
        base_tools=("search",),
        prepend_web_search=True,
        keyword_signals=("搜索", "找歌", "search", "联网", "真实", "最新"),
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
        keyword_signals=("品味", "分析我", "taste"),
    ),
    "import": IntentDef(
        name="import",
        summary="用户要导入网易云歌单作为推荐输入。",
        prompt_desc="import：导入网易云歌单",
        base_tools=("import",),
        prepend_web_search=False,
        keyword_signals=("导入",),
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
_CONTINUATION_SIGNALS = (
    "再来", "再推", "再给", "换一批", "换一首", "换几首", "还要", "还想",
    "多来", "多给", "类似", "差不多", "类似这个", "像这样", "像这个", "继续",
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
]


def is_continuation(query: str) -> bool:
    """判断是否为延续指令（省略实体、依赖上一轮上下文）。

    包括两种情况：
    1. 明确的延续信号（再来/换一批/继续等）
    2. 指代消解（"他的歌"/"只要他的"/"不要其他的"），需要继承前文实体
    仅在输入较短时才认定为延续，避免长查询（自带新实体）误判。
    """
    q = query.strip().lower()
    if len(query.strip()) > 30:
        return False
    # 明确延续信号
    if any(sig in q or sig in query for sig in _CONTINUATION_SIGNALS):
        return True
    # 指代消解信号
    if any(pat.search(query) for pat in _COREFERENCE_PATTERNS):
        return True
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
