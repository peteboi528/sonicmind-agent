"""复合任务检测：判定一条 query 是否需要多步规划（Deep/Agentic 模式）。

单意图（「推荐几首歌」）走 LangGraph 主图；复合多步任务（「导入歌单，然后挑适合
跑步的」）由同一 Runner 拆解后逐个调用 compiled LangGraph 子图。

启发式（保守——宁可漏判走图，也不要把简单查询误判成复合、徒增延迟与 token）：
  1. 显式链式词（然后/之后/接着/and then...）→ 复合
  2. ≥2 个不同动作类别（导入/搜索/推荐/做歌单/分析品味/旅程/找视频）→ 复合

可选的 LLM 分类器作为后续加强（处理「分析曲风比例再…」这类隐性复合）；当前启发式
零依赖、确定性，进 CI 可测。
"""
from __future__ import annotations

import re

# 强链式词：出现即高置信度复合
_STRONG_CHAIN = [
    "然后", "之后", "接着", "随后", "最后", "下一步", "再然后", "并最后",
    "and then", "after that", "then ", "finally", "next,",
]

# 动作类别（同组词互为同义，只计一次）。词用 re.search 匹配，故可含 .* 等正则。
_ACTION_CATEGORIES: list[tuple[str, list[str]]] = [
    ("import", ["导入", "import "]),
    ("search", ["搜索", "搜一下", "帮我搜", "search "]),
    ("recommend", ["推荐", "recommend"]),
    ("playlist", ["做歌单", "生成歌单", "建歌单", "整理歌单", "做个歌单", "做.*歌单", "generate playlist", "make a playlist"]),
    ("taste", ["分析品味", "分析.*品味", "我的品味", "品味分析", "taste profile"]),
    ("journey", ["音乐旅程", "做.*旅程", "个.*旅程", "journey"]),
    ("video", ["找.*mv", "找.*现场", "找.*演唱会", "music video"]),
]


def is_compound_task(query: str) -> bool:
    """判定 query 是否为复合多步任务。保守：歧义时返回 False（走图，更快更省）。"""
    if not query or not query.strip():
        return False
    q = query.lower()
    # 1) 显式链式词
    if any(w in q for w in _STRONG_CHAIN):
        return True
    # 2) ≥2 个不同动作类别
    cats: set[str] = set()
    for cat, words in _ACTION_CATEGORIES:
        if any(re.search(w, q) for w in words):
            cats.add(cat)
    return len(cats) >= 2
