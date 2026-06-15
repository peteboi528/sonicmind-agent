from __future__ import annotations

import re
from dataclasses import dataclass

from app.llm.observability import capture_llm_stats, empty_runtime_metrics
from app.llm.routing import select_llm
from app.llm.structured import extract_json_dict
from app.compound import _STRONG_CHAIN, is_compound_task
from app.intents import match_intent_by_keywords
from app.prompts import DECOMPOSE_SYSTEM, DECOMPOSE_USER, DECOMPOSE_VERSION


@dataclass(frozen=True)
class SubTask:
    intent: str
    query: str
    depends_on_prev: bool = False


_SEGMENT_SPLIT_RE = re.compile(
    r"(?:然后|之后|接着|随后|最后|下一步|再然后|并最后|and then|after that|then\b|finally|next,|；|;|，再|, then|, and then)",
    re.IGNORECASE,
)
_DEPENDENCY_HINTS = ("他", "她", "它", "这个", "那个", "这些", "那些", "类似", "基于", "继续", "再", "上一步", "前面")


def decompose_compound(agent, query: str, history: list[dict[str, str]] | None = None) -> list[SubTask]:
    tasks, _, _ = decompose_compound_with_meta(agent, query, history)
    return tasks


def decompose_compound_with_meta(
    agent,
    query: str,
    history: list[dict[str, str]] | None = None,
) -> tuple[list[SubTask], dict[str, str], dict[str, float | int]]:
    """把复合 query 拆成有序子任务；优先走结构化输出，失败时退化为启发式。"""
    if not is_compound_task(query):
        return [_make_subtask(query, index=0)], {}, empty_runtime_metrics()

    structured = _decompose_with_llm(agent, query, history)
    if structured[0]:
        return structured[0], {"decompose": DECOMPOSE_VERSION}, structured[1]

    parts = [_clean_segment(part) for part in _SEGMENT_SPLIT_RE.split(query) if _clean_segment(part)]
    if len(parts) < 2:
        fallback = _split_by_action_markers(query)
        parts = fallback if len(fallback) >= 2 else [query]

    tasks = [_make_subtask(part, index=i) for i, part in enumerate(parts)]
    return tasks or [_make_subtask(query, index=0)], {}, empty_runtime_metrics()


def summarize_subtasks(subtasks: list[SubTask]) -> str:
    if not subtasks:
        return "按单任务处理。"
    labels = [f"{idx}. {task.intent}: {task.query}" for idx, task in enumerate(subtasks, start=1)]
    return "复合任务拆解为：\n" + "\n".join(labels)


def _make_subtask(segment: str, index: int) -> SubTask:
    intent = match_intent_by_keywords(segment) or "recommend"
    depends = index > 0 and any(token in segment for token in _DEPENDENCY_HINTS)
    return SubTask(intent=intent, query=segment, depends_on_prev=depends)


def _decompose_with_llm(
    agent,
    query: str,
    history: list[dict[str, str]] | None,
) -> tuple[list[SubTask], dict[str, float | int]]:
    llm = select_llm(agent, "fast")
    if llm is None:
        return [], empty_runtime_metrics()
    history_text = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in (history or [])[-6:])
    prompt = DECOMPOSE_USER(query, history_text)
    try:
        raw = llm.generate(prompt, system=DECOMPOSE_SYSTEM, temperature=0.1)
        metrics = capture_llm_stats(llm)
        data = extract_json_dict(raw)
        items = data.get("subtasks") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return [], metrics
        tasks: list[SubTask] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            task_query = _clean_segment(str(item.get("query", "")))
            if not task_query:
                continue
            intent = str(item.get("intent") or match_intent_by_keywords(task_query) or "recommend")
            depends = bool(item.get("depends_on_prev")) if index > 0 else False
            tasks.append(SubTask(intent=intent, query=task_query, depends_on_prev=depends))
        return tasks, metrics
    except Exception:
        return [], capture_llm_stats(llm)


def _split_by_action_markers(query: str) -> list[str]:
    markers = ["导入", "搜索", "搜一下", "推荐", "生成歌单", "做个歌单", "分析", "介绍", "找"]
    lowered = query.lower()
    hits: list[int] = []
    for marker in markers:
        pos = lowered.find(marker.lower())
        if pos > 0:
            hits.append(pos)
    cuts = sorted(set(hits))
    if not cuts:
        return [query]
    parts: list[str] = []
    start = 0
    for cut in cuts:
        if cut <= start:
            continue
        parts.append(query[start:cut])
        start = cut
    parts.append(query[start:])
    return [_clean_segment(part) for part in parts if _clean_segment(part)]


def _clean_segment(segment: str) -> str:
    cleaned = segment.strip(" ，,。；; \n\t")
    for token in _STRONG_CHAIN:
        if cleaned.lower().startswith(token.lower()):
            cleaned = cleaned[len(token):].strip(" ，,。；; \n\t")
    return cleaned
