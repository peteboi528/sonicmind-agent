from __future__ import annotations

import re
from dataclasses import dataclass

from app.compound import _STRONG_CHAIN, is_compound_task
from app.intents import match_intent_by_keywords
from app.llm.observability import capture_llm_stats, empty_runtime_metrics
from app.llm.routing import select_llm
from app.llm.structured import extract_json_dict
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


async def decompose_compound_async(
    agent,
    query: str,
    history: list[dict[str, str]] | None = None,
) -> tuple[list[SubTask], dict[str, str], dict[str, float | int]]:
    """Async decomposition used by the sole production LangGraph path."""
    if not is_compound_task(query):
        return [_make_subtask(query, index=0)], {}, empty_runtime_metrics()
    llm = select_llm(agent, "fast")
    history_text = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in (history or [])[-6:])
    try:
        raw = await llm.agenerate(
            DECOMPOSE_USER(query, history_text),
            system=DECOMPOSE_SYSTEM,
            temperature=0.1,
        )
        metrics = capture_llm_stats(llm)
        data = extract_json_dict(raw)
        items = data.get("subtasks") if isinstance(data, dict) else None
        tasks = _parse_subtasks(items)
        if tasks:
            return tasks, {"decompose": DECOMPOSE_VERSION}, metrics
    except Exception:
        metrics = capture_llm_stats(llm)
    parts = [_clean_segment(part) for part in _SEGMENT_SPLIT_RE.split(query) if _clean_segment(part)]
    if len(parts) < 2:
        fallback = _split_by_action_markers(query)
        parts = fallback if len(fallback) >= 2 else [query]
    return [_make_subtask(part, index=i) for i, part in enumerate(parts)], {}, metrics


def _parse_subtasks(items) -> list[SubTask]:
    if not isinstance(items, list):
        return []
    tasks: list[SubTask] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        task_query = _clean_segment(str(item.get("query", "")))
        if not task_query:
            continue
        tasks.append(
            SubTask(
                intent=str(item.get("intent") or match_intent_by_keywords(task_query) or "recommend"),
                query=task_query,
                depends_on_prev=bool(item.get("depends_on_prev")) if index > 0 else False,
            )
        )
    return tasks


def summarize_subtasks(subtasks: list[SubTask]) -> str:
    if not subtasks:
        return "按单任务处理。"
    labels = [f"{idx}. {task.intent}: {task.query}" for idx, task in enumerate(subtasks, start=1)]
    return "复合任务拆解为：\n" + "\n".join(labels)


def _make_subtask(segment: str, index: int) -> SubTask:
    intent = match_intent_by_keywords(segment) or "recommend"
    depends = index > 0 and any(token in segment for token in _DEPENDENCY_HINTS)
    return SubTask(intent=intent, query=segment, depends_on_prev=depends)


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
    cleaned = segment.strip(" ，,。；; \n\t")  # noqa: B005 - 刻意按字符集剥离空白/标点
    for token in _STRONG_CHAIN:
        if cleaned.lower().startswith(token.lower()):
            cleaned = cleaned[len(token) :].strip(" ，,。；; \n\t")  # noqa: B005
    return cleaned
