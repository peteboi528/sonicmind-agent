"""graph 层子模块共享的小工具函数。

这些函数被多个域模块（planning/continuation/execution/recovery/budget/finalize）
共同使用，为避免循环依赖而抽到本模块。本模块不依赖任何其他 graph 子模块。
"""
from __future__ import annotations

from typing import Any

from app.answer import collect_tracks as _collect_tracks
from app.models import AgentPlan


def _is_knowledge_intent(intent: str) -> bool:
    """判断意图是否属于知识链路（music_dossier / compare / fact_check 等）。"""
    try:
        from app.knowledge import is_knowledge_intent

        return is_knowledge_intent(intent)
    except Exception:
        return False


def _merge_prompt_versions(existing: Any, incoming: dict[str, str] | None) -> dict[str, str]:
    merged = dict(existing or {})
    for key, value in (incoming or {}).items():
        if value:
            merged[key] = value
    return merged


def _format_prompt_versions(versions: dict[str, str]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(versions.items()))


def _select_listed_tracks(results: list[dict[str, Any]], plan: AgentPlan) -> list[Any]:
    """返回答案文本实际会列出的那批曲目（与 compose_answer 的截断逻辑严格一致）。

    仅对会渲染确定性曲目清单的意图返回非空（recommend/search/playlist/journey）；
    chat/discuss/taste 的文本不是一行行歌名，返回 [] 表示不接管卡片，
    让前端保留流式预览卡片。

    多意图：primary 可能是知识类（返 []），但某个 sub_plan 是 track 类——此时按
    track 型 sub_plan 取卡片，让答案底部仍出 song cards。primary 本身是 track 类时
    走 primary（sub_plan 是知识 dossier，不产 track 卡片）。
    """
    tracks = _select_listed_tracks_single(results, plan)
    if tracks or not plan.is_multi_intent:
        return tracks
    for sp in plan.sub_plans:
        sub_tracks = _select_listed_tracks_single(results, sp)
        if sub_tracks:
            return sub_tracks
    return []


def _select_listed_tracks_single(results: list[dict[str, Any]], plan: AgentPlan) -> list[Any]:
    if plan.intent == "playlist":
        pl = next((r["playlist"] for r in results if r.get("type") == "playlist"), None)
        tracks = list(pl.tracks) if pl and pl.tracks else _collect_tracks(results)
        return tracks[: plan.target_count or 30]
    if plan.intent == "recommend":
        recommendation = next(
            (result.get("recommendation") for result in results if result.get("type") == "daily_recommend"),
            None,
        )
        tracks = [item.asset for item in recommendation.tracks] if recommendation else []
        return (tracks or _collect_tracks(results))[: plan.target_count or 12]
    if plan.intent == "search":
        response = next(
            (result.get("response") for result in results if result.get("type") == "search"),
            None,
        )
        tracks = [*(response.external if response else []), *(response.local if response else [])]
        return (tracks or _collect_tracks(results))[: plan.target_count or 12]
    if plan.intent == "journey":
        tracks = _collect_tracks(results)
        # 旅程的阶段数决定自然长度。未显式指定首数时必须保留全部阶段，不能套用
        # 普通推荐的 12 张默认上限，否则 final 事件会把流式阶段的完整卡片截掉。
        return tracks[:plan.target_count] if plan.target_count else tracks
    if plan.intent == "import":
        return _collect_tracks(results)[: plan.target_count or 12]
    if plan.intent == "video":
        tracks = _collect_tracks(results)
        if plan.target_count:
            tracks = tracks[: plan.target_count]
        return tracks[: plan.target_count or 12]
    return []


def _similar_artists_payload(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for result in results:
        if result.get("type") == "similar_artists":
            return list(result.get("artists") or [])
    return []
