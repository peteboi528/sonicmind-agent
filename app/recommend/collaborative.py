"""P2-H：item-item 协同过滤（可选第四锚）。

思想（轻量版，无矩阵分解）：同一个用户在收听历史里一起听过的曲目互相"投票"，
形成 item→item 的共现计数。给某用户推荐时，候选曲目按它与该用户**近期听过**
曲目的共现强度打分，归一到 [0,1]。

与三锚的关系：
- 语义锚看「内容像不像」，个性化锚看「标签合不合口味」，行为锚看「自己听得爽不爽」，
  协同锚看「听相似曲目的人还听什么」——是唯一引入跨用户信号的锚。
- 冷启动（无跨用户共现、或用户无近期收听）：返回全 0 且 available=False，
  由 rerank 的权重重分配机制把它的权重让给其余锚，行为与三锚时代完全一致。

数据来源：所有用户的 UserMemory.listening_history（按 asset_id 共现）。
归一化：候选的原始共现分 / 本批候选里的最大共现分，避免绝对量纲漂移。
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


def build_cooccurrence(histories: list[list[str]]) -> dict[str, dict[str, int]]:
    """从多用户收听序列构建 item-item 共现计数。

    histories：每个用户一条 asset_id 列表（去重后视为"该用户听过的集合"）。
    返回 {item: {co_item: count}}，对称矩阵，不含自共现。
    """
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for history in histories:
        items = sorted(set(h for h in history if h))  # 去重 + 定序，保证确定性
        for i, a in enumerate(items):
            for b in items[i + 1 :]:
                matrix[a][b] += 1
                matrix[b][a] += 1
    return {k: dict(v) for k, v in matrix.items()}


def collaborative_scores(
    candidate_ids: list[str],
    recent_item_ids: list[str],
    cooccurrence: dict[str, dict[str, int]],
) -> tuple[list[float], bool]:
    """候选按与用户近期听曲的共现强度打分，归一到 [0,1]。

    候选 c 的原始分 = Σ_{r∈recent} cooccurrence[c][r]。
    归一化按本批候选最大原始分；全 0（冷启动/无交集）时返回全 0 + available=False。
    返回 (scores, available)，与 _behavior_anchor 同形，便于 rerank 统一处理。
    """
    n = len(candidate_ids)
    if not candidate_ids or not recent_item_ids or not cooccurrence:
        return [0.0] * n, False
    recent = set(r for r in recent_item_ids if r)
    raw: list[float] = []
    for cid in candidate_ids:
        if not cid:
            raw.append(0.0)
            continue
        neighbors = cooccurrence.get(cid, {})
        raw.append(float(sum(neighbors.get(r, 0) for r in recent)))
    peak = max(raw)
    if peak <= 0:
        return [0.0] * n, False
    return [r / peak for r in raw], True


def recent_listened_ids(listening_history: list[Any], limit: int = 30) -> list[str]:
    """取用户最近 limit 条收听的 asset_id（去重、保最近优先顺序）。"""
    seen: set[str] = set()
    out: list[str] = []
    for event in reversed(listening_history or []):
        aid = getattr(event, "asset_id", "") or ""
        if aid and aid not in seen:
            seen.add(aid)
            out.append(aid)
        if len(out) >= limit:
            break
    return out
