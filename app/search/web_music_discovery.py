"""LLM 候选生成 + 网易云验证：对齐 SoulTuner 的歌曲发现路径。

SoulTuner 用 Zhipu/Tavily/SearxNG 联网搜索 → LLM 从文本提取歌名 → Netease 验证。
我们没有这些搜索 API，替代方案：LLM 根据品味档案直接生成候选 → Netease 逐首验证。

核心保证：**LLM 只生成候选，不直接出现在最终结果中。每首歌必须通过 Netease 验证。**
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app.llm.structured import extract_json_list
from app.models import ExternalTrack
from app.prompts.candidate_generator import CANDIDATE_GENERATOR_PROMPT
from app.search.verifier import batch_verify

if TYPE_CHECKING:
    from app.llm.protocol import LLMProvider

logger = logging.getLogger(__name__)


def generate_llm_candidates(
    query: str,
    taste_summary: str,
    exclusion_rules: list[str] | None = None,
    library_artists: list[str] | None = None,
    target_count: int = 12,
    llm: LLMProvider | None = None,
) -> list[dict[str, str]]:
    """用 LLM 根据品味档案生成候选歌曲列表。

    返回 [{"title": "Nikes", "artist": "Frank Ocean"}, ...]。
    这些候选尚未验证，需要通过 batch_verify 到网易云验证。
    """
    if llm is None:
        return []

    prompt = CANDIDATE_GENERATOR_PROMPT.format(
        taste_summary=taste_summary or "未知",
        exclusion_rules="、".join(exclusion_rules) if exclusion_rules else "无",
        library_artists="、".join(library_artists[:10]) if library_artists else "暂无",
        query=query,
        target_count=target_count,
    )

    try:
        raw = llm.generate(prompt, temperature=0.7)
    except Exception:
        logger.debug("LLM candidate generation failed", exc_info=True)
        return []

    # 提取 candidates 数组
    candidates_raw = extract_json_list(raw)
    if not candidates_raw:
        # 尝试从 JSON 对象中提取 candidates 字段
        from app.llm.structured import extract_json_dict
        data = extract_json_dict(raw)
        if data and isinstance(data.get("candidates"), list):
            candidates_raw = data["candidates"]

    if not candidates_raw:
        return []

    # 清洗：只保留有 title 的项
    cleaned: list[dict[str, str]] = []
    for item in candidates_raw:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        artist = str(item.get("artist", "")).strip()
        if title:
            cleaned.append({"title": title, "artist": artist})
    return cleaned


def discover_from_llm(
    query: str,
    taste_summary: str,
    exclusion_rules: list[str] | None = None,
    library_artists: list[str] | None = None,
    target_count: int = 12,
    llm: LLMProvider | None = None,
) -> list[ExternalTrack]:
    """完整流程：LLM 生成候选 → 网易云验证 → 返回有真实播放链接的 ExternalTrack。"""
    candidates = generate_llm_candidates(
        query=query,
        taste_summary=taste_summary,
        exclusion_rules=exclusion_rules,
        library_artists=library_artists,
        target_count=target_count,
        llm=llm,
    )
    if not candidates:
        logger.debug("LLM generated 0 candidates for query=%r", query)
        return []

    logger.debug("LLM generated %d candidates, verifying against Netease...", len(candidates))
    verified = batch_verify(candidates, max_verify=min(target_count * 2, 20))
    logger.debug("Verified %d/%d candidates", len(verified), len(candidates))
    return verified
