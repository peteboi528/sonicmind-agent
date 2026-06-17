"""统一 answer 生成模块：候选收集/去重、已知标题白名单、Answer Guard、
回复模板与歌曲卡片。

Graph（live 路径）与 ReAct（fallback）共用同一套候选收集、去重、来源标签、
数量解析、Answer Guard 与目标进度序列化，避免三处逻辑漂移。本模块为唯一真源；
`app.graph.nodes` 与 `app.react_loop` 从这里导入并按需 re-export，保持旧 import
路径与测试不破裂。
"""
from __future__ import annotations

import re
from typing import Any

from app.models import AgentGoal, TrackRef

# ── 回复模板（集中管理，Graph 与 ReAct fallback 共用） ──────────────────
RESPONSE_TEMPLATES = {
    "no_candidates": "这轮没有拿到可追溯的音乐候选；我不会用未核实歌名硬凑结果。",
    "intro_verified": "我优先采用真实线上候选，先给你这 {n} 首可追溯结果：",
    "intro_fallback": "真实线上候选不足，这轮主要是 fallback/本地候选，共 {n} 首：",
    "shortfall": "\n说明：你要求 {target} 首，但目前可追溯候选只有 {n} 首；我不会用未核实歌曲强行补齐。",
    "opinion_verified": "\n\n我的判断：我会先听《{title}》，因为它来自真实平台结果，可信度比本地 fallback 更高。",
    "opinion_fallback": "\n\n我的判断：这批候选质量一般，属于降级结果；更适合继续换关键词联网补搜。",
    "track_list_header": "📋 可追溯候选：",
    "chat_default": "你好，我在。有什么音乐上的事可以帮你?",
    "discuss_fallback": "抱歉，我暂时无法讨论这个话题。",
}

_VERIFIED_SOURCES = {"netease", "bilibili", "youtube"}


def is_verified_source(source: str) -> bool:
    return source in _VERIFIED_SOURCES


def is_fallback_source(source: str) -> bool:
    return "fallback" in source or source in {"mock", "llm"}


def source_label(source: str) -> str:
    if source == "netease":
        return "网易云真实曲目"
    if source == "bilibili":
        return "B 站真实视频/MV"
    if source == "youtube":
        return "YouTube 真实视频"
    if is_fallback_source(source):
        return f"fallback:{source}"
    return "本地库"


def infer_count(text: str) -> int | None:
    """从用户输入解析请求的数量（"推荐5首"→5）。1~100 截断。"""
    match = re.search(r"(\d{1,3})\s*(?:首|个|tracks?|songs?)?", text, re.IGNORECASE)
    if not match:
        return None
    return max(1, min(int(match.group(1)), 100))


def dedupe_tracks(tracks: list[Any]) -> list[Any]:
    """按 title|artist 跨源去重，丢弃空标题。title/artist 大小写不敏感。"""
    seen: set[str] = set()
    unique: list[Any] = []
    for track in tracks:
        title = getattr(track, "title", "")
        if not title:
            continue
        artist = getattr(track, "artist", "") or ""
        key = f"{title.lower()}|{artist.lower()}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(track)
    return unique


def collect_tracks(results: list[dict[str, Any]]) -> list[Any]:
    """从工具结果里收集所有候选 track 并去重（Graph live 路径用）。

    覆盖 web_music_search / daily_recommend / playlist / search / video_search / import。
    排序策略：verified 在线（netease/bilibili/youtube）优先 → 其他在线 → 本地库。
    避免本地歌淹没在线真实候选。
    """
    verified_online: list[Any] = []
    other_online: list[Any] = []
    local: list[Any] = []
    for result in results:
        t = result.get("type")
        if t == "web_music_search":
            for track in result["tracks"]:
                src = getattr(track, "source", "")
                if src in _VERIFIED_SOURCES:
                    verified_online.append(track)
                else:
                    other_online.append(track)
        elif t == "daily_recommend":
            for item in result["recommendation"].tracks:
                verified_online.append(item.asset)
        elif t == "playlist":
            verified_online.extend(result["playlist"].tracks)
        elif t == "search":
            for track in result["response"].external:
                src = getattr(track, "source", "")
                if src in _VERIFIED_SOURCES:
                    verified_online.append(track)
                else:
                    other_online.append(track)
            local.extend(result["response"].local)
        elif t == "video_search":
            for track in result["tracks"]:
                src = getattr(track, "source", "")
                if src in _VERIFIED_SOURCES:
                    verified_online.append(track)
                else:
                    other_online.append(track)
        elif t == "import_netease_playlist":
            verified_online.extend(result["result"].get("tracks", []))
    # verified 在线优先，其他在线其次，本地兜底
    merged = [*verified_online, *other_online, *local]
    return dedupe_tracks(merged)


def collect_known_titles(results: list[dict[str, Any]]) -> set[str]:
    """Answer Guard 白名单：所有可追溯候选的标题（Graph live 路径用）。

    含专辑名——artist_albums 结果里的专辑名来自网易云回查、可追溯，必须纳入白名单，
    否则 guard 会把答案里的《专辑名》当成幻觉歌名删掉。
    """
    titles = {
        getattr(track, "title", "")
        for track in collect_tracks(results)
        if getattr(track, "title", "")
    }
    for r in results:
        if r.get("type") == "artist_albums":
            for a in r.get("albums") or []:
                name = (a.get("name") or "").strip() if isinstance(a, dict) else ""
                if name:
                    titles.add(name)
        if r.get("type") == "taste_experiment":
            exp = r.get("experiment")
            for segment in getattr(exp, "segments", []) or []:
                for item in getattr(segment, "tracks", []) or []:
                    title = getattr(getattr(item, "track", None), "title", "")
                    if title:
                        titles.add(title)
    return titles


def song_card(
    track: Any,
    reason: str = "",
    score: float | None = None,
    components: dict | None = None,
) -> dict[str, Any]:
    """把一个 track 压成前端歌曲卡片所需的精简字段。"""
    # 确保 playback_url 存在：优先用 track 自带，兜底按 source+id 构造
    playback_url = getattr(track, "playback_url", None) or getattr(track, "source_url", None)
    if not playback_url:
        source = getattr(track, "source", "")
        ext_id = getattr(track, "external_id", "") or getattr(track, "asset_id", "")
        if source == "netease" and ext_id:
            playback_url = f"https://music.163.com/song?id={ext_id}"
    return {
        "title": getattr(track, "title", ""),
        "artist": getattr(track, "artist", "") or "未知",
        "source": getattr(track, "source", "local"),
        "source_id": getattr(track, "external_id", "") or getattr(track, "asset_id", ""),
        "playback_url": playback_url,
        "genre": getattr(track, "genre", []) or [],
        "mood": getattr(track, "mood", []) or [],
        "reason": reason,
        "score": score,
        "components": components or {},
        "candidate_kind": getattr(track, "candidate_kind", "track"),
    }


def track_ref(track: Any, score: float | None = None, components: dict[str, float] | None = None) -> TrackRef:
    """把曲目对象压成结构化推荐结果，供 AgentAnswer / eval 指标复用。"""
    return TrackRef(
        title=getattr(track, "title", ""),
        artist=getattr(track, "artist", "") or "",
        source=getattr(track, "source", "local"),
        source_id=getattr(track, "external_id", "") or getattr(track, "asset_id", ""),
        genre=list(getattr(track, "genre", []) or []),
        mood=list(getattr(track, "mood", []) or []),
        score=score,
        components=components or {},
    )


def track_ref_from_card(card: dict[str, Any]) -> TrackRef:
    """从前端卡片字段回构 TrackRef，保留 score/components。"""
    return TrackRef(
        title=str(card.get("title", "")),
        artist=str(card.get("artist", "") or ""),
        source=str(card.get("source", "local")),
        source_id=str(card.get("source_id", "")),
        genre=list(card.get("genre", []) or []),
        mood=list(card.get("mood", []) or []),
        score=card.get("score"),
        components=dict(card.get("components", {}) or {}),
    )


def guard_answer(answer: str, known_titles: set[str]) -> tuple[str, list[str]]:
    """Answer Guard：扫描答案里 《》 包裹的歌名，剔除白名单之外的幻觉曲目。

    返回 (清洗后的答案, 被移除的幻觉歌名列表)。中文场景下歌名几乎都用
    书名号包裹，这是高可靠、低误伤的程序化信号。
    """
    if not answer:
        return answer, []
    known_norm = {t.strip().lower() for t in known_titles}
    hallucinated: list[str] = []

    def _is_known(name: str) -> bool:
        n = name.strip().lower()
        if not n:
            return True
        if n in known_norm:
            return True
        # 容忍真实标题带副标题/译名等额外信息的包含匹配
        return any(n in kt or kt in n for kt in known_norm)

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        prefix = answer[max(0, match.start() - 8):match.start()]
        if any(token in prefix for token in ["歌单", "专辑", "报告", "列表", "标题"]):
            return match.group(0)
        if _is_known(name):
            return match.group(0)
        hallucinated.append(name)
        return ""  # 直接删除未经核实的歌名

    cleaned = re.sub(r"《([^》]+)》", _replace, answer)

    # 纵深防御：英文/中文引号包裹的疑似歌名也校验（LLM 自由生成文案时
    # 可能不用书名号提歌名，绕过上面的 《》 扫描）。仅当被引内容像歌名
    # （短、不含句末标点）才判定，避免误伤普通引用。
    quoted_src = cleaned

    def _replace_quoted(match: re.Match[str]) -> str:
        name = match.group(1).strip()
        if not name or len(name) > 40 or any(p in name for p in "。！？.!?，,；;\n"):
            return match.group(0)
        prefix = quoted_src[max(0, match.start() - 8):match.start()]
        if any(token in prefix for token in ["歌单", "专辑", "报告", "列表", "标题", "需求", "请求"]):
            return match.group(0)
        if _is_known(name):
            return match.group(0)
        hallucinated.append(name)
        return ""

    cleaned = re.sub(r'[“"\']([^”"\'\n]+)[”"\']', _replace_quoted, quoted_src)
    cleaned = re.sub(r"[、，,]\s*(?=[、，,。；;])", "", cleaned)  # 清理删除后残留的孤立标点
    cleaned = re.sub(r"[^\S\r\n]{2,}", " ", cleaned).strip()
    return cleaned, hallucinated


def goal_progress(goal: AgentGoal | None) -> list[str]:
    """把 AgentGoal 序列化为人类可读的状态行列表。"""
    if goal is None:
        return []
    lines = [f"status={goal.status}", f"goal={goal.goal}"]
    if goal.steps_done:
        lines.append("done=" + "、".join(goal.steps_done))
    if goal.steps_pending:
        lines.append("pending=" + "、".join(goal.steps_pending))
    return lines


# ── ReAct fallback 专用：grounded 答案模板 ──────────────────────────────
def grounded_track_list(query: str, results: list[dict[str, Any]], candidates: list[Any]) -> str:
    """生成可追溯歌曲清单（作为 LLM 自然语言回答的附录）。

    candidates 由调用方传入（fallback 路径用其自己的窄收集器），避免与
    live 路径的 collect_tracks 耦合。无候选时返回空串。
    """
    target = infer_count(query)
    tracks = dedupe_tracks(candidates)
    verified = [t for t in tracks if is_verified_source(getattr(t, "source", ""))]
    fallback = [t for t in tracks if is_fallback_source(getattr(t, "source", ""))]
    local = [t for t in tracks if getattr(t, "source", "local") == "local"]
    selected = [*verified, *fallback, *local]
    if target:
        selected = selected[:target]
    if not selected:
        return ""
    lines = [RESPONSE_TEMPLATES["track_list_header"]]
    for index, track in enumerate(selected[: max(target or 8, 8)], start=1):
        title = getattr(track, "title", "")
        artist = getattr(track, "artist", "") or "未知"
        label = source_label(getattr(track, "source", "local"))
        lines.append(f"{index}. 《{title}》 - {artist}（{label}）")
    return "\n".join(lines)


def grounded_music_answer(query: str, results: list[dict[str, Any]], candidates: list[Any]) -> str:
    target = infer_count(query)
    tracks = dedupe_tracks(candidates)
    verified = [t for t in tracks if is_verified_source(getattr(t, "source", ""))]
    fallback = [t for t in tracks if is_fallback_source(getattr(t, "source", ""))]
    local = [t for t in tracks if getattr(t, "source", "local") == "local"]

    selected = [*verified, *fallback, *local]
    if target:
        selected = selected[:target]

    if not selected:
        return RESPONSE_TEMPLATES["no_candidates"]

    lines = []
    for index, track in enumerate(selected[: max(target or 8, 8)], start=1):
        title = getattr(track, "title", "")
        artist = getattr(track, "artist", "") or "未知"
        label = source_label(getattr(track, "source", "local"))
        lines.append(f"{index}. 《{title}》 - {artist}（{label}）")

    if verified:
        intro = RESPONSE_TEMPLATES["intro_verified"].format(n=len(selected))
    else:
        intro = RESPONSE_TEMPLATES["intro_fallback"].format(n=len(selected))
    if target and len(selected) < target:
        intro += RESPONSE_TEMPLATES["shortfall"].format(target=target, n=len(selected))

    opinion = ""
    if verified:
        opinion = RESPONSE_TEMPLATES["opinion_verified"].format(title=getattr(verified[0], "title", ""))
    elif fallback:
        opinion = RESPONSE_TEMPLATES["opinion_fallback"]

    return intro + "\n" + "\n".join(lines) + opinion
