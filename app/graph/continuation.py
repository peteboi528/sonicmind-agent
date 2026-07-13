"""graph 层跨轮对话、去重、查询改写与状态持久化逻辑。"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from app.graph._shared import _select_listed_tracks, _similar_artists_payload
from app.graph.tag_rules import extract_tags
from app.intents import expand_content_negation, extract_content_negations, get_intent, is_continuation
from app.models import AgentPlan, RetrievalPlan

if TYPE_CHECKING:
    from app.agent import AudioVisualAgent
    from app.graph.state import AgentState

logger = logging.getLogger(__name__)


def _apply_dialogue_continuation(
    plan: AgentPlan,
    query: str,
    dialogue_state: dict[str, Any] | None,
) -> tuple[AgentPlan, str]:
    """延续指令（再来几首/换一批/不要重复）程序化继承上一轮实体与标签，并挂跨轮去重排除集。

    仅当本轮被判为延续、且本轮没抽到自己的实体时才继承实体；返回 (新计划, 继承说明)。
    话题切换（本轮自带新实体/明确意图）不继承实体——交由 finalize 时覆盖旧状态。

    关键：跨轮去重的排除集（_excluded_tracks）与"是否自带实体"解耦——只要本轮是
    延续指令就挂上。否则用户重提歌手名（"The Weeknd 再来几首"或再次搜同名）时，
    rp.entities 非空会让这里整体提前 return，排除集永不挂，去重失效，netease 永远
    返回同一批 top-N。
    """
    if not dialogue_state or not is_continuation(query):
        return plan, ""

    def _mount_excluded(p: AgentPlan) -> AgentPlan:
        prev_shown = dialogue_state.get("shown_tracks") or []
        if prev_shown:
            p._excluded_tracks = prev_shown  # type: ignore[attr-defined]
        prev_artists = dialogue_state.get("shown_artists") or []
        if prev_artists:
            p._excluded_artists = prev_artists  # type: ignore[attr-defined]
        return p

    rp = plan.retrieval_plan
    negative_constraints = extract_content_negations(query)
    current_entities = _without_negative_constraints(rp.entities, negative_constraints)
    if plan.intent == "similar_artists":
        generic_refs = {"同类型", "同风格", "类似", "相似", "同类", "相似歌手", "类似歌手"}
        current_entities = [item for item in current_entities if _constraint_key(item) not in generic_refs]
    # 本轮正向实体代表话题切换；但“不要越南”里被 LLM 抽出的“越南”不是正向实体。
    if current_entities:
        if current_entities != rp.entities:
            plan = plan.model_copy(update={
                "retrieval_plan": rp.model_copy(update={"entities": current_entities}),
            })
        return _mount_excluded(plan), ""

    prev_entities = _without_negative_constraints(dialogue_state.get("entities") or [], negative_constraints)
    prev_intent = dialogue_state.get("last_intent") or plan.intent
    if not prev_entities and prev_intent in {"chat"}:
        return _mount_excluded(plan), ""

    # 继承上一轮意图（除非本轮 LLM 给了更具体的非 chat 意图）。
    intent = plan.intent if plan.intent not in {"chat"} else prev_intent
    spec = get_intent(intent)
    use_web = plan.online_required or spec.online_default
    merged = RetrievalPlan(
        use_local=rp.use_local,
        use_vector=rp.use_vector or bool(dialogue_state.get("mood_tags") or dialogue_state.get("scenario_tags")),
        use_web=use_web,
        entities=prev_entities,
        genre_filter=rp.genre_filter or dialogue_state.get("genre_tags") or [],
        mood_filter=rp.mood_filter or dialogue_state.get("mood_tags") or [],
        scenario_filter=rp.scenario_filter or dialogue_state.get("scenario_tags") or [],
        # LLM 改写为空、仍含否定或丢失前文种子时，程序化合并上一轮正向上下文。
        search_query=_continuation_search_query(rp, dialogue_state, negative_constraints, prev_entities),
        search_variants=_continuation_search_variants(rp.search_variants, negative_constraints),
        language_filter=_continuation_language_filter(rp.language_filter, negative_constraints),
        excluded_terms=list(dict.fromkeys([*rp.excluded_terms, *negative_constraints])),
    )
    new_plan = plan.model_copy(update={
        "intent": intent,
        "strategy": spec.strategy_for(use_web),
        "tools_needed": spec.tools_for(use_web),
        "online_required": use_web,
        "retrieval_plan": merged,
    })
    return _mount_excluded(new_plan), "、".join(prev_entities) or prev_intent


def _constraint_key(value: str) -> str:
    return re.sub(r"[^a-z0-9一-龥]+", "", value.lower())


def _without_negative_constraints(values: list[str], constraints: list[str]) -> list[str]:
    blocked = [
        _constraint_key(alias)
        for item in constraints
        for alias in expand_content_negation(item)
        if _constraint_key(alias)
    ]
    if not blocked:
        return list(values)
    return [
        value for value in values
        if not any(block in _constraint_key(value) or _constraint_key(value) in block for block in blocked)
    ]


def _strip_negative_query(text: str, constraints: list[str]) -> str:
    cleaned = re.sub(
        r"(?:不要|别放|别推|别推荐|不想听|排除|避开|去掉|no\s+|without\s+|exclude\s+|avoid\s+)"
        r"[^，。,.!?！？]*",
        " ", text, flags=re.IGNORECASE,
    )
    for constraint in constraints:
        for alias in sorted(expand_content_negation(constraint), key=len, reverse=True):
            cleaned = re.sub(re.escape(alias), " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?:帮我|给我|推荐|来点|来些|找点|想听|几首|一些)", " ", cleaned)
    cleaned = re.sub(r"(?:歌曲|音乐)", " ", cleaned)
    return " ".join(cleaned.split()).strip()


def _continuation_search_query(
    retrieval: RetrievalPlan,
    dialogue_state: dict[str, Any],
    constraints: list[str],
    inherited_entities: list[str],
) -> str:
    llm_query = _strip_negative_query(retrieval.search_query, constraints)
    inherited = [
        *inherited_entities,
        *(dialogue_state.get("genre_tags") or []),
        *(dialogue_state.get("mood_tags") or []),
        *(dialogue_state.get("scenario_tags") or []),
    ]
    prior_query = _strip_negative_query(str(dialogue_state.get("last_query") or ""), constraints)
    parts = [llm_query, *inherited]
    if prior_query and not llm_query and not any(str(item).strip() for item in inherited):
        parts.append(prior_query)
    deduped: list[str] = []
    seen_keys: list[str] = []
    for part in parts:
        part = _normalize_query_fragment(part)
        key = _constraint_key(part)
        if not part or not key:
            continue
        replaced = False
        for idx, existing_key in enumerate(seen_keys):
            if key == existing_key or key in existing_key:
                replaced = True
                break
            if existing_key in key:
                deduped[idx] = part
                seen_keys[idx] = key
                replaced = True
                break
        if replaced:
            continue
        deduped.append(part)
        seen_keys.append(key)
    return " ".join(deduped).strip() or "音乐"


def _normalize_query_fragment(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    tokens = text.split()
    if len(tokens) >= 2 and len(tokens) % 2 == 0:
        half = len(tokens) // 2
        if [token.lower() for token in tokens[:half]] == [token.lower() for token in tokens[half:]]:
            tokens = tokens[:half]
    compact: list[str] = []
    for token in tokens:
        if (
            compact
            and compact[-1].lower() == token.lower()
            and re.fullmatch(r"[一-龥]{1,8}", token)
        ):
            continue
        compact.append(token)
    return " ".join(compact).strip()


def _continuation_search_variants(variants: list[str], constraints: list[str]) -> list[str]:
    cleaned = [_strip_negative_query(item, constraints) for item in variants]
    return list(dict.fromkeys(item for item in cleaned if item))


def _continuation_language_filter(current: str, constraints: list[str]) -> str:
    if current:
        return current
    normalized = " ".join(constraints).lower()
    if any(token in normalized for token in ("中文", "华语", "国语", "chinese")):
        return "en"
    if any(token in normalized for token in ("英文", "英语", "欧美", "english")):
        return "zh"
    return ""


def _query_with_entities(query: str, plan: AgentPlan) -> str:
    """把 plan 里的实体拼进 query。延续指令（"多来几首"）本身不含实体，
    实体是从上一轮 DialogueState 继承到 retrieval_plan 的；不拼进 query 的话
    recommend/search/playlist 工具拿不到实体，会走情绪/场景路由搜空。

    P1 查询改写：plan 带 LLM 合成的 search_query 时，以它为基底（已融合多轮
    上下文 + 否定转正向），而非原始 query——否则"不要中文歌曲"会原样发给搜索层。
    实体已出现在基底中则不重复拼接（避免 "找 The Weeknd The Weeknd"）。
    """
    base = (getattr(plan.retrieval_plan, "search_query", "") or "").strip() or query
    entities = plan.retrieval_plan.entities
    if not entities:
        return base
    lowered = base.lower()
    extra = [e for e in entities if e and e.lower() not in lowered]
    if not extra:
        return base
    return " ".join([base, *extra]).strip()


def _merge_excluded_tracks(existing: list[dict[str, str]], new_items: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in [*existing, *new_items]:
        title = item.get("title", "").lower().strip()
        source_id = item.get("source_id", "").strip()
        if not title:
            continue
        key = (title, source_id)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _merge_excluded_artists(existing: list[dict[str, str]], new_items: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in [*existing, *new_items]:
        name = item.get("name", "").lower().strip()
        source = item.get("source", "").strip()
        if not name:
            continue
        key = (name, source)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _persist_dialogue_state(agent: AudioVisualAgent, state: AgentState) -> None:
    """把本轮意图/实体/标签写回 DialogueState，供下一轮延续指令继承。

    chat 意图视为话题中断，清空旧状态（下一轮"再来几首"无可继承对象时
    不会错误复用过期实体）。其余意图覆盖保存最新一轮上下文。
    同时记录本轮展示给用户的曲目（shown_tracks），供下一轮去重。

    shown_tracks 的累积规则：本轮是延续指令时，把本轮 shown 追加到前轮累积记录上，
    去重后封顶保存——这样第 N 轮"不要重复"能排除整个会话里已展示过的曲目，而不只是
    上一轮的。非延续（全新话题）则重置为本轮 shown。

    关键：累积只看 is_continuation，不看 rp.entities。因为延续指令会从上一轮"继承"
    实体（"多来几首"继承 The Weeknd），继承后 rp.entities 非空——若用 `not rp.entities`
    做条件，会把"继承实体"误判成"话题切换"而重置，导致前几轮的展示记录被丢、
    第三轮去重又把第一轮的歌捞回来。
    """
    plan = state["plan"]
    user_id = state["user_id"]
    if plan.intent == "chat":
        agent.memory.clear_dialogue_state(user_id)
        return
    rp = plan.retrieval_plan
    # DialogueState must not depend solely on LLM-populated plan fields.  Recover
    # deterministic genre/mood/scenario seeds from the raw turn and preserve prior
    # positive tags on a continuation when one side is incomplete.
    turn_tags = extract_tags(state["query"])
    prior_dialogue = (state.get("context") or {}).get("dialogue_state") or {}

    def merged_tags(current: list[str], derived: list[str], prior: list[str]) -> list[str]:
        values = [*current, *derived]
        if is_continuation(state["query"]):
            values.extend(prior)
        return list(dict.fromkeys(item for item in values if item))

    genre_tags = merged_tags(rp.genre_filter, turn_tags["genre"], prior_dialogue.get("genre_tags") or [])
    mood_tags = merged_tags(rp.mood_filter, turn_tags["mood"], prior_dialogue.get("mood_tags") or [])
    scenario_tags = merged_tags(
        rp.scenario_filter, turn_tags["scenario"], prior_dialogue.get("scenario_tags") or [],
    )
    # 收集本轮最终展示的曲目摘要，用于下一轮去重
    listed = _select_listed_tracks(state.get("results", []), plan)
    shown = [
        {
            "title": getattr(t, "title", ""),
            "artist": getattr(t, "artist", "") or "",
            "source": getattr(t, "source", "local"),
            "source_id": getattr(t, "external_id", "") or getattr(t, "asset_id", ""),
        }
        for t in listed
    ]
    shown_artists = [
        {
            "name": str(artist.get("name") or ""),
            "source": str(artist.get("source") or "local_library"),
        }
        for artist in _similar_artists_payload(state.get("results", []))
        if str(artist.get("name") or "").strip()
    ]
    # 跨轮累积：延续指令时并入前轮记录（继承实体也算同一话题），否则视为新话题重置。
    prev_shown = prior_dialogue.get("shown_tracks") or []
    if is_continuation(state["query"]):
        merged_shown = _merge_excluded_tracks(prev_shown, shown)
    else:
        merged_shown = shown
    merged_shown = merged_shown[:80]  # 封顶，避免长期会话排除集无限增长
    prev_shown_artists = prior_dialogue.get("shown_artists") or []
    if is_continuation(state["query"]):
        merged_shown_artists = _merge_excluded_artists(prev_shown_artists, shown_artists)
    else:
        merged_shown_artists = shown_artists
    merged_shown_artists = merged_shown_artists[:80]
    agent.memory.save_dialogue_state(
        user_id,
        intent=plan.intent,
        query=state["query"],
        entities=rp.entities,
        genre_tags=genre_tags,
        mood_tags=mood_tags,
        scenario_tags=scenario_tags,
        shown_tracks=merged_shown,
        shown_artists=merged_shown_artists,
    )
