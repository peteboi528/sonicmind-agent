"""真 ReAct 循环：think -> act -> observe -> reflect -> ... 直到 done 或达到上限。

新版采用 native function calling：
- LLM 通过 tool_calls 主动选择要执行的工具，无需手动 JSON 解析
- 每轮工具结果反馈给 LLM，由 LLM 决定下一步或终止
- 设置 MAX_STEPS 防止失控

向后兼容：
- ActionType 枚举保留（供测试和 fallback 用）
- _think / _keyword_think 保留作为 fallback 路径
- run() 签名不变
"""

from __future__ import annotations

import json
import logging
import re
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from app.llm.protocol import LLMResponse, ToolCall
from app.llm.structured import extract_json_dict
from app.llm.tools import (
    AGENT_TOOLS,
    ALL_TOOL_NAMES,
    TOOL_ANALYZE,
    TOOL_FETCH_METADATA,
    TOOL_IMPORT_NETEASE_PLAYLIST,
    TOOL_MEMORY_UPDATE,
    TOOL_PLAYLIST,
    TOOL_RECOMMEND,
    TOOL_REPORT,
    TOOL_RETRIEVE,
    TOOL_SEARCH,
    TOOL_SIMILAR_CROSS,
    TOOL_SIMILAR_INTRA,
    TOOL_TASTE,
    TOOL_WEB_MUSIC_SEARCH,
)
from app.models import AgentAnswer, AgentGoal, ReActStep
from app.prompts import AGENT_SYSTEM_PROMPT, INTENT_CLASSIFIER_SYSTEM

if TYPE_CHECKING:
    from app.agent import CineSonicAgent


MAX_REACT_STEPS = 5  # ReAct 循环最大轮数（防失控）
logger = logging.getLogger(__name__)


class ActionType(StrEnum):
    """向后兼容：保留枚举，新流程内部不再依赖。"""
    RETRIEVE = "retrieve"
    RECOMMEND = "recommend"
    SEARCH = "search"
    PLAYLIST = "playlist"
    TASTE = "taste"
    SIMILAR_CROSS = "similar_cross"
    SIMILAR_INTRA = "similar_intra"
    ANALYZE = "analyze"
    MEMORY_UPDATE = "memory_update"
    REPORT = "report"
    WEB_MUSIC_SEARCH = "web_music_search"
    FETCH_METADATA = "fetch_metadata"
    IMPORT_NETEASE_PLAYLIST = "import_netease_playlist"


# 关键词 fallback 规则（_think 用）
_INTENT_RULES: list[tuple[list[str], ActionType]] = [
    (["similar video", "similar asset", "类似视频", "相似视频", "like this video"], ActionType.SIMILAR_CROSS),
    (["similar segment", "similar moment", "类似片段", "相似片段"], ActionType.SIMILAR_INTRA),
    (["search", "find songs", "搜索", "找歌", "找一些"], ActionType.SEARCH),
    (["playlist", "歌单", "合集"], ActionType.PLAYLIST),
    (["taste", "品味", "风格分析", "分析我"], ActionType.TASTE),
    (["recommend", "suggest", "推荐", "建议"], ActionType.RECOMMEND),
    (["analyze", "分析", "index", "索引"], ActionType.ANALYZE),
    (["report", "summary", "报告", "摘要", "总结"], ActionType.REPORT),
    (["remember", "preference", "记住", "偏好"], ActionType.MEMORY_UPDATE),
]

_VALID_ACTIONS = {a.value for a in ActionType}

# tool 名 → ActionType 反向映射（仅给 trace 用）
_TOOL_TO_ACTION = {
    TOOL_RECOMMEND: ActionType.RECOMMEND,
    TOOL_SEARCH: ActionType.SEARCH,
    TOOL_PLAYLIST: ActionType.PLAYLIST,
    TOOL_TASTE: ActionType.TASTE,
    TOOL_SIMILAR_CROSS: ActionType.SIMILAR_CROSS,
    TOOL_SIMILAR_INTRA: ActionType.SIMILAR_INTRA,
    TOOL_RETRIEVE: ActionType.RETRIEVE,
    TOOL_ANALYZE: ActionType.ANALYZE,
    TOOL_REPORT: ActionType.REPORT,
    TOOL_MEMORY_UPDATE: ActionType.MEMORY_UPDATE,
    TOOL_WEB_MUSIC_SEARCH: ActionType.WEB_MUSIC_SEARCH,
    TOOL_FETCH_METADATA: ActionType.FETCH_METADATA,
    TOOL_IMPORT_NETEASE_PLAYLIST: ActionType.IMPORT_NETEASE_PLAYLIST,
}


def _matched_keyword_actions(query: str) -> list[ActionType]:
    lowered = query.lower()
    matched: list[ActionType] = []
    for keywords, action in _INTENT_RULES:
        if any(kw in lowered for kw in keywords) and action not in matched:
            matched.append(action)
    return matched


class ReActLoop:
    def __init__(self, agent: CineSonicAgent) -> None:
        self.agent = agent

    def run(
        self,
        user_id: str,
        asset_id: str | None,
        query: str,
        top_k: int = 5,
        history: list[dict[str, Any]] | None = None,
    ) -> AgentAnswer:
        """主入口：真迭代 ReAct 循环。

        LLM 通过 tool_calls 主动决定每一步动作，根据 observation 决定继续或终止。
        失败时降级到旧的一次性 plan 路径。
        """
        goal = self.agent.memory.ensure_goal(user_id, query)
        last_exc: Exception | None = None
        for _ in range(2):
            try:
                return self._tool_calling_loop(user_id, asset_id, query, top_k, history, goal)
            except Exception as exc:
                last_exc = exc
        assert last_exc is not None
        return self._legacy_run(user_id, asset_id, query, top_k, history, last_exc, goal)

    # =============================================================
    # 新流程：真 ReAct + tool calling
    # =============================================================

    def _tool_calling_loop(
        self,
        user_id: str,
        asset_id: str | None,
        query: str,
        top_k: int,
        history: list[dict[str, Any]] | None,
        goal: AgentGoal | None,
    ) -> AgentAnswer:
        steps: list[ReActStep] = []
        results: list[dict[str, Any]] = []
        tokens_total = 0

        # 构造消息：system + history + 当前 user query
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        ]
        if history:
            messages.extend({"role": m["role"], "content": m["content"]} for m in history)
        ctx_parts: list[str] = []
        if asset_id:
            ctx_parts.append(f"当前媒体上下文 asset_id={asset_id}")
        if goal is not None:
            done = "、".join(goal.steps_done) or "暂无"
            pending = "、".join(goal.steps_pending) or "暂无"
            ctx_parts.append(f"当前长期任务：{goal.goal}；已完成：{done}；待完成：{pending}")
        ctx_hint = "（" + "；".join(ctx_parts) + "）\n" if ctx_parts else ""
        messages.append({"role": "user", "content": f"{ctx_hint}{query}"})

        final_answer = ""

        for step_idx in range(MAX_REACT_STEPS):
            response: LLMResponse = self.agent.llm.chat_with_tools(
                messages, AGENT_TOOLS, temperature=0.3
            )
            tokens_total += response.prompt_tokens + response.completion_tokens

            # LLM 返回错误 → 降级到 fallback
            if response.finish_reason == "error":
                raise RuntimeError(f"LLM 错误: {response.error}")

            # LLM 不再调用工具 → 收尾
            if not response.tool_calls:
                final_answer = response.content.strip()
                steps.append(ReActStep(
                    thought=f"第 {step_idx + 1} 步：LLM 判定可以收尾。",
                    action="finalize",
                    observation=f"输出最终答案（{len(final_answer)} 字符）。",
                ))
                break

            # 把 assistant 的 tool_calls 加回 messages（保持对话连续性）
            messages.append({
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                    }
                    for tc in response.tool_calls
                ],
            })
            plan_text = _public_plan_summary(query, response.content, response.tool_calls, results)
            steps.append(ReActStep(
                thought=plan_text,
                action="plan",
                observation="准备调用：" + "、".join(tc.name for tc in response.tool_calls),
            ))

            # 执行每个工具调用
            for tc in response.tool_calls:
                step, result, obs_text = self._execute_tool(
                    tc, user_id, asset_id, query, top_k, step_idx, results
                )
                steps.append(step)
                if result is not None:
                    results.append(result)
                eval_result = _evaluate_progress(query, results)
                if eval_result:
                    steps.append(ReActStep(
                        thought=eval_result,
                        action="eval",
                        observation="结果尚需检查或补充。",
                    ))
                eval_hint = _eval_hint(eval_result, step_idx)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": (obs_text + eval_hint)[:1800],  # 限制 observation 长度防爆 context
                })
            if _playlist_target_satisfied(query, results):
                final_answer = self._compose_from_results(query, results, history)
                steps.append(ReActStep(
                    thought="歌单数量目标已满足，提前收尾。",
                    action="finalize",
                    observation="避免继续重复搜索或超出最大步数。",
                ))
                break

        else:
            steps.append(ReActStep(
                thought=f"已达最大轮数 {MAX_REACT_STEPS}，强制收尾。",
                action="max_steps_reached",
                observation="agent 主动终止迭代。",
            ))

        # 如果 LLM 没给出最终答案（被截断或 max_steps），用结构化整合兜底
        if not final_answer:
            final_answer = self._compose_from_results(query, results, history)
        final_answer = _append_progress_note(final_answer, query, results)

        known_titles = _collect_known_titles(results)
        if _should_use_grounded_music_answer(query, results):
            final_answer = _grounded_music_answer(query, results)

        # Answer Guard：剔除答案里追溯不到任何工具结果的幻觉歌名
        final_answer, hallucinated = guard_answer(final_answer, known_titles)

        auto_memory_updated = self.agent.memory.auto_learn_from_turn(user_id, query, results)
        goal = self.agent.memory.update_goal_progress(user_id, goal, [s.action for s in steps])
        trace = [f"[{s.action}] {s.thought} → {s.observation}" for s in steps]
        if hallucinated:
            trace.append(f"[answer_guard] 已移除 {len(hallucinated)} 个未核实歌名：{'、'.join(hallucinated[:8])}")
        trace.append(f"[meta] tokens={tokens_total}")
        return AgentAnswer(
            answer=final_answer,
            evidences=self._collect_evidences(results)[:8],
            recommended_segments=self._collect_segments(results),
            memory_updated=any(
                r.get("type") == "memory_update" and r.get("changed") for r in results
            ) or auto_memory_updated,
            agent_trace=trace,
            pending_goal=goal.goal if goal and goal.status == "active" else None,
            goal_progress=_goal_progress(goal),
        )

    def _execute_tool(
        self,
        tc: ToolCall,
        user_id: str,
        asset_id: str | None,
        query: str,
        top_k: int,
        step_idx: int,
        prior_results: list[dict[str, Any]] | None = None,
    ) -> tuple[ReActStep, dict[str, Any] | None, str]:
        """执行一次工具调用，返回 (step记录, 结构化结果, 给LLM看的observation文本)。"""
        name = tc.name
        args = tc.arguments
        # 工具不在白名单 → 拒绝执行
        if name not in ALL_TOOL_NAMES:
            return (
                ReActStep(thought=f"未知工具 {name}", action=name, observation="已拒绝"),
                None,
                f"未知工具 {name}，已拒绝执行。",
            )

        try:
            if name == TOOL_RECOMMEND:
                effective_top_k = args.get("top_k", top_k)
                q = args.get("query", query)
                if asset_id:
                    ans = self.agent.recommend_with_memory(asset_id, user_id, q, effective_top_k)
                    obs = f"生成 {len(ans.recommended_segments)} 个片段推荐。"
                    obs_text = ans.answer[:600]
                    return (
                        ReActStep(thought="为当前媒体生成记忆感知推荐。", action="recommend", observation=obs),
                        {"type": "recommend", "answer": ans},
                        obs_text,
                    )
                rec = self.agent.recommend_for_query(user_id, q, top_k=effective_top_k)
                obs = f"生成 {len(rec.tracks)} 首曲目推荐。"
                titles = "; ".join(f"{t.asset.title} - {t.reason}" for t in rec.tracks[:5])
                return (
                    ReActStep(thought="生成个性化音乐推荐。", action="recommend", observation=obs),
                    {"type": "daily_recommend", "recommendation": rec},
                    f"推荐 {len(rec.tracks)} 首：{titles}",
                )

            if name == TOOL_SEARCH:
                q = args.get("query", query)
                include_ext = args.get("include_external", True)
                resp = self.agent.search(user_id, q, include_external=include_ext, top_k=max(top_k, 8))
                obs = f"搜索返回 {len(resp.local)} 首本地、{len(resp.external)} 首外部曲目。"
                preview = ", ".join(
                    [a.title for a in resp.local[:3]] + [t.title for t in resp.external[:3]]
                )
                return (
                    ReActStep(thought="搜索本地库和外部曲库。", action="search", observation=obs),
                    {"type": "search", "response": resp},
                    f"{obs} 示例：{preview}",
                )

            if name == TOOL_PLAYLIST:
                instr = args.get("instruction", query)
                target_count = args.get("target_count") or _infer_requested_count(instr)
                pl = self.agent.generate_playlist(
                    user_id,
                    instr,
                    seed_tracks=_collect_track_candidates(prior_results or []),
                    target_count=target_count,
                )
                obs = f"生成包含 {len(pl.tracks)} 首的歌单《{pl.name}》。"
                return (
                    ReActStep(thought="根据用户意图生成歌单。", action="playlist", observation=obs),
                    {"type": "playlist", "playlist": pl},
                    obs,
                )

            if name == TOOL_TASTE:
                summary = self.agent.summarize_taste(user_id)
                return (
                    ReActStep(thought="总结用户音乐品味。", action="taste", observation="品味摘要已生成。"),
                    {"type": "taste", "summary": summary},
                    summary,
                )

            if name == TOOL_SIMILAR_CROSS and asset_id:
                similar = self.agent.find_similar_assets(asset_id, args.get("top_k", top_k))
                obs = f"找到 {len(similar)} 个相似媒体。"
                return (
                    ReActStep(thought="在库中查找相似视频。", action="similar_cross", observation=obs),
                    {"type": "similar_cross", "results": similar},
                    obs,
                )

            if name == TOOL_SIMILAR_INTRA and asset_id:
                segs = self.agent.media.get_segments(asset_id)
                similar = self.agent.find_similar_segments(asset_id, segs[0].segment_id, args.get("top_k", top_k)) if segs else []
                obs = f"找到 {len(similar)} 个相似片段。" if similar else "暂无可用片段。"
                return (
                    ReActStep(thought="在视频内查找相似片段。", action="similar_intra", observation=obs),
                    {"type": "similar_intra", "results": similar},
                    obs,
                )

            if name == TOOL_RETRIEVE and asset_id:
                q = args.get("query", query)
                evidences = self.agent.retrieve_evidence(asset_id, q, args.get("top_k", top_k))
                obs = f"检索到 {len(evidences)} 个证据片段。"
                return (
                    ReActStep(thought="检索相关证据。", action="retrieve", observation=obs),
                    {"type": "retrieve", "evidences": evidences},
                    obs,
                )

            if name == TOOL_ANALYZE and asset_id:
                asset, segs = self.agent.analyze_media(asset_id)
                obs = f"已分析 {asset.title}：生成 {len(segs)} 个片段。"
                return (
                    ReActStep(thought="执行媒体分析。", action="analyze", observation=obs),
                    {"type": "analyze", "asset": asset, "segments": segs},
                    obs,
                )

            if name == TOOL_REPORT and asset_id:
                report = self.agent.generate_report(asset_id)
                return (
                    ReActStep(thought="生成资产报告。", action="report", observation="报告已生成。"),
                    {"type": "report", "report": report},
                    report.get("summary", "报告已生成"),
                )

            if name == TOOL_MEMORY_UPDATE:
                from app.models import MemoryUpdateRequest
                _, changed = self.agent.update_memory(
                    MemoryUpdateRequest(user_id=user_id, event=args.get("event", query), asset_id=asset_id)
                )
                obs = f"记忆{'已更新' if changed else '无变化'}。"
                return (
                    ReActStep(thought="更新用户记忆。", action="memory_update", observation=obs),
                    {"type": "memory_update", "changed": changed},
                    obs,
                )

            if name == TOOL_WEB_MUSIC_SEARCH:
                q = args.get("query", query)
                tracks = self.agent.search_web_music(q, top_k=args.get("top_k", top_k))
                verified = [t for t in tracks if "fallback" not in t.source and t.source != "llm"]
                fallback = len(tracks) - len(verified)
                if verified and fallback:
                    obs = f"联网搜索返回 {len(verified)} 个真实候选，另用 {fallback} 个离线候选补足。"
                elif verified:
                    obs = f"联网搜索返回 {len(verified)} 个真实候选。"
                else:
                    obs = f"联网未拿到稳定曲目，返回 {fallback} 个离线候选。"
                preview = "; ".join(f"{t.title} - {t.artist} ({t.source})" for t in tracks[:5])
                return (
                    ReActStep(thought="联网搜索真实音乐或视频候选。", action="web_music_search", observation=obs),
                    {"type": "web_music_search", "tracks": tracks},
                    f"{obs} {preview}",
                )

            if name == TOOL_FETCH_METADATA:
                data = self.agent.fetch_track_metadata(
                    asset_id=args.get("asset_id") or asset_id,
                    url=args.get("url"),
                    use_network=args.get("use_network", True),
                )
                obs = "元数据抓取完成。" if data.get("found") else "未抓到可用元数据。"
                return (
                    ReActStep(thought="抓取或补全真实元数据。", action="fetch_metadata", observation=obs),
                    {"type": "fetch_metadata", "metadata": data},
                    json.dumps(data, ensure_ascii=False)[:1200],
                )

            if name == TOOL_IMPORT_NETEASE_PLAYLIST:
                imported = self.agent.import_netease_playlist(
                    args.get("playlist_ref", query),
                    user_id=user_id,
                    limit=args.get("limit", 100),
                )
                obs = f"导入网易云歌单《{imported.get('name', '')}》：新增 {imported.get('imported', 0)} 首，跳过 {imported.get('skipped', 0)} 首。"
                return (
                    ReActStep(thought="导入真实网易云歌单作为后续推荐输入。", action="import_netease_playlist", observation=obs),
                    {"type": "import_netease_playlist", "result": imported},
                    obs,
                )

            # 需要 asset_id 但没有
            obs = f"动作 {name} 跳过（缺少媒体上下文）。"
            return (
                ReActStep(thought="缺少 asset_id，无法执行。", action=name, observation=obs),
                None,
                obs,
            )

        except Exception as exc:
            return (
                ReActStep(thought=f"尝试 {name}。", action=name, observation=f"错误: {exc}"),
                None,
                f"工具 {name} 执行失败: {exc}",
            )

    def _compose_from_results(
        self,
        query: str,
        results: list[dict[str, Any]],
        history: list[dict[str, Any]] | None,
    ) -> str:
        """LLM 没给文本时，用结构化结果拼一个答案（兼容旧 _compose 路径）。"""
        return _legacy_compose(query, results, history, self.agent.llm)

    @staticmethod
    def _collect_evidences(results: list[dict[str, Any]]) -> list[Any]:
        evidences: list[Any] = []
        for r in results:
            if r["type"] == "retrieve":
                evidences.extend(r["evidences"])
            elif r["type"] == "recommend":
                evidences.extend(r["answer"].evidences)
            elif r["type"] == "daily_recommend":
                evidences.extend(r["recommendation"].evidences)
            elif r["type"] == "search":
                evidences.extend(r["response"].evidences)
        return evidences

    @staticmethod
    def _collect_segments(results: list[dict[str, Any]]) -> list[Any]:
        for r in results:
            if r["type"] == "recommend":
                return r["answer"].recommended_segments
        return []

    # =============================================================
    # 向后兼容：旧 _think / _keyword_think / _act / _compose
    # =============================================================

    def _think(
        self, query: str, asset_id: str | None, history: list[dict[str, Any]] | None
    ) -> tuple[list[ActionType], str]:
        """旧入口（test_react.py 直接调用）。新流程不再使用这条路径。"""
        try:
            context = f"当前是否有媒体上下文：{'是' if asset_id else '否'}\n用户输入：{query}"
            if history:
                recent = history[-3:]
                ctx_lines = [f"{m['role']}: {m['content']}" for m in recent]
                context = "近期对话：\n" + "\n".join(ctx_lines) + "\n\n" + context
            result = self.agent.llm.generate(context, system=INTENT_CLASSIFIER_SYSTEM, temperature=0.1)
            data = extract_json_dict(result)
            if data and isinstance(data.get("actions"), list):
                actions = [ActionType(a) for a in data["actions"] if a in _VALID_ACTIONS]
                if actions:
                    for action in _matched_keyword_actions(query):
                        if action not in actions:
                            actions.append(action)
                    return actions, data.get("reason", "LLM 分类")
        except Exception:
            logger.debug("Intent classifier failed; using keyword fallback", exc_info=True)
        return self._keyword_think(query, asset_id), "关键词规则 fallback"

    def _keyword_think(self, query: str, asset_id: str | None) -> list[ActionType]:
        matched = _matched_keyword_actions(query)
        if not matched:
            matched = [ActionType.RETRIEVE, ActionType.RECOMMEND] if asset_id else [ActionType.TASTE, ActionType.RECOMMEND]
        online_intents = {ActionType.RECOMMEND, ActionType.SEARCH, ActionType.PLAYLIST}
        if not asset_id and any(action in online_intents for action in matched) and ActionType.WEB_MUSIC_SEARCH not in matched:
            matched.insert(0, ActionType.WEB_MUSIC_SEARCH)
        return matched

    def _legacy_run(
        self,
        user_id: str,
        asset_id: str | None,
        query: str,
        top_k: int,
        history: list[dict[str, Any]] | None,
        exc: Exception,
        goal: AgentGoal | None = None,
    ) -> AgentAnswer:
        """旧的一次性 plan 流程作为终极兜底。"""
        steps: list[ReActStep] = [ReActStep(
            thought=f"新流程失败 ({exc.__class__.__name__})，降级到关键词规则。",
            action="fallback",
            observation=str(exc)[:200],
        )]
        actions = self._keyword_think(query, asset_id)
        steps.append(ReActStep(
            thought=f"关键词分类: {[a.value for a in actions]}",
            action="plan",
            observation=f"将顺序执行 {len(actions)} 个动作。",
        ))
        results: list[dict[str, Any]] = []
        for action in actions:
            # 用 ToolCall 复用 _execute_tool
            tool_name = _action_to_tool(action)
            if tool_name is None:
                continue
            tc = ToolCall(id=f"legacy_{action.value}", name=tool_name, arguments={"query": query, "top_k": top_k})
            step, result, _ = self._execute_tool(tc, user_id, asset_id, query, top_k, 0, results)
            steps.append(step)
            if result is not None:
                results.append(result)
        answer = _legacy_compose(query, results, history, self.agent.llm)
        answer = _append_progress_note(answer, query, results)
        if _should_use_grounded_music_answer(query, results):
            answer = _grounded_music_answer(query, results)
        answer, hallucinated = guard_answer(answer, _collect_known_titles(results))
        auto_memory_updated = self.agent.memory.auto_learn_from_turn(user_id, query, results)
        goal = self.agent.memory.update_goal_progress(user_id, goal, [s.action for s in steps])
        legacy_trace = [f"[{s.action}] {s.thought} → {s.observation}" for s in steps]
        if hallucinated:
            legacy_trace.append(f"[answer_guard] 已移除 {len(hallucinated)} 个未核实歌名：{'、'.join(hallucinated[:8])}")
        return AgentAnswer(
            answer=answer,
            evidences=self._collect_evidences(results)[:8],
            recommended_segments=self._collect_segments(results),
            memory_updated=any(r.get("type") == "memory_update" and r.get("changed") for r in results) or auto_memory_updated,
            agent_trace=legacy_trace,
            pending_goal=goal.goal if goal and goal.status == "active" else None,
            goal_progress=_goal_progress(goal),
        )


def _public_plan_summary(
    query: str,
    content: str | None,
    tool_calls: list[ToolCall],
    results: list[dict[str, Any]],
) -> str:
    cleaned = (content or "").strip()
    if cleaned:
        cleaned = re.sub(r"</?think[^>]*>", "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned:
            return cleaned[:220]
    tools = [tc.name for tc in tool_calls]
    target = _infer_requested_count(query)
    if TOOL_WEB_MUSIC_SEARCH in tools:
        return "用户需要音乐候选，优先联网获取真实平台结果，再根据数量和质量决定下一步。"
    if TOOL_PLAYLIST in tools:
        found = _candidate_count(results)
        target_text = f"{target} 首" if target else "目标数量"
        return f"准备基于已收集的 {found} 个候选生成歌单，并检查是否满足 {target_text}。"
    if TOOL_RECOMMEND in tools:
        return "准备结合用户记忆、真实候选和推荐模型排序，给出有依据的推荐。"
    if TOOL_SEARCH in tools:
        return "准备补充本地库命中，用于解释用户历史和线上候选之间的关系。"
    return "准备调用最相关工具推进用户目标，并在 observation 后评估是否足够。"


def _evaluate_progress(query: str, results: list[dict[str, Any]]) -> str:
    target = _infer_requested_count(query)
    playlist_count = _playlist_count(results)
    candidate_count = _candidate_count(results)
    verified_online = _verified_online_count(results)
    fallback_count = _fallback_count(results)

    if target and playlist_count is not None and playlist_count < target:
        return f"歌单只有 {playlist_count}/{target} 首，不能声称完成；需要继续补充真实候选或诚实说明不足。"
    if target and playlist_count is None and candidate_count < target:
        return f"候选只有 {candidate_count}/{target} 个，数量不足；优先换关键词继续联网搜索。"
    if candidate_count and verified_online == 0 and fallback_count:
        return f"当前只有 {fallback_count} 个 fallback 候选，真实平台结果不足；最终回答必须标注降级。"
    if candidate_count and verified_online:
        return f"已有 {verified_online} 个真实线上候选，可继续排序、推荐或生成歌单。"
    return "当前结果还需要和用户目标做匹配检查。"


def _eval_hint(eval_result: str, step_idx: int) -> str:
    if step_idx >= MAX_REACT_STEPS - 1:
        return "\n\n评估：已接近最大步数，请不要编造；若不足，请按实际数量诚实收尾。"
    return (
        "\n\n评估："
        f"{eval_result}"
        " 如果信息不足，请换关键词或调用其他工具补充；如果已经足够，请直接最终回答。"
    )


def _append_progress_note(answer: str, query: str, results: list[dict[str, Any]]) -> str:
    target = _infer_requested_count(query)
    if not target:
        return answer
    playlist_count = _playlist_count(results)
    if playlist_count is not None and playlist_count < target:
        note = f"说明：你要求 {target} 首，但目前可追溯候选只生成了 {playlist_count} 首；我不会用未核实歌曲强行补齐。"
        return f"{answer}\n\n{note}" if answer else note
    return answer


def _should_use_grounded_music_answer(query: str, results: list[dict[str, Any]]) -> bool:
    lowered = query.lower()
    music_intent = any(token in lowered for token in ["推荐", "歌单", "搜索", "找歌", "chill", "playlist", "recommend", "song"])
    has_music_results = any(
        result.get("type") in {"web_music_search", "search", "daily_recommend", "playlist"}
        for result in results
    )
    return music_intent and has_music_results


def _grounded_music_answer(query: str, results: list[dict[str, Any]]) -> str:
    target = _infer_requested_count(query)
    tracks = _collect_track_candidates(results)
    tracks = _dedupe_candidates(tracks)
    verified = [track for track in tracks if getattr(track, "source", "") in {"netease", "bilibili", "youtube"}]
    fallback = [
        track for track in tracks
        if "fallback" in getattr(track, "source", "") or getattr(track, "source", "") in {"mock", "llm"}
    ]
    local = [track for track in tracks if getattr(track, "source", "local") == "local"]

    selected = [*verified, *fallback, *local]
    if target:
        selected = selected[:target]

    if not selected:
        return "这轮没有拿到可追溯的音乐候选；我不会用未核实歌名硬凑结果。"

    lines = []
    for index, track in enumerate(selected[: max(target or 8, 8)], start=1):
        title = getattr(track, "title", "")
        artist = getattr(track, "artist", "") or "未知"
        source = getattr(track, "source", "local")
        source_label = _source_label(source)
        lines.append(f"{index}. 《{title}》 - {artist}（{source_label}）")

    if verified:
        intro = f"我优先采用真实线上候选，先给你这 {len(selected)} 首可追溯结果："
    else:
        intro = f"真实线上候选不足，这轮主要是 fallback/本地候选，共 {len(selected)} 首："
    if target and len(selected) < target:
        intro += f"\n说明：你要求 {target} 首，但目前可追溯候选只有 {len(selected)} 首；我不会用未核实歌曲强行补齐。"

    opinion = ""
    if verified:
        first = verified[0]
        opinion = f"\n\n我的判断：我会先听《{getattr(first, 'title', '')}》，因为它来自真实平台结果，可信度比本地 fallback 更高。"
    elif fallback:
        opinion = "\n\n我的判断：这批候选质量一般，属于降级结果；更适合继续换关键词联网补搜。"

    return intro + "\n" + "\n".join(lines) + opinion


def _source_label(source: str) -> str:
    if source == "netease":
        return "网易云真实曲目"
    if source == "bilibili":
        return "B 站真实视频/MV"
    if source == "youtube":
        return "YouTube 真实视频"
    if "fallback" in source or source in {"mock", "llm"}:
        return f"fallback:{source}"
    return "本地库"


def _dedupe_candidates(tracks: list[Any]) -> list[Any]:
    seen: set[str] = set()
    unique: list[Any] = []
    for track in tracks:
        title = getattr(track, "title", "")
        artist = getattr(track, "artist", "")
        source = getattr(track, "source", "local")
        key = f"{source}|{title.lower()}|{artist.lower()}"
        if not title or key in seen:
            continue
        seen.add(key)
        unique.append(track)
    return unique


def _candidate_count(results: list[dict[str, Any]]) -> int:
    return len(_collect_track_candidates(results))


def _playlist_count(results: list[dict[str, Any]]) -> int | None:
    for result in reversed(results):
        if result.get("type") == "playlist":
            return len(result["playlist"].tracks)
    return None


def _verified_online_count(results: list[dict[str, Any]]) -> int:
    count = 0
    for track in _collect_track_candidates(results):
        source = getattr(track, "source", "local")
        if source in {"netease", "bilibili", "youtube"}:
            count += 1
    return count


def _fallback_count(results: list[dict[str, Any]]) -> int:
    count = 0
    for track in _collect_track_candidates(results):
        source = getattr(track, "source", "local")
        if "fallback" in source or source in {"mock", "llm"}:
            count += 1
    return count


def _action_to_tool(action: ActionType) -> str | None:
    for tool, act in _TOOL_TO_ACTION.items():
        if act == action:
            return tool
    return None


def _legacy_compose(
    query: str,
    results: list[dict[str, Any]],
    history: list[dict[str, Any]] | None,
    llm: Any,
) -> str:
    """旧的结构化答案拼接（fallback / LLM 未给文本时用）。"""
    answer_parts: list[str] = []
    for r in results:
        t = r["type"]
        if t == "retrieve":
            answer_parts.append(f"找到 {len(r['evidences'])} 个相关证据片段。")
        elif t == "recommend":
            answer_parts.append(r["answer"].answer)
        elif t == "daily_recommend":
            rec = r["recommendation"]
            lines = [f"{i}. {tr.asset.title} — {tr.reason}" for i, tr in enumerate(rec.tracks[:5], 1)]
            answer_parts.append("根据你的记忆、品味档案和历史库，我推荐：\n" + "\n".join(lines))
            if rec.reason_summary:
                answer_parts.append("推荐依据：" + rec.reason_summary)
        elif t == "search":
            resp = r["response"]
            answer_parts.append(resp.summary)
            if loc := [a.title for a in resp.local[:3]]:
                answer_parts.append("本地命中：" + "、".join(loc))
            if ext := [tr.title for tr in resp.external[:3]]:
                answer_parts.append("外部补充：" + "、".join(ext))
        elif t == "playlist":
            pl = r["playlist"]
            preview = [tr.title for tr in pl.tracks[:5]]
            answer_parts.append(f"已生成歌单《{pl.name}》：{pl.description or '围绕你的指令整理。'}")
            if preview:
                answer_parts.append("前几首：" + "、".join(preview))
        elif t == "taste":
            answer_parts.append(r["summary"])
        elif t == "similar_cross":
            items = r["results"]
            if items:
                lines = [f"- {it.title}（相似度: {it.score}）" for it in items[:5]]
                answer_parts.append("相似视频：\n" + "\n".join(lines))
            else:
                answer_parts.append("库中暂无相似视频。")
        elif t == "similar_intra":
            items = r["results"]
            if items:
                lines = [f"- {it.segment.timestamp}（相似度: {it.score}）" for it in items[:5]]
                answer_parts.append("相似片段：\n" + "\n".join(lines))
        elif t == "report":
            answer_parts.append(f"报告：{r['report'].get('summary', '')}")
        elif t == "memory_update":
            answer_parts.append("已记录你的偏好。")
        elif t == "web_music_search":
            tracks = r["tracks"]
            if tracks:
                lines = [f"{i}. {tr.title} - {tr.artist}（{tr.source}）" for i, tr in enumerate(tracks[:5], 1)]
                answer_parts.append("联网候选：\n" + "\n".join(lines))
            else:
                answer_parts.append("联网搜索没有拿到稳定候选，已保留离线结果作为兜底。")
        elif t == "fetch_metadata":
            meta = r["metadata"]
            answer_parts.append(f"元数据：{meta.get('title') or meta.get('asset_id') or '暂无明确标题'}")
        elif t == "import_netease_playlist":
            result = r["result"]
            answer_parts.append(
                f"已导入歌单《{result.get('name', '')}》，新增 {result.get('imported', 0)} 首，跳过 {result.get('skipped', 0)} 首。"
            )

    if history and answer_parts:
        base = "\n\n".join(answer_parts)
        try:
            polished = llm.chat(
                list(history) + [
                    {"role": "assistant", "content": base},
                    {"role": "user", "content": "请基于上述对话上下文，用自然、友好的语气重新整合这个回复，保留所有关键信息，不超过200字。"},
                ],
                temperature=0.5,
            )
            if polished:
                return polished
        except Exception:
            logger.debug("LLM polish step failed; returning structured fallback", exc_info=True)
        return base

    return "\n\n".join(answer_parts) if answer_parts else f"已处理你的请求：{query}"


def _infer_requested_count(text: str) -> int | None:
    match = re.search(r"(\d{1,3})\s*(?:首|个|tracks?|songs?)?", text, re.IGNORECASE)
    if not match:
        return None
    value = int(match.group(1))
    return max(1, min(value, 100))


def _collect_track_candidates(results: list[dict[str, Any]]) -> list[Any]:
    tracks: list[Any] = []
    for result in results:
        result_type = result.get("type")
        if result_type == "search":
            response = result["response"]
            tracks.extend(response.external)
            tracks.extend(response.local)
        elif result_type == "web_music_search":
            tracks.extend(result["tracks"])
        elif result_type == "daily_recommend":
            tracks.extend(item.asset for item in result["recommendation"].tracks)
    return tracks


def _playlist_target_satisfied(query: str, results: list[dict[str, Any]]) -> bool:
    target = _infer_requested_count(query)
    if target is None:
        return False
    for result in reversed(results):
        if result.get("type") == "playlist":
            return len(result["playlist"].tracks) >= target
    return False


def _collect_known_titles(results: list[dict[str, Any]]) -> set[str]:
    """从工具结果里收集所有"可追溯"的真实曲目标题，构成 Answer Guard 白名单。

    任何在最终答案里出现、却不在这个集合内的歌名，都判定为幻觉。
    """
    titles: set[str] = set()

    def _add(title: str | None) -> None:
        if title and title.strip():
            titles.add(title.strip())

    for r in results:
        t = r.get("type")
        if t == "search":
            resp = r["response"]
            for a in resp.local:
                _add(a.title)
            for tr in resp.external:
                if tr.source != "llm":
                    _add(tr.title)
        elif t == "web_music_search":
            for tr in r["tracks"]:
                if tr.source != "llm":
                    _add(tr.title)
        elif t == "daily_recommend":
            for item in r["recommendation"].tracks:
                _add(getattr(item.asset, "title", None))
        elif t == "recommend":
            for seg in r["answer"].recommended_segments:
                _add(getattr(seg, "scene_summary", None))
        elif t == "playlist":
            for tr in r["playlist"].tracks:
                if getattr(tr, "source", "local") != "llm":
                    _add(tr.title)
        elif t == "fetch_metadata":
            _add(r["metadata"].get("title"))
        elif t == "import_netease_playlist":
            for tr in r["result"].get("tracks", []):
                _add(getattr(tr, "title", None))
    return titles


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
    cleaned = re.sub(r"[、，,]\s*(?=[、，,。；;])", "", cleaned)  # 清理删除后残留的孤立标点
    cleaned = re.sub(r"[^\S\r\n]{2,}", " ", cleaned).strip()
    return cleaned, hallucinated


def _goal_progress(goal: AgentGoal | None) -> list[str]:
    if goal is None:
        return []
    lines = [f"status={goal.status}", f"goal={goal.goal}"]
    if goal.steps_done:
        lines.append("done=" + "、".join(goal.steps_done))
    if goal.steps_pending:
        lines.append("pending=" + "、".join(goal.steps_pending))
    return lines
