from __future__ import annotations

import asyncio
import random
import re
import uuid
from typing import Any

from app.intents import get_intent, is_continuation, match_intent_by_keywords
from app.llm.protocol import LLMResponse, ToolCall
from app.llm.structured import extract_json_dict
from app.llm.tools import (
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

REASON_TEMPLATES = [
    "因为你喜欢{genre}风格，这首歌的{mood}氛围很适合{time}听。",
    "根据你最近的收听习惯，这首{genre}曲目的节奏和能量感很匹配你的偏好。",
    "这首歌融合了{mood}的情绪和{genre}的元素，和你的品味非常契合。",
    "你之前喜欢的音乐中有类似的{mood}氛围，推荐你试试这首。",
    "这是一首{genre}风格的佳作，{mood}的基调很适合现在的时段。",
    "基于你的品味档案，这首歌在旋律和情绪上都很对味。",
]

SEARCH_TEMPLATES = [
    "为你找到了{count}首相关内容，其中包含{genre}风格和{mood}氛围的作品。",
    "根据你的搜索，以下是最匹配的{count}首推荐。",
]

DAILY_SUMMARY_TEMPLATES = [
    "今天为你精选了{count}首歌曲，涵盖{genres}等风格，希望能陪伴你度过美好的一天。",
    "根据你的品味和当前时段，今日推荐以{mood}为主基调，共{count}首精选。",
]

CHAT_TEMPLATES = [
    "好的，让我根据你的偏好来推荐。你喜欢{genre}风格，我找到了一些很适合的内容。",
    "明白了！根据你的收听历史，我推荐以下几首{mood}氛围的音乐。",
]


# 工具决策规则（仅用于 MockLLM 第一轮）
_TOOL_RULES: list[tuple[list[str], str]] = [
    (["网易云歌单", "netease playlist", "playlist?id", "导入歌单"], TOOL_IMPORT_NETEASE_PLAYLIST),
    (["联网", "真实", "最新", "网易云", "bilibili", "b站", "线上"], TOOL_WEB_MUSIC_SEARCH),
    (["metadata", "元数据", "标题", "补全信息"], TOOL_FETCH_METADATA),
    (["similar video", "similar asset", "类似视频", "相似视频"], TOOL_SIMILAR_CROSS),
    (["similar segment", "类似片段", "相似片段", "similar moment"], TOOL_SIMILAR_INTRA),
    (["search", "find songs", "搜索", "找歌"], TOOL_SEARCH),
    (["playlist", "歌单", "合集"], TOOL_PLAYLIST),
    (["taste", "品味", "风格分析", "分析我"], TOOL_TASTE),
    (["recommend", "suggest", "推荐", "建议", "适合", "挑", "跑步"], TOOL_RECOMMEND),
    (["analyze", "分析素材", "index"], TOOL_ANALYZE),
    (["report", "摘要", "报告"], TOOL_REPORT),
    (["remember", "preference", "记住", "我喜欢"], TOOL_MEMORY_UPDATE),
    (["what happens", "时间点", "minute mark", "片段"], TOOL_RETRIEVE),
]


class MockLLM:
    async def agenerate(self, prompt, system=None, temperature=0.7, thinking=None):
        await asyncio.sleep(0)
        return self.generate(prompt, system=system, temperature=temperature, thinking=thinking)

    async def agenerate_stream(self, prompt, system=None, temperature=0.7, thinking=None):
        await asyncio.sleep(0)
        for piece in self.generate_stream(prompt, system=system, temperature=temperature, thinking=thinking):
            yield piece

    def __init__(self) -> None:
        self._tool_call_history: dict[str, list[str]] = {}
        self.last_stats: dict[str, float | int | str] = {}

    def _mark_call(self, kind: str) -> None:
        self.last_stats = {
            "llm_calls": 1,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0.0,
            "estimated_cost_usd": 0.0,
            "provider": f"mock:{kind}",
        }

    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.7, thinking: bool | None = None) -> str:
        self._mark_call("generate")
        return self._dispatch(prompt, system)

    def _dispatch(self, prompt: str, system: str | None) -> str:
        # 同时检查 system，让意图分类器在 mock 下也能命中（system 含"意图"）
        haystack = f"{system or ''}\n{prompt}".lower()
        if "复合任务拆解器" in (system or ""):
            return self._decompose(prompt)
        if "复合任务综合器" in (system or ""):
            return self._compound_synthesis(prompt)
        if "意图规划器修复器" in (system or ""):
            return self._query_plan_repair(prompt)
        if "意图规划器" in (system or "") or "query_plan" in haystack:
            return self._query_plan(prompt)
        if "推荐理由" in prompt or "reason" in haystack:
            return self._reason(prompt)
        if "搜索" in prompt or "search" in haystack:
            return self._search_summary(prompt)
        if "每日" in prompt or "daily" in haystack:
            return self._daily_summary(prompt)
        if "意图" in haystack or "intent" in haystack or "actiontype" in haystack:
            return self._intent(prompt)
        return self._chat(prompt)

    def generate_stream(self, prompt: str, system: str | None = None, temperature: float = 0.7, thinking: bool | None = None) -> Any:
        """Mock 流式：先算出整段，再按块 yield（mock 无真实网络，模拟首字即可）。

        注意不走 generate()，避免重复 _mark_call；直接复用 _dispatch 同一份派发逻辑，
        保证流式与非流式产出的文本一致。
        """
        self._mark_call("generate_stream")
        text = self._dispatch(prompt, system)
        step = max(1, len(text) // 8)
        for i in range(0, len(text), step):
            yield text[i:i + step]

    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.7, thinking: bool | None = None) -> str:
        self._mark_call("chat")
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        return self.generate(last_user)

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.3,
        tool_choice: str = "auto",
        thinking: bool | None = None,
    ) -> LLMResponse:
        self._mark_call("chat_with_tools")
        """Mock 实现：第一轮根据关键词选 1 个 tool 调用，第二轮收到工具结果后给最终答案。"""
        # 找到最后一条 user 消息（原始 query）
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        # 检查对话里已经发生的工具调用，避免重复
        called_tools = [
            tc.get("function", {}).get("name", "")
            for m in messages
            if m.get("role") == "assistant"
            for tc in (m.get("tool_calls") or [])
        ]

        tool_names = {t["function"]["name"] for t in tools}
        next_tool = self._pick_tool(last_user, called_tools, tool_names)

        if next_tool is None:
            return LLMResponse(
                content=self._final_answer(last_user, called_tools),
                finish_reason="stop",
            )

        return LLMResponse(
            tool_calls=[
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    name=next_tool,
                    arguments=self._mock_args(next_tool, last_user),
                )
            ],
            finish_reason="tool_calls",
        )

    def _pick_tool(self, query: str, already_called: list[str], available: set[str]) -> str | None:
        lowered = query.lower()
        wants_music_candidates = any(
            kw in lowered
            for kw in ["recommend", "suggest", "推荐", "搜索", "找歌", "歌单", "playlist", "chill", "适合", "跑步"]
        )
        if wants_music_candidates and TOOL_WEB_MUSIC_SEARCH in available and TOOL_WEB_MUSIC_SEARCH not in already_called:
            return TOOL_WEB_MUSIC_SEARCH
        if TOOL_WEB_MUSIC_SEARCH in already_called:
            if any(kw in lowered for kw in ["playlist", "歌单", "合集"]) and TOOL_PLAYLIST in available and TOOL_PLAYLIST not in already_called:
                return TOOL_PLAYLIST
            if any(kw in lowered for kw in ["搜索", "找歌", "search"]) and TOOL_SEARCH in available and TOOL_SEARCH not in already_called:
                return TOOL_SEARCH
            if TOOL_RECOMMEND in available and TOOL_RECOMMEND not in already_called:
                return TOOL_RECOMMEND
        for keywords, tool in _TOOL_RULES:
            if any(kw in lowered for kw in keywords) and tool in available and tool not in already_called:
                return tool
        if not already_called and TOOL_RECOMMEND in available:
            return TOOL_RECOMMEND
        return None

    def _mock_args(self, tool: str, query: str) -> dict[str, Any]:
        if tool in {TOOL_RECOMMEND, TOOL_SEARCH, TOOL_RETRIEVE, TOOL_WEB_MUSIC_SEARCH}:
            return {"query": query, "top_k": _infer_count(query) or 5}
        if tool == TOOL_PLAYLIST:
            return {"instruction": query, "target_count": _infer_count(query)}
        if tool == TOOL_MEMORY_UPDATE:
            return {"event": query}
        if tool == TOOL_FETCH_METADATA:
            return {"url": query, "use_network": False}
        if tool == TOOL_IMPORT_NETEASE_PLAYLIST:
            return {"playlist_ref": query, "limit": 20}
        return {}

    def _final_answer(self, query: str, called: list[str]) -> str:
        if not called:
            return f"已处理你的请求：{query}"
        if TOOL_RECOMMEND in called:
            return random.choice(CHAT_TEMPLATES).format(genre="流行", mood="轻松")
        if TOOL_IMPORT_NETEASE_PLAYLIST in called or TOOL_WEB_MUSIC_SEARCH in called:
            return f"我已经根据真实输入工具的 observation 整合了结果：{query}"
        return f"已根据 {len(called)} 个工具的结果整合答案：{query}"

    def _intent(self, prompt: str) -> str:
        return '{"actions": ["recommend"], "confidence": 0.8, "reason": "mock intent"}'

    def _query_plan(self, prompt: str) -> str:
        """模拟结构化意图规划：根据关键词判意图，复刻真实 LLM 的 JSON 输出。"""
        import json as _json

        current_query = _extract_current_query(prompt)
        intent = _mock_intent_for_query(current_query)
        spec = get_intent(intent)
        target = _infer_count(current_query)
        use_web = spec.online_default
        use_local = intent not in {"chat", "taste", "video", "artist_info", "import"}
        use_vector = intent in {"recommend", "playlist", "journey", "taste_experiment"}
        query_for_rewrite = prompt if _needs_history_for_mock_rewrite(current_query) else current_query
        entities = _mock_entities(current_query, intent)
        search_query, language = self._mock_search_query(query_for_rewrite, intent, entities)
        search_variants = _mock_search_variants(current_query, search_query, intent, entities)
        return _json.dumps({
            "intent": intent, "entities": entities, "use_local": use_local,
            "use_vector": use_vector, "use_web": use_web,
            "search_query": search_query, "search_variants": search_variants, "language": language,
            "target_count": target,
            "reasoning": f"mock：{intent} 意图",
        }, ensure_ascii=False)

    def _decompose(self, prompt: str) -> str:
        import json as _json

        query = prompt.split("用户请求：", 1)[-1].strip()
        lowered = query.lower()
        if "然后" in prompt or "and then" in lowered:
            raw_parts = re.split(r"(?:然后|之后|接着|and then|after that|finally)", query)
            parts = [p.strip(" ：:\n\t，,。；;") for p in raw_parts if p.strip(" ：:\n\t，,。；;")]
        else:
            parts = [query]
        subtasks = []
        for idx, part in enumerate(parts):
            intent = match_intent_by_keywords(part) or "recommend"
            subtasks.append({
                "intent": intent,
                "query": part,
                "depends_on_prev": idx > 0 and any(token in part for token in ["再", "类似", "基于", "继续"]),
            })
        return _json.dumps({"subtasks": subtasks}, ensure_ascii=False)

    def _compound_synthesis(self, prompt: str) -> str:
        query_match = re.search(r"用户原始请求：(.*?)\n\n", prompt, re.S)
        query = query_match.group(1).strip() if query_match else "这次请求"
        steps = re.findall(r"子任务 \d+（.*?）\n任务：(.+?)\n结果摘要：(.+?)(?:\n主要答复：(.+?))?(?=\n\n子任务 |\Z)", prompt, re.S)
        if not steps:
            return f"我已经完成“{query}”的处理，并整理好了结果。"
        lines = [f"我已经把“{query}”分步处理好了。"]
        for index, (_, summary, answer) in enumerate(steps, start=1):
            lines.append(f"{index}. {summary.strip()}")
            if (answer or "").strip():
                lines.append(answer.strip().splitlines()[0][:120])
        return "\n".join(lines)

    def _query_plan_repair(self, prompt: str) -> str:
        raw = prompt.split("待修复原始输出：", 1)[-1]
        repaired = extract_json_dict(raw)
        if repaired:
            import json as _json

            repaired.setdefault("entities", [])
            repaired.setdefault("use_local", True)
            repaired.setdefault("use_vector", False)
            repaired.setdefault("use_web", False)
            repaired.setdefault("search_query", "")
            repaired.setdefault("search_variants", [])
            repaired.setdefault("language", "")
            repaired.setdefault("target_count", None)
            repaired.setdefault("reasoning", "")
            repaired["intent"] = repaired.get("intent") or "chat"
            return _json.dumps(repaired, ensure_ascii=False)
        return '{"intent":"chat","entities":[],"use_local":false,"use_vector":false,"use_web":false,"search_query":"","search_variants":[],"language":"","target_count":null,"reasoning":"repair fallback"}'

    @staticmethod
    def _mock_search_query(prompt: str, intent: str, entities: list[str] | None = None) -> tuple[str, str]:
        """Mock 版查询改写：粗粒度把否定转正向 + 抓场景词，复刻真实 LLM 的 search_query。

        不追求精度（mock 仅供 demo/测试），但要体现"否定不原样发搜索、保留场景"的形状：
        - 含"不要中文/英文歌" → 正向词加"英文 欧美"，language=en
        - 含"不要英文/中文歌" → language=zh
        - 抓常见场景/情绪词（深夜/学习/跑步/放松…）拼进正向词
        """
        if intent in {"taste", "chat", "import"}:
            return "", ""
        lowered = prompt.lower()
        language = ""
        positives: list[str] = []
        entities = [e for e in (entities or []) if e]
        if ("不要中文" in prompt) or ("不要华语" in prompt) or ("英文" in prompt and "不要英文" not in prompt):
            language = "en"
            positives.append("英文 欧美")
        elif ("不要英文" in prompt) or ("要中文" in prompt) or ("中文" in prompt and "不要中文" not in prompt):
            language = "zh"
            positives.append("华语")
        scene_words = ["深夜", "学习", "跑步", "运动", "放松", "睡前", "通勤", "工作", "chill", "安静"]
        for w in scene_words:
            if w in lowered or w in prompt:
                positives.append(w)
        if intent == "taste_experiment":
            positives.extend(["探索", "新风格", "相邻风格"])
        if entities:
            positives = [*entities, *positives]
        if intent == "video" and positives and not any(k in " ".join(positives).lower() for k in ["mv", "video"]):
            positives.append("MV")
        return " ".join(dict.fromkeys(positives)).strip(), language

    def _reason(self, prompt: str) -> str:
        template = random.choice(REASON_TEMPLATES)
        return template.format(
            genre=_extract_or_default(prompt, "流行"),
            mood=_extract_or_default(prompt, "轻松"),
            time="现在",
        )

    def _search_summary(self, prompt: str) -> str:
        template = random.choice(SEARCH_TEMPLATES)
        return template.format(count=5, genre="流行", mood="欢快")

    def _daily_summary(self, prompt: str) -> str:
        template = random.choice(DAILY_SUMMARY_TEMPLATES)
        return template.format(count=25, genres="流行、电子、民谣", mood="轻松愉快")

    def _chat(self, prompt: str) -> str:
        template = random.choice(CHAT_TEMPLATES)
        return template.format(genre="流行", mood="轻松")


def _extract_or_default(text: str, default: str) -> str:
    keywords = ["流行", "摇滚", "电子", "古典", "爵士", "民谣", "说唱", "R&B"]
    for kw in keywords:
        if kw in text:
            return kw
    return default


def _infer_count(text: str) -> int | None:
    match = re.search(r"(\d{1,3})\s*(?:首|个|tracks?|songs?)?", text, re.IGNORECASE)
    if not match:
        return None
    return max(1, min(int(match.group(1)), 100))


def _mock_search_variants(query: str, search_query: str, intent: str, entities: list[str] | None = None) -> list[str]:
    if intent in {"chat", "taste", "import"}:
        return []
    variants: list[str] = []
    lowered = f"{query} {search_query}".lower()
    typo_map = {
        "emenem": "Eminem",
        "taylor swift": "Taylor Swift pop country",
        "newjeans": "NewJeans K-pop",
    }
    synonym_map = {
        "说唱": "rap hip hop",
        "嘻哈": "hip hop rap",
        "r&b": "rnb soul",
        "rnb": "R&B soul",
        "爵士": "jazz",
        "放松": "chill relaxing",
        "深夜": "late night chill",
        "跑步": "running workout",
        "学习": "focus study",
        "梦核": "dream pop dreamy",
    }
    for key, value in {**typo_map, **synonym_map}.items():
        if key in lowered:
            variants.append(value)
    if entities:
        for entity in entities[:2]:
            if intent == "video":
                variants.append(f"{entity} MV")
            elif intent == "artist_albums":
                variants.append(f"{entity} album")
    seen = {search_query.strip().lower()} if search_query.strip() else set()
    out: list[str] = []
    for item in variants:
        value = item.strip()
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out[:4]


def _extract_current_query(prompt: str) -> str:
    """Extract the current user turn from the query-plan prompt.

    Mock planning must not scan the whole prompt/history; otherwise an old
    "音乐旅程/热身/冲刺" turn poisons later unrelated turns.
    """
    text = (prompt or "").strip()
    if "【本轮输入】" in text:
        return text.rsplit("【本轮输入】", 1)[-1].strip()
    assistant_matches = list(
        re.finditer(r"(?:^|\n)\s*(?:assistant|助手)\s*[:：][^\n]*(?:\n|$)", text, flags=re.IGNORECASE)
    )
    if assistant_matches:
        trailing = text[assistant_matches[-1].end():].strip()
        if trailing:
            return trailing
    matches = re.findall(r"(?:^|\n)\s*(?:用户|user)\s*[:：]\s*(.+)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return text


def _mock_intent_for_query(query: str) -> str:
    lowered = query.lower()
    stripped = lowered.strip()
    if (
        any(k in query for k in ["你好", "嗨", "在吗"])
        or re.fullmatch(r"(hi|hello|hey)[!！。.\s]*", stripped)
    ):
        return "chat"
    if any(k in lowered for k in ["网易云歌单", "导入歌单", "playlist?id", "导入"]):
        return "import"
    # "跑步冲刺歌单" 是歌单，不是多阶段旅程；只有明确说旅程/journey/从X到Y才走 journey。
    if any(k in lowered for k in ["歌单", "playlist", "合集"]):
        return "playlist"
    if any(k in lowered for k in ["音乐旅程", "journey"]) or ("从" in query and "到" in query):
        return "journey"
    return match_intent_by_keywords(query) or "recommend"


def _needs_history_for_mock_rewrite(query: str) -> bool:
    return is_continuation(query) or any(token in query for token in ("不要", "别", "换成", "改成", "太吵", "安静点"))


def _mock_entities(query: str, intent: str) -> list[str]:
    if intent in {"chat", "taste", "taste_experiment", "import", "journey"}:
        return []
    if any(token in query for token in ("不要", "别", "换成", "改成")):
        return []
    english_stop = {
        "mv", "video", "live", "concert", "playlist", "album", "albums",
        "recommend", "suggest", "search", "find", "about", "biography",
    }
    english = [
        token for token in re.findall(r"[A-Za-z][A-Za-z0-9'&\-]*", query)
        if token.lower() not in english_stop
    ]
    entities: list[str] = []
    if english:
        entities.append(" ".join(english))

    cleaned = query
    cjk_noise = [
        "帮我", "给我", "推荐", "搜索", "搜一下", "找", "找一下", "一些",
        "几首", "一首", "的歌", "歌曲", "音乐", "专辑", "唱片", "大碟",
        "有哪些", "有哪几张", "哪几张", "几张", "介绍一下", "介绍",
        "背景", "成员", "出道", "简介", "资料", "百科", "是谁", "这个团体",
        "牛逼吗", "怎么样", "评价", "风格", "什么水平", "视频", "现场",
        "演唱会", "导入", "网易云歌单", "歌单",
    ]
    for noise in sorted(cjk_noise, key=len, reverse=True):
        cleaned = cleaned.replace(noise, " ")
    for token in re.findall(r"[一-鿿㐀-䶿豈-﫿]{2,}", cleaned):
        token = token.strip()
        if token and token not in entities:
            entities.append(token)
    return entities[:3]
