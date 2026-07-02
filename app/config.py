from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)


def _absolute_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _default_store_root() -> Path:
    """Pick the populated legacy store once, independently of process cwd.

    Older releases interpreted ``data/store`` relative to the launch directory,
    so starting uvicorn from ``frontend/`` created a second store.  We do not
    move or merge user data automatically: among the two known locations, keep
    using the one containing the most assets.  An explicit STORE_ROOT always
    wins.

    When BOTH locations hold assets we must guess by count — that ambiguity is
    exactly what makes data look like it "vanished" after a restart from a
    different cwd.  Loudly warn so the operator pins STORE_ROOT.
    """
    candidates = [PROJECT_ROOT / "data/store", PROJECT_ROOT / "frontend/data/store"]
    counts = {
        path: len(list((path / "assets").glob("*.json"))) if (path / "assets").exists() else 0
        for path in candidates
    }
    populated = [p for p, n in counts.items() if n > 0]
    if len(populated) > 1:
        logger.warning(
            "检测到多个曲库目录都有数据 %s——正按文件数量挑选，重启/换启动目录可能读到不同库。"
            "请在 .env 显式设置 STORE_ROOT 锁定，避免曲库'忽然消失'。",
            {str(p): n for p, n in counts.items()},
        )
    return max(candidates, key=lambda path: counts[path])


class Settings:
    def __init__(self) -> None:
        self.llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        self.llm_api_key: str = os.getenv("LLM_API_KEY", "")
        self.llm_model: str = os.getenv("LLM_MODEL", "qwen2.5")
        self.llm_fast_model: str = os.getenv("LLM_FAST_MODEL", self.llm_model)
        self.llm_strong_model: str = os.getenv("LLM_STRONG_MODEL", self.llm_model)
        self.llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
        # 连接超时（建连阶段）：连接类卡死应快速失败，别等满 llm_timeout_seconds。
        self.llm_connect_timeout: float = float(os.getenv("LLM_CONNECT_TIMEOUT", "8"))
        # LLM 网络瞬时错误（连接重置 / 5xx / 429）的重试次数。超时与 4xx 不重试——
        # 超时多半是慢生成，重试只会加倍等待；4xx（鉴权/格式）重试无意义。
        self.llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "1"))
        # 推理模型（deepseek-v4-flash 等）会先消耗 token 做推理，再产出 content。
        # 1024 容易被推理吃光导致 content 为空，故默认提到 2048。
        self.llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
        # DeepSeek-V4 思考模式开关。服务端默认 enabled（每次先吐 reasoning_content，用户看不到却要等）。
        # 对结构化/闲聊小任务推理无收益：默认 false 关掉，单次调用快约 35%、输出 token 少约 3 倍。
        # 真正需要推理的调用可按 thinking=True 显式开启；设 true 则全局恢复思考。
        self.llm_thinking: bool = os.getenv("LLM_THINKING", "false").lower() == "true"
        # 思考开启时的强度（DeepSeek-V4：low/medium→high，xhigh→max）。仅 thinking=true 时生效。
        self.llm_reasoning_effort: str = os.getenv("LLM_REASONING_EFFORT", "")
        self.llm_input_price_per_1m_tokens: float = float(os.getenv("LLM_INPUT_PRICE_PER_1M_TOKENS", "0"))
        self.llm_output_price_per_1m_tokens: float = float(os.getenv("LLM_OUTPUT_PRICE_PER_1M_TOKENS", "0"))
        # 温度三档：结构化任务要稳定、对话要自然、生成文案要有变化。
        self.struct_task_temperature: float = float(os.getenv("STRUCT_TASK_TEMPERATURE", "0.1"))
        self.dialog_temperature: float = float(os.getenv("DIALOG_TEMPERATURE", "0.6"))
        self.generation_temperature: float = float(os.getenv("GENERATION_TEMPERATURE", "0.7"))
        self.external_source: str = os.getenv("EXTERNAL_SOURCE", "netease")
        self.lastfm_api_key: str = os.getenv("LASTFM_API_KEY", "")
        self.tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
        self.spotify_client_id: str = os.getenv("SPOTIFY_CLIENT_ID", "")
        self.spotify_client_secret: str = os.getenv("SPOTIFY_CLIENT_SECRET", "")
        self.discogs_token: str = os.getenv("DISCOGS_TOKEN", "")
        store_env = os.getenv("STORE_ROOT", "").strip()
        store_root = _absolute_path(store_env) if store_env else _default_store_root()
        self.store_candidates: dict[str, int] = {
            str(path.resolve()): len(list((path / "assets").glob("*.json")))
            if (path / "assets").exists() else 0
            for path in (PROJECT_ROOT / "data/store", PROJECT_ROOT / "frontend/data/store")
        }
        data_root = store_root.parent
        self.store_root: str = str(store_root.resolve())
        self.media_root: str = str(_absolute_path(os.getenv("MEDIA_ROOT", "").strip()).resolve()) if os.getenv("MEDIA_ROOT", "").strip() else str((data_root / "media").resolve())
        resource_env = os.getenv("RESOURCE_LIBRARY_PATH", "").strip()
        self.resource_library_path: str = str(_absolute_path(resource_env).resolve()) if resource_env else str((data_root / "resource_library.sqlite").resolve())
        self.agent_checkpoint_path: str = str((data_root / "agent_checkpoints.sqlite").resolve())
        self.agent_trace_path: str = str((data_root / "agent_traces.sqlite").resolve())
        self.agent_retention_days: int = int(os.getenv("AGENT_RETENTION_DAYS", "30"))
        # Web 历史持久化：完整对话用于跨重启恢复；推荐历史带 TTL，避免旧推荐长期污染演示。
        self.chat_history_max_threads: int = int(os.getenv("CHAT_HISTORY_MAX_THREADS", "30"))
        self.chat_history_max_messages_per_thread: int = int(os.getenv("CHAT_HISTORY_MAX_MESSAGES_PER_THREAD", "80"))
        self.recommendation_history_ttl_days: int = int(os.getenv("RECOMMENDATION_HISTORY_TTL_DAYS", "14"))
        self.recommendation_history_max_items: int = int(os.getenv("RECOMMENDATION_HISTORY_MAX_ITEMS", "120"))
        self.agent_checkpoints: bool = _bool_env("AGENT_CHECKPOINTS", True)
        self.local_tracing: bool = _bool_env("LOCAL_TRACING", True)
        self.llm_json_mode: str = os.getenv("LLM_JSON_MODE", "auto").strip().lower()
        self.allowed_origins: list[str] = _csv_env("ALLOWED_ORIGINS", "*")
        self.daily_rec_count: int = int(os.getenv("DAILY_REC_COUNT", "25"))
        self.enable_online_enrich: bool = os.getenv("ENABLE_ONLINE_ENRICH", "false").lower() == "true"
        # Phase 3：embedding 检索。默认 auto：装了 sentence-transformers 自动启用，否则回退 TF cosine + 同义词 boost。
        self.enable_embeddings: bool = os.getenv("ENABLE_EMBEDDINGS", "auto").lower() in ("true", "auto", "1")
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
        # Phase 1：三锚精排权重（自动归一化；缺项时重分配给其余锚）。
        self.tri_anchor_w_semantic: float = float(os.getenv("TRI_ANCHOR_W_SEMANTIC", "0.45"))
        self.tri_anchor_w_personal: float = float(os.getenv("TRI_ANCHOR_W_PERSONAL", "0.30"))
        self.tri_anchor_w_behavior: float = float(os.getenv("TRI_ANCHOR_W_BEHAVIOR", "0.25"))
        # P2-H：可选第四锚——协同过滤（item-item 共现）。默认 0：无 CF 数据/不启用时
        # 权重重分配让回三锚，行为与三锚时代一致。设 >0 才把跨用户信号纳入精排。
        self.tri_anchor_w_collaborative: float = float(os.getenv("TRI_ANCHOR_W_COLLABORATIVE", "0.20"))
        # MMR 多样性重排：λ 越大越偏相关性，越小越偏多样性。
        self.mmr_lambda: float = float(os.getenv("MMR_LAMBDA", "0.7"))
        # rerank 微调魔法数（量级远小于相关性差异；进 config 便于 P1 eval 做 ablation）。
        # profile 艺人关系：core/rising 加分、avoid 减分；语言加权 multiplier = base + span*share；
        # 场景 vibe 降分幅度；hygiene 语义 junk 判定余量。
        self.rerank_profile_core_delta: float = float(os.getenv("RERANK_PROFILE_CORE_DELTA", "0.06"))
        self.rerank_profile_avoid_delta: float = float(os.getenv("RERANK_PROFILE_AVOID_DELTA", "-0.12"))
        self.rerank_language_weight_base: float = float(os.getenv("RERANK_LANGUAGE_WEIGHT_BASE", "0.85"))
        self.rerank_language_weight_span: float = float(os.getenv("RERANK_LANGUAGE_WEIGHT_SPAN", "0.30"))
        self.rerank_scene_vibe_penalty: float = float(os.getenv("RERANK_SCENE_VIBE_PENALTY", "-0.08"))
        self.hygiene_junk_margin: float = float(os.getenv("HYGIENE_JUNK_MARGIN", "0.08"))
        # recommend 本地占比：默认（exact/anchor 路由）、每日推荐（略压本地让位线上发现）。
        self.recommend_local_ratio_default: float = float(os.getenv("RECOMMEND_LOCAL_RATIO_DEFAULT", "0.4"))
        self.daily_local_ratio: float = float(os.getenv("DAILY_LOCAL_RATIO", "0.3"))
        # Thompson Sampling 探索：尾部候选中用于探索的比例。
        self.exploration_ratio: float = float(os.getenv("EXPLORATION_RATIO", "0.2"))
        self.enable_explore: bool = os.getenv("ENABLE_EXPLORE", "true").lower() == "true"
        self.explore_ratio: float = float(os.getenv("EXPLORE_RATIO", str(self.exploration_ratio)))
        self.tri_anchor_w_explore: float = float(os.getenv("TRI_ANCHOR_W_EXPLORE", "0.15"))
        self.fuzzy_threshold: int = int(os.getenv("FUZZY_THRESHOLD", "82"))
        self.max_search_variants: int = int(os.getenv("MAX_SEARCH_VARIANTS", "4"))
        self.dense_recall_min_score: float = float(os.getenv("DENSE_RECALL_MIN_SCORE", "0.55"))
        self.enable_rerank: bool = os.getenv("ENABLE_RERANK", "true").lower() == "true"
        self.enable_parallel_tools: bool = os.getenv("ENABLE_PARALLEL_TOOLS", "true").lower() == "true"
        # 多意图并行：一句话同时含两类意图（"推几首 X 顺便讲讲他"）时，除 primary 外
        # 再挂 ≤1 个 secondary 子计划，两条工具链并行跑、一条 message 出两段结果。
        # 默认关闭 → 即使 LLM 填了 secondary，planner 也丢弃，行为字节级等于单意图今天。
        self.enable_multi_intent: bool = os.getenv("ENABLE_MULTI_INTENT", "false").lower() == "true"
        # reflect 候选补量回环：reflect 剔除违规候选后若不足，回 execute_tools+reflect 再补一轮。
        # 默认关闭——这会引入第 4/5 次串行往返（含联网搜索）。reflect 本身仍跑（thinking-off 后很快），
        # 不足时由 _compose_intro 如实说明 shortfall，不阻塞主流程。
        self.enable_reflect_refine: bool = os.getenv("ENABLE_REFLECT_REFINE", "false").lower() == "true"
        # 零候选/工具错误恢复（P4 阶梯）：attempt0=正向词变体重搜，attempt1=变体耗尽切本地召回，
        # 再超则诚实空。LLM 恢复单趟（recovery_llm_used），第二 attempt 走纯确定性，不增 LLM 成本。
        self.enable_empty_result_recovery: bool = os.getenv("ENABLE_EMPTY_RESULT_RECOVERY", "true").lower() == "true"
        # 知识意图自省（Reflexion）：reflect 里对知识链路做确定性核对（resolve 是否空 / 档案是否降级）
        # ——核对本身零 LLM、零延迟、默认始终跑（只出 trace 可观测）。仅当档案真正降级时，开此开关才会
        # 回 execute_tools 用清洗后的实体名重试一次 resolve（重跑知识链路 ~20-40s，故默认关，对比手感用）。
        self.enable_knowledge_refine: bool = os.getenv("ENABLE_KNOWLEDGE_REFINE", "false").lower() == "true"
        self.empty_result_recovery_max_attempts: int = max(
            0, int(os.getenv("EMPTY_RESULT_RECOVERY_MAX_ATTEMPTS", "2"))
        )
        # P4 全局单轮墙钟预算（通用路径 recommend/search/playlist）。超预算则 reflect 停止
        # refine/recovery 回环，由 finalize 的 shortfall 兜底诚实说明（治"超时卡死"）。
        # knowledge 路径走自己的 knowledge_turn_budget_seconds（36s），不在此列。
        self.turn_budget_seconds: float = float(os.getenv("TURN_BUDGET_SECONDS", "15"))
        # P4v2：在真正耗尽 turn budget 之前，先按剩余预算做渐进降级：
        # soft = 关 search_variants / 禁用 LLM recovery；hard = recovery 直接切本地。
        self.turn_budget_soft_degrade_seconds: float = float(os.getenv("TURN_BUDGET_SOFT_DEGRADE_SECONDS", "6"))
        self.turn_budget_hard_degrade_seconds: float = float(os.getenv("TURN_BUDGET_HARD_DEGRADE_SECONDS", "3"))
        # Music knowledge agent latency budget. 这些链路会并行查资料/乐评，
        # 必须有全链路墙钟上限，避免单个请求因为搜索链条过长而崩掉。
        # 注意：默认值按「慢网络」校准——从国内访问 MusicBrainz/last.fm/Discogs 等
        # 西方 API 单次常 5–8s（实测）。源超时给不到就会全员超时→实体 unresolved→档案降级。
        # 快网络不受影响（源 <1s 返回，不会真等满超时）。需要更激进可在 .env 调小。
        self.knowledge_turn_budget_seconds: float = float(os.getenv("KNOWLEDGE_TURN_BUDGET_SECONDS", "50"))
        self.knowledge_quick_budget_seconds: float = float(os.getenv("KNOWLEDGE_QUICK_BUDGET_SECONDS", "12"))
        self.knowledge_source_timeout_seconds: float = float(os.getenv("KNOWLEDGE_SOURCE_TIMEOUT_SECONDS", "10"))
        self.knowledge_review_timeout_seconds: float = float(os.getenv("KNOWLEDGE_REVIEW_TIMEOUT_SECONDS", "12"))
        self.knowledge_llm_timeout_seconds: float = float(os.getenv("KNOWLEDGE_LLM_TIMEOUT_SECONDS", "5"))
        self.knowledge_max_review_sources: int = max(1, int(os.getenv("KNOWLEDGE_MAX_REVIEW_SOURCES", "5")))
        self.knowledge_max_search_queries: int = max(1, int(os.getenv("KNOWLEDGE_MAX_SEARCH_QUERIES", "3")))
        self.knowledge_max_citations: int = max(1, int(os.getenv("KNOWLEDGE_MAX_CITATIONS", "8")))
        self.knowledge_deep_review_enabled: bool = os.getenv("KNOWLEDGE_DEEP_REVIEW_ENABLED", "false").lower() == "true"
        # 乐评正文抓取（Tavily Extract / Discogs API）预算：把 MB relations 里那批
        # 高价值来源（last.fm/Discogs/Genius/AllMusic…）的真实正文读回来填进 citation.excerpt，
        # 喂给合成 LLM 出专业中文乐评。受保护预算——不让 MB 把它饿死。
        self.knowledge_review_extract_timeout_seconds: float = float(os.getenv("KNOWLEDGE_REVIEW_EXTRACT_TIMEOUT_SECONDS", "10"))
        self.knowledge_review_extract_max_sources: int = max(1, int(os.getenv("KNOWLEDGE_REVIEW_EXTRACT_MAX_SOURCES", "4")))
        # dossier 工具外层超时：要装得下「抓 N 个正文 + 1 次合成 LLM（开思考模式）」，给难抓的专辑留余地。
        self.knowledge_dossier_timeout_seconds: float = float(os.getenv("KNOWLEDGE_DOSSIER_TIMEOUT_SECONDS", "24"))
        # 元数据波(MB/Spotify/Discogs/web) 内部预算封顶——它是 bonus 内容，不能让它吃光整轮预算饿死合成。
        # 慢网络下单源要 ~7s，故提到 15s；快网络仍随源返回即结束，不真等满。
        self.knowledge_metadata_timeout_seconds: float = float(os.getenv("KNOWLEDGE_METADATA_TIMEOUT_SECONDS", "15"))
        # 知识合成这一处开 LLM 思考模式（只此一处，不全局开以免拖慢规划/对话/反思）。
        self.knowledge_synth_thinking_enabled: bool = os.getenv("KNOWLEDGE_SYNTH_THINKING_ENABLED", "true").lower() == "true"
        # MusicBrainz 结构化知识层（免费、无 key）：实体消歧 + 权威元数据。
        # 评测/离线测试可关，保证确定性。
        self.enable_musicbrainz: bool = os.getenv("ENABLE_MUSICBRAINZ", "true").lower() == "true"
        # Spotify(audio_features/genres/popularity)与 Discogs(发行/styles/tracklist)结构化知识层。
        # 都需凭证；缺失或关闭时知识链路自动降级到其余来源。
        self.enable_spotify: bool = os.getenv("ENABLE_SPOTIFY", "true").lower() == "true"
        self.enable_discogs: bool = os.getenv("ENABLE_DISCOGS", "true").lower() == "true"
        # P1-G 记忆升级：语义召回 + LLM 偏好抽取兜底 + 巩固画像。
        # 仅真实 LLM 下做 LLM 抽取/巩固；语义召回随 embeddings 开关自动降级。
        self.enable_semantic_memory: bool = os.getenv("ENABLE_SEMANTIC_MEMORY", "true").lower() == "true"
        self.memory_consolidation_interval: int = int(os.getenv("MEMORY_CONSOLIDATION_INTERVAL", "5"))
        self.memory_recall_top_k: int = int(os.getenv("MEMORY_RECALL_TOP_K", "3"))
        self.episodic_memory_cap: int = int(os.getenv("EPISODIC_MEMORY_CAP", "120"))

        # ---- 专辑封面视觉识别（上传封面 → 识别专辑 → 复用知识链路）----
        # DeepSeek API 无视觉能力（仅文本模型），封面识别走独立 OpenAI 兼容视觉模型。
        # 默认阿里百炼 DashScope Qwen-VL（中英文封面 OCR 最强，CN/国际双端点）。
        # VISION_LLM_API_KEY 留空则禁用视觉，降级到本地 OCR（rapidocr，需单独装）→
        # 再读不出文字则提示用户直接输入专辑名/歌手。
        # 端点：国内 https://dashscope.aliyuncs.com/compatible-mode/v1；
        #       国际 https://dashscope-intl.aliyuncs.com/compatible-mode/v1。
        self.vision_llm_base_url: str = os.getenv("VISION_LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.vision_llm_api_key: str = os.getenv("VISION_LLM_API_KEY", "")
        self.vision_llm_model: str = os.getenv("VISION_LLM_MODEL", "qwen-vl-max-latest")
        self.vision_llm_timeout_seconds: float = float(os.getenv("VISION_LLM_TIMEOUT_SECONDS", "20"))
        self.vision_llm_connect_timeout: float = float(os.getenv("VISION_LLM_CONNECT_TIMEOUT", "8"))
        self.vision_llm_max_tokens: int = int(os.getenv("VISION_LLM_MAX_TOKENS", "512"))
        # 视觉识别置信阈值：低于此值视为不可信，回退 OCR / 提示输入。
        self.vision_confidence_threshold: float = float(os.getenv("VISION_CONFIDENCE_THRESHOLD", "0.5"))
        # 上传封面体积上限（字节）；超出 413 拒绝。送视觉前先用 Pillow 降到长边 ≤ 下值（省 token、避尺寸限制）。
        self.album_cover_max_bytes: int = int(os.getenv("ALBUM_COVER_MAX_BYTES", str(10 * 1024 * 1024)))
        self.vision_image_max_side: int = int(os.getenv("VISION_IMAGE_MAX_SIDE", "1024"))
        # OCR 兜底引擎开关（需装 rapidocr-onnxruntime；未装自动降级，不报错）。
        self.cover_ocr_enabled: bool = os.getenv("COVER_OCR_ENABLED", "true").lower() == "true"

        # ---- 强搜索 provider 化（知识类问答统一走 web_knowledge_search）----
        # auto 顺序：web(openai/tavily 若配置) → deepseek_parametric → duckduckgo → none。
        self.knowledge_search_provider: str = os.getenv("KNOWLEDGE_SEARCH_PROVIDER", "auto").strip().lower()
        # DeepSeek 直答要生成一整篇中文长文（参考裸 chat 效果），产出几百~上千 token，
        # 20s 不够会被 wait_for 砍掉→掉到稀疏 web→回到"资料不足"。给到 40s；turn budget 同步放宽。
        self.web_knowledge_timeout_seconds: float = float(os.getenv("WEB_KNOWLEDGE_TIMEOUT_SECONDS", "40"))
        self.web_knowledge_cache_ttl_hours: int = int(os.getenv("WEB_KNOWLEDGE_CACHE_TTL_HOURS", "24"))
        self.web_knowledge_max_sources: int = int(os.getenv("WEB_KNOWLEDGE_MAX_SOURCES", "8"))
        self.web_knowledge_direct_answer: bool = os.getenv("WEB_KNOWLEDGE_DIRECT_ANSWER", "false").lower() == "true"
        # DeepSeek 先验 provider：DeepSeek API 无 web-search 工具，只能召回训练知识。
        # 作为 web 不可用时的兜底——claim 全标 unverified/tier C/置信封顶，且 concert/fact_check
        # 等时效/精确性意图禁用（必须真来源或诚实拒答，否则就是幻觉）。
        self.deepseek_parametric_enabled: bool = os.getenv("DEEPSEEK_PARAMETRIC_ENABLED", "true").lower() == "true"
        self.deepseek_parametric_confidence_cap: float = float(os.getenv("DEEPSEEK_PARAMETRIC_CONFIDENCE_CAP", "0.45"))

        # ---- Bot 适配器配置（留空禁用） ----
        self.feishu_app_id: str = os.getenv("FEISHU_APP_ID", "")
        self.feishu_app_secret: str = os.getenv("FEISHU_APP_SECRET", "")
        self.feishu_verification_token: str = os.getenv("FEISHU_VERIFICATION_TOKEN", "")
        self.feishu_encrypt_key: str = os.getenv("FEISHU_ENCRYPT_KEY", "")
        self.wechat_token: str = os.getenv("WECHAT_TOKEN", "")
        self.wechat_app_id: str = os.getenv("WECHAT_APP_ID", "")
        self.wechat_app_secret: str = os.getenv("WECHAT_APP_SECRET", "")

        # ---- API 鉴权（多租户/部署用）----
        # auth_enabled=false（默认）：本地 demo 不校验，前端/测试无需带 key。
        # auth_enabled=true：
        # - USER_API_KEYS 非空：X-API-Key 解析为 user_id，服务端覆盖客户端传入的 user_id。
        # - USER_API_KEYS 为空：退回共享 API_KEY，只做访问门禁，兼容旧部署。
        self.auth_enabled: bool = os.getenv("AUTH_ENABLED", "false").lower() == "true"
        self.api_key: str = os.getenv("API_KEY", "")
        self.user_api_keys: dict[str, str] = _parse_user_api_keys(os.getenv("USER_API_KEYS", ""))

    def user_id_for_api_key(self, api_key: str | None) -> str | None:
        """Return the bound user_id for a per-user API key, or None for shared-key auth."""
        if not api_key:
            return None
        return self.user_api_keys.get(api_key)

    @property
    def mock_mode(self) -> bool:
        return not self.llm_api_key

    @property
    def vision_enabled(self) -> bool:
        """视觉识别是否可用：配置了视觉模型 API key 才算开。"""
        return bool(self.vision_llm_api_key)


def _csv_env(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or [default]


def _bool_env(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in {"true", "1", "yes", "on"}


def _parse_user_api_keys(raw: str) -> dict[str, str]:
    """Parse USER_API_KEYS as "user_id:key,user2:key2" into {key: user_id}."""
    mapping: dict[str, str] = {}
    for item in raw.split(","):
        if ":" not in item:
            continue
        user_id, key = (part.strip() for part in item.split(":", 1))
        if user_id and key:
            mapping[key] = user_id
    return mapping


settings = Settings()
