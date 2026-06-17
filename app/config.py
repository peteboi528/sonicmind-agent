from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Settings:
    def __init__(self) -> None:
        self.llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        self.llm_api_key: str = os.getenv("LLM_API_KEY", "")
        self.llm_model: str = os.getenv("LLM_MODEL", "qwen2.5")
        self.llm_fast_model: str = os.getenv("LLM_FAST_MODEL", self.llm_model)
        self.llm_strong_model: str = os.getenv("LLM_STRONG_MODEL", self.llm_model)
        self.llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
        # 推理模型（deepseek-v4-flash 等）会先消耗 token 做推理，再产出 content。
        # 1024 容易被推理吃光导致 content 为空，故默认提到 2048。
        self.llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
        self.llm_input_price_per_1m_tokens: float = float(os.getenv("LLM_INPUT_PRICE_PER_1M_TOKENS", "0"))
        self.llm_output_price_per_1m_tokens: float = float(os.getenv("LLM_OUTPUT_PRICE_PER_1M_TOKENS", "0"))
        # 温度三档：结构化任务要稳定、对话要自然、生成文案要有变化。
        self.struct_task_temperature: float = float(os.getenv("STRUCT_TASK_TEMPERATURE", "0.1"))
        self.dialog_temperature: float = float(os.getenv("DIALOG_TEMPERATURE", "0.6"))
        self.generation_temperature: float = float(os.getenv("GENERATION_TEMPERATURE", "0.7"))
        self.external_source: str = os.getenv("EXTERNAL_SOURCE", "netease")
        self.lastfm_api_key: str = os.getenv("LASTFM_API_KEY", "")
        self.tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
        self.store_root: str = os.getenv("STORE_ROOT", "data/store")
        self.media_root: str = os.getenv("MEDIA_ROOT", "data/media")
        self.resource_library_path: str = os.getenv("RESOURCE_LIBRARY_PATH", "data/resource_library.sqlite")
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
        # Thompson Sampling 探索：尾部候选中用于探索的比例。
        self.exploration_ratio: float = float(os.getenv("EXPLORATION_RATIO", "0.2"))
        self.enable_rerank: bool = os.getenv("ENABLE_RERANK", "true").lower() == "true"
        # Deep/Agentic 模式：复合多步任务走真迭代 ReAct（一级分支，非降级兜底）。
        # 仅真实 LLM（非 mock）下生效；mock 模式仍走图，保持测试/demo 稳定。
        self.enable_deep_mode: bool = os.getenv("ENABLE_DEEP_MODE", "true").lower() == "true"
        # P1-G 记忆升级：语义召回 + LLM 偏好抽取兜底 + 巩固画像。
        # 仅真实 LLM 下做 LLM 抽取/巩固；语义召回随 embeddings 开关自动降级。
        self.enable_semantic_memory: bool = os.getenv("ENABLE_SEMANTIC_MEMORY", "true").lower() == "true"
        self.memory_consolidation_interval: int = int(os.getenv("MEMORY_CONSOLIDATION_INTERVAL", "5"))
        self.memory_recall_top_k: int = int(os.getenv("MEMORY_RECALL_TOP_K", "3"))
        self.episodic_memory_cap: int = int(os.getenv("EPISODIC_MEMORY_CAP", "120"))

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


def _csv_env(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or [default]


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
