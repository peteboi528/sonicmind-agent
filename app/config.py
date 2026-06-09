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
        self.llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
        self.llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
        self.external_source: str = os.getenv("EXTERNAL_SOURCE", "mock")
        self.store_root: str = os.getenv("STORE_ROOT", "data/store")
        self.media_root: str = os.getenv("MEDIA_ROOT", "data/media")
        self.resource_library_path: str = os.getenv("RESOURCE_LIBRARY_PATH", "data/resource_library.sqlite")
        self.daily_rec_count: int = int(os.getenv("DAILY_REC_COUNT", "25"))
        self.enable_online_enrich: bool = os.getenv("ENABLE_ONLINE_ENRICH", "false").lower() == "true"
        # Phase 3：embedding 检索。默认关闭，缺依赖/加载失败时自动回退 TF cosine。
        self.enable_embeddings: bool = os.getenv("ENABLE_EMBEDDINGS", "false").lower() == "true"
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
        # Phase 1：三锚精排权重（自动归一化；缺项时重分配给其余锚）。
        self.tri_anchor_w_semantic: float = float(os.getenv("TRI_ANCHOR_W_SEMANTIC", "0.45"))
        self.tri_anchor_w_personal: float = float(os.getenv("TRI_ANCHOR_W_PERSONAL", "0.30"))
        self.tri_anchor_w_behavior: float = float(os.getenv("TRI_ANCHOR_W_BEHAVIOR", "0.25"))
        # MMR 多样性重排：λ 越大越偏相关性，越小越偏多样性。
        self.mmr_lambda: float = float(os.getenv("MMR_LAMBDA", "0.7"))
        # Thompson Sampling 探索：尾部候选中用于探索的比例。
        self.exploration_ratio: float = float(os.getenv("EXPLORATION_RATIO", "0.2"))
        self.enable_rerank: bool = os.getenv("ENABLE_RERANK", "true").lower() == "true"

    @property
    def mock_mode(self) -> bool:
        return not self.llm_api_key


settings = Settings()
