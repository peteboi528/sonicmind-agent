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
        self.external_source: str = os.getenv("EXTERNAL_SOURCE", "mock")
        self.store_root: str = os.getenv("STORE_ROOT", "data/store")
        self.media_root: str = os.getenv("MEDIA_ROOT", "data/media")
        self.daily_rec_count: int = int(os.getenv("DAILY_REC_COUNT", "25"))
        self.enable_online_enrich: bool = os.getenv("ENABLE_ONLINE_ENRICH", "false").lower() == "true"

    @property
    def mock_mode(self) -> bool:
        return not self.llm_api_key


settings = Settings()
