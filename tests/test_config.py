"""Settings 空串回退测试（Issue 2）。

裸 os.getenv(NAME, default) 在 env 存在但为 "" 时返回 ""，静默压掉默认值。GitHub 工作流
里 ${{ secrets.X }} 在 Secret 未配时求值为 ""，曾导致 LLM_BASE_URL/LLM_MODEL 被空串覆盖。
_env_str 助手对 LLM 端点/模型字段做 strip+回退；llm_api_key 空=mock mode 是既定语义，不回退。
"""
from __future__ import annotations

from app.config import Settings


def test_only_api_key_falls_back_to_defaults(monkeypatch):
    """仅配 LLM_API_KEY、base_url/model 留空 → 回退默认值，不空串。"""
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_BASE_URL", "")
    monkeypatch.setenv("LLM_MODEL", "")
    monkeypatch.setenv("LLM_FAST_MODEL", "")
    monkeypatch.setenv("LLM_STRONG_MODEL", "")
    s = Settings()
    assert s.llm_api_key == "sk-test"
    assert s.llm_base_url == "http://localhost:11434/v1"
    assert s.llm_model == "qwen2.5"
    assert s.llm_fast_model == "qwen2.5"
    assert s.llm_strong_model == "qwen2.5"
    assert s.mock_mode is False


def test_empty_api_key_still_means_mock_mode(monkeypatch):
    """LLM_API_KEY 留空仍是 mock mode（空 key 有语义，不回退默认）。"""
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_BASE_URL", "")
    monkeypatch.setenv("LLM_MODEL", "")
    s = Settings()
    assert s.llm_api_key == ""
    assert s.mock_mode is True
    # base_url/model 不因 key 空而崩，仍回退默认
    assert s.llm_base_url == "http://localhost:11434/v1"
    assert s.llm_model == "qwen2.5"


def test_explicit_llm_config_respected(monkeypatch):
    """显式配置正常读取，不被回退逻辑误伤。"""
    monkeypatch.setenv("LLM_API_KEY", "sk-x")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    s = Settings()
    assert s.llm_base_url == "https://api.example.com/v1"
    assert s.llm_model == "gpt-4o"
    assert s.llm_fast_model == "gpt-4o"
    assert s.mock_mode is False


def test_whitespace_only_llm_config_falls_back(monkeypatch):
    """纯空白串也视为未设 → 回退默认。"""
    monkeypatch.setenv("LLM_API_KEY", "sk-x")
    monkeypatch.setenv("LLM_BASE_URL", "   ")
    monkeypatch.setenv("LLM_MODEL", "\t")
    s = Settings()
    assert s.llm_base_url == "http://localhost:11434/v1"
    assert s.llm_model == "qwen2.5"


def test_vision_llm_empty_falls_back(monkeypatch):
    """视觉 LLM 端点/模型空串也回退默认；api_key 空保持禁用语义。"""
    monkeypatch.setenv("VISION_LLM_BASE_URL", "")
    monkeypatch.setenv("VISION_LLM_MODEL", "")
    monkeypatch.setenv("VISION_LLM_API_KEY", "")
    s = Settings()
    assert s.vision_llm_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert s.vision_llm_model == "qwen-vl-max-latest"
    assert s.vision_llm_api_key == ""
    assert s.vision_enabled is False
