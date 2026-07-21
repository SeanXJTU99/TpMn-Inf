"""
test_config.py — 配置管理单元测试。
"""

import os
import pytest
from dataclasses import asdict

from config import Config


class TestConfigDefaults:
    """默认值测试。"""

    def setup_method(self):
        self.config = Config()

    def test_deepseek_base_url(self):
        assert self.config.DEEPSEEK_BASE_URL == "https://api.deepseek.com/v1"

    def test_ollama_base_url(self):
        assert self.config.OLLAMA_BASE_URL == "http://localhost:11434/v1"

    def test_ollama_router_model(self):
        assert self.config.OLLAMA_ROUTER_MODEL == "qwen2.5:3b"

    def test_ollama_timeout(self):
        assert self.config.OLLAMA_TIMEOUT_SEC == 5.0

    def test_tier_threshold(self):
        assert self.config.TIER_THRESHOLD == 9

    def test_servants_per_game(self):
        assert self.config.SERVANTS_PER_GAME == 7

    def test_max_days(self):
        assert self.config.MAX_DAYS == 7

    def test_session_ttl(self):
        assert self.config.SESSION_TTL_MINUTES == 120

    def test_temperatures(self):
        assert self.config.ROUTER_TEMPERATURE == 0.1
        assert self.config.ARBITER_TEMPERATURE == 0.0
        assert self.config.NARRATOR_TEMPERATURE == 0.7

    def test_server_defaults(self):
        assert self.config.HOST == "0.0.0.0"
        assert self.config.PORT == 8000

    def test_max_output_tokens(self):
        assert self.config.MAX_OUTPUT_TOKENS == 16384


class TestConfigValidation:
    """validate() 方法测试。"""

    def test_missing_api_key(self):
        """没有 API Key 时应报告。"""
        # 确保环境变量未设置
        old = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            c = Config(DEEPSEEK_API_KEY="")
            issues = c.validate()
            assert len(issues) >= 1
            assert any("DEEPSEEK_API_KEY" in i for i in issues)
        finally:
            if old:
                os.environ["DEEPSEEK_API_KEY"] = old

    def test_with_api_key(self):
        """有 API Key 时 validate 应返回空列表。"""
        c = Config(DEEPSEEK_API_KEY="sk-test-key")
        issues = c.validate()
        assert issues == []


class TestConfigCustomValues:
    """自定义值测试。"""

    def test_custom_model_names(self):
        c = Config(
            ARBITER_LOW_TIER_MODEL="custom-model",
            ARBITER_HIGH_TIER_MODEL="custom-reasoner",
        )
        assert c.ARBITER_LOW_TIER_MODEL == "custom-model"
        assert c.ARBITER_HIGH_TIER_MODEL == "custom-reasoner"

    def test_custom_threshold(self):
        c = Config(TIER_THRESHOLD=7)
        assert c.TIER_THRESHOLD == 7

    def test_from_env_var(self):
        """环境变量应被读取。"""
        os.environ["DEEPSEEK_API_KEY"] = "sk-env-test"
        try:
            c = Config()  # 重新实例化以读取新的环境变量
            assert c.DEEPSEEK_API_KEY == "sk-env-test"
        finally:
            del os.environ["DEEPSEEK_API_KEY"]
