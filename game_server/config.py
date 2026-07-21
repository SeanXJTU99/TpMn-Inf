"""
《万能愿望机：残响协议》— 集中配置中心
==============================================
所有模型名称、API 密钥、阈值参数集中管理，一键切换。
"""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # ==========================================
    # 版本
    # ==========================================
    VERSION: str = "3.2.0"

    # ==========================================
    # API 密钥
    # ==========================================
    DEEPSEEK_API_KEY: str = field(
        default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", "")
    )

    # ==========================================
    # DeepSeek API 端点
    # ==========================================
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"

    # ==========================================
    # Ollama 本地模型配置
    # ==========================================
    OLLAMA_BASE_URL: str = "http://localhost:11434/v1"
    OLLAMA_ROUTER_MODEL: str = "qwen2.5:3b"       # 本地路由打分模型
    OLLAMA_TIMEOUT_SEC: float = 5.0                 # 超时秒数，超时自动降级

    # ==========================================
    # 云端 AI 模型分配（DeepSeek V4 系列）
    # ==========================================
    # deepseek-v4-flash: 快速/经济型 (284B total, 13B active), 1M ctx
    # deepseek-v4-pro:   旗舰型 (1.6T total, 49B active), 1M ctx, 最强推理
    #
    # 注: deepseek-chat / deepseek-reasoner 是旧别名，2026-07-24 停用
    #
    # 路由层（Ollama 不可用时的降级后备）
    ROUTER_FALLBACK_MODEL: str = "deepseek-v4-flash"

    # 裁判层 — 常规战术（复杂度 1-8）。主裁判是游戏核心，全程使用最强模型。
    ARBITER_LOW_TIER_MODEL: str = "deepseek-v4-pro"

    # 裁判层 — 掀桌脑洞（复杂度 9-10，旗舰推理）
    ARBITER_HIGH_TIER_MODEL: str = "deepseek-v4-pro"

    # 说书人 — 文学渲染（长文生成，用 Flash 性价比高）
    NARRATOR_MODEL: str = "deepseek-v4-flash"

    # ==========================================
    # 分级阈值
    # ==========================================
    TIER_THRESHOLD: int = 9  # >= 9 分触发至高裁判升舱

    # ==========================================
    # 游戏参数
    # ==========================================
    SERVANTS_PER_GAME: int = 7           # 每局抽取英灵数
    MAX_DAYS: int = 7                    # 游戏最大昼夜数（7天后无赢家=平局）
    SESSION_TTL_MINUTES: int = 120       # Session 过期时间

    # ==========================================
    # AI 生成参数
    # ==========================================
    ROUTER_TEMPERATURE: float = 0.1
    ARBITER_TEMPERATURE: float = 0.0     # 裁判绝对零度
    NARRATOR_TEMPERATURE: float = 0.7    # 说书人文学释放
    MAX_OUTPUT_TOKENS: int = 16384  # 裁判需输出 14 角色完整快照 + 中文判定报告，4K 不够

    # ==========================================
    # 服务器
    # ==========================================
    HOST: str = "0.0.0.0"  # 监听所有网卡，手机可通过局域网访问
    PORT: int = 8000

    def validate(self) -> list[str]:
        """启动时自检，返回缺失项列表"""
        issues = []
        if not self.DEEPSEEK_API_KEY:
            issues.append("DEEPSEEK_API_KEY 未设置 — 云端 AI 将不可用")
        return issues


# 全局单例
config = Config()
