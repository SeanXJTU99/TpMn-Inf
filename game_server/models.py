"""
《万能愿望机：残响协议》— Pydantic 强类型数据模型
====================================================
杜绝 AI 拼错 key 导致的运行时崩溃。
所有状态字段均有明确物理含义与数值约束。
"""

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, model_validator, ConfigDict


# ==========================================
# 核心角色状态（替代泛型 Dict[str, Any]）
# ==========================================
class CharacterState(BaseModel):
    """御主或英灵的运行时状态快照。

    AI 裁判输出的所有 key 在此被强制校验 ——
    拼错 key、类型错误、数值越界均会在 model_validate_json 阶段被拦截。
    """
    hp: int = Field(
        default=100,
        ge=0,
        le=100,
        description="当前生命值百分比。0=濒死/无法行动，100=满血。"
    )
    max_hp: int = Field(
        default=100,
        ge=0,
        le=100,
        description="最大生命值百分比。通常为 100，重伤后可能永久降低。"
    )
    status: str = Field(
        default="待命中",
        description="当前生理/精神状态简述，如 '轻伤·右臂骨折'、'魔力枯竭'、'满状态'。"
    )
    location: str = Field(
        default="冬木市·未知区域",
        description="当前所在地点，用于空间逻辑判定（如移动时间、射程）。"
    )
    command_spells: int = Field(
        default=0,
        ge=0,
        le=3,
        description="剩余令咒次数。仅御主（Master）角色有此属性，英灵固定为 0。"
    )
    is_alive: bool = Field(
        default=True,
        description="是否存活。False = 已确认死亡/退场，后续回合绝对不可操作。"
    )
    mana_remaining: int = Field(
        default=100,
        ge=0,
        le=100,
        description="剩余魔力百分比。英灵释放宝具需消耗大量魔力。"
    )

    model_config = ConfigDict(extra="forbid")  # 禁止 AI 自行添加不在模型中的字段


# ==========================================
# 双轨制记忆系统
# ==========================================
class GameMemorySystem(BaseModel):
    """完整的双轨制游戏记忆。

    - chronicle_history: 不可篡改的历史铁事实（追加式）
    - current_snapshot: 实时动态状态（每回合覆写式）
    - current_day / current_phase: 圣杯战争时间轴（7个昼夜后强制结束）
    """
    active_servant_keys: List[str] = Field(
        default_factory=list,
        description="本局游戏抽中并存活/激活的英灵 ID 列表。",
        min_length=1,
        max_length=7,
    )
    chronicle_history: List[str] = Field(
        default_factory=list,
        description="【轨道一：编年史】按时间轴叠加的不可篡改历史事实。只增不删。"
    )
    current_snapshot: Dict[str, CharacterState] = Field(
        default_factory=dict,
        description="【轨道二：动态快照】key=角色名（如 'Emiya_Kiritsugu'），value=运行时状态。"
    )
    current_day: int = Field(
        default=1,
        ge=1,
        description="当前是圣杯战争第几天（1-N）。超过 MAX_DAYS 时由硬原子规则强制终结。"
    )
    current_phase: Literal["day", "night"] = Field(
        default="night",
        description="当前昼夜阶段。圣杯战争从夜间开幕。day=白天（交涉·侦察），night=夜晚（战斗·暗杀）。"
    )

    @model_validator(mode="after")
    def validate_game_state(self):
        """跨字段逻辑一致性校验。"""
        for name, state in self.current_snapshot.items():
            if state.hp <= 0 and state.is_alive:
                raise ValueError(
                    f"逻辑矛盾：{name} 的 hp={state.hp} 但 is_alive=True。"
                    f"HP≤0 的角色必须标记为死亡。"
                )
        return self


# ==========================================
# API 请求 / 响应模型
# ==========================================
class GameInitRequest(BaseModel):
    """游戏初始化请求（可选：允许客户端指定偏好）"""
    preferred_servants: Optional[List[str]] = Field(
        default=None,
        max_length=7,
        description="可选偏好英灵 ID 列表（用于测试指定组合）"
    )


class GameInitResponse(BaseModel):
    """游戏初始化响应"""
    session_id: str = Field(description="本局唯一标识，后续所有请求需携带")
    active_servants: Dict[str, str] = Field(
        description="玩家契约的英灵（仅1个），key=ID, value=true_name。其余6骑身份隐匿。"
    )
    memory_system: GameMemorySystem = Field(description="初始记忆系统（空快照）")
    player_servant_key: str = Field(description="玩家契约英灵的数据库 key")
    player_servant_name: str = Field(description="玩家契约英灵的真名")
    message: str = Field(description="开局叙事引言")


class GameTurnRequest(BaseModel):
    """一回合推演请求"""
    session_id: str = Field(description="游戏 session ID（由 /init 返回）")
    player_input: str = Field(
        min_length=1,
        max_length=2000,
        description="玩家本回合输入的操作命令"
    )


class RouterAssessment(BaseModel):
    """路由打分结果"""
    complexity_score: int = Field(ge=1, le=10, description="复杂度打分 1-10")
    reason: str = Field(description="打分理由")
    router_source: str = Field(description="'ollama' 或 'deepseek'，标明打分来源")


class ArbiterJudgment(BaseModel):
    """裁判 AI 判定结果"""
    judgment_report: str = Field(description="冰冷的因果逻辑判定报告")
    updated_memory_system: GameMemorySystem = Field(description="更新后的完整记忆系统")
    arbiter_model: str = Field(description="实际使用的裁判模型名称")


class GameOverInfo(BaseModel):
    """游戏结束判定结果。"""
    is_over: bool = Field(default=False, description="游戏是否已结束")
    result: Literal["victory", "draw", "defeat"] = Field(
        default="draw",
        description="victory=玩家获胜 | draw=平局 | defeat=玩家败北"
    )
    winner_name: str = Field(
        default="",
        description="获胜方名称（result=victory 时）或空字符串（平局时）"
    )
    epilogue: str = Field(
        default="",
        description="结局摘要：谁存活、谁退场、圣杯是否显现。"
    )


class EngineFinalResponse(BaseModel):
    """引擎最终响应（发回前端）"""
    narrative: str = Field(description="给玩家看的暗黑剧情文本")
    memory_system: GameMemorySystem = Field(description="更新后的记忆数据，下一回合回传")
    turn_summary: dict = Field(
        default_factory=dict,
        description="本回合元数据：评分、模型选择、token 消耗等（调试用）"
    )
    game_over: Optional[GameOverInfo] = Field(
        default=None,
        description="游戏结束信息。null=继续，非null=本回合触发了结局。"
    )


class SessionInfo(BaseModel):
    """Session 信息（调试端点）"""
    session_id: str
    turn_count: int
    active_servant_keys: List[str]
    created_at: str
    last_turn_at: str


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    version: str = ""
    ollama_available: bool = False
    deepseek_configured: bool = False
    servant_count: int = 0
