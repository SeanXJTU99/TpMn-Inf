"""
test_models.py — Pydantic 强类型数据模型单元测试。

覆盖：
  - CharacterState 默认值、边界值、类型校验
  - extra="forbid" 防 AI 私加字段
  - GameMemorySystem 逻辑一致性校验
  - 各 API Request/Response 模型的构造
"""

import pytest
from pydantic import ValidationError

from models import (
    CharacterState,
    GameMemorySystem,
    GameTurnRequest,
    GameInitRequest,
    GameInitResponse,
    RouterAssessment,
    ArbiterJudgment,
    EngineFinalResponse,
    SessionInfo,
    HealthResponse,
)


# ==========================================
# CharacterState
# ==========================================
class TestCharacterState:
    """CharacterState 模型单元测试。"""

    def test_default_values(self):
        """默认值应正确。"""
        cs = CharacterState()
        assert cs.hp == 100
        assert cs.max_hp == 100
        assert cs.status == "待命中"
        assert cs.location == "城区·未知区域"
        assert cs.command_spells == 0
        assert cs.is_alive is True
        assert cs.mana_remaining == 100

    def test_full_construction(self):
        """完整构造应保留所有字段。"""
        cs = CharacterState(
            hp=75,
            max_hp=100,
            status="右臂骨折·轻伤",
            location="城区·港口仓库",
            command_spells=2,
            is_alive=True,
            mana_remaining=60,
        )
        assert cs.hp == 75
        assert cs.status == "右臂骨折·轻伤"
        assert cs.command_spells == 2

    def test_hp_lower_bound(self):
        """hp 最小值 0。"""
        cs = CharacterState(hp=0, is_alive=False)
        assert cs.hp == 0

    def test_hp_negative_raises(self):
        """hp 不能为负数。"""
        with pytest.raises(ValidationError):
            CharacterState(hp=-1)

    def test_hp_exceeds_100_raises(self):
        """hp 不能超过 100。"""
        with pytest.raises(ValidationError):
            CharacterState(hp=101)

    def test_command_spells_negative_raises(self):
        """令咒数不能为负。"""
        with pytest.raises(ValidationError):
            CharacterState(command_spells=-1)

    def test_command_spells_exceeds_3_raises(self):
        """令咒数不能超过 3。"""
        with pytest.raises(ValidationError):
            CharacterState(command_spells=4)

    def test_mana_remaining_bounds(self):
        """魔力在 0-100 之间。"""
        cs = CharacterState(mana_remaining=0)
        assert cs.mana_remaining == 0

        cs = CharacterState(mana_remaining=100)
        assert cs.mana_remaining == 100

        with pytest.raises(ValidationError):
            CharacterState(mana_remaining=-1)

        with pytest.raises(ValidationError):
            CharacterState(mana_remaining=101)

    def test_extra_fields_forbidden(self):
        """extra='forbid' —— AI 不能私加字段（防 key 拼错）。"""
        with pytest.raises(ValidationError):
            CharacterState(
                hp=100,
                health_points=100,  # ❌ 拼错 —— 应该被拦截
            )

        with pytest.raises(ValidationError):
            CharacterState(
                hp=100,
                unknown_field="AI 的幻觉输出",  # ❌ 擅自加字段
            )

    def test_hp_coerces_string_to_int(self):
        """Pydantic v2 默认宽松模式：字符串 '100' 自动转换为整数 100。
        这对 AI 输出的容错性有利——AI 偶尔输出字符串格式的数值也不会崩溃。"""
        cs = CharacterState(hp="100")
        assert cs.hp == 100
        assert isinstance(cs.hp, int)

    def test_is_alive_coerces_string_to_bool(self):
        """Pydantic v2 默认宽松模式：'true'/'false' 自动转换为 bool。"""
        cs = CharacterState(is_alive="true")
        assert cs.is_alive is True
        cs2 = CharacterState(is_alive="false")
        assert cs2.is_alive is False

    def test_serialize_deserialize_roundtrip(self):
        """model_dump → model_validate 来回一致。"""
        original = CharacterState(
            hp=50,
            max_hp=80,  # 重伤导致上限降低
            status="重伤·魔力回路受损",
            location="城区·地下室",
            command_spells=1,
            is_alive=True,
            mana_remaining=30,
        )
        data = original.model_dump()
        restored = CharacterState.model_validate(data)
        assert restored == original

    def test_servant_has_zero_command_spells_by_default(self):
        """英灵默认令咒为 0。"""
        servant = CharacterState()
        assert servant.command_spells == 0

    def test_master_can_have_three_spells(self):
        """御主可以有 3 划令咒。"""
        master = CharacterState(command_spells=3)
        assert master.command_spells == 3


# ==========================================
# GameMemorySystem
# ==========================================
class TestGameMemorySystem:
    """GameMemorySystem 模型单元测试。"""

    def test_minimal_construction(self, empty_memory):
        """最小化构造。"""
        assert len(empty_memory.active_servant_keys) == 1
        assert len(empty_memory.chronicle_history) == 1
        assert len(empty_memory.current_snapshot) == 2

    def test_empty_active_servant_keys_raises(self):
        """active_servant_keys 不能为空。"""
        with pytest.raises(ValidationError):
            GameMemorySystem(
                active_servant_keys=[],
                chronicle_history=["start"],
                current_snapshot={},
            )

    def test_too_many_servant_keys_raises(self):
        """最多 7 个英灵。"""
        with pytest.raises(ValidationError):
            GameMemorySystem(
                active_servant_keys=[f"Servant_{i}" for i in range(8)],
                chronicle_history=["start"],
                current_snapshot={},
            )

    def test_hp_zero_but_alive_raises(self):
        """逻辑矛盾：hp=0 但 is_alive=True 应报错。"""
        with pytest.raises(ValidationError, match="逻辑矛盾"):
            GameMemorySystem(
                active_servant_keys=["Saber_Artoria"],
                chronicle_history=["start"],
                current_snapshot={
                    "Saber_Artoria": CharacterState(
                        hp=0, is_alive=True  # ← 矛盾
                    ),
                },
            )

    def test_negative_command_spells_caught_by_character_state(self):
        """负令咒数直接在 CharacterState 字段校验层拦截（ge=0），
        无需到达 GameMemorySystem 的 model_validator。"""
        with pytest.raises(ValidationError):
            CharacterState(command_spells=-1)

    def test_roundtrip(self, sample_memory):
        """序列化-反序列化来回一致。"""
        data = sample_memory.model_dump()
        restored = GameMemorySystem.model_validate(data)
        assert restored == sample_memory

    def test_chronicle_history_is_list_of_strings(self):
        """编年史必须是字符串列表。"""
        with pytest.raises(ValidationError):
            GameMemorySystem(
                active_servant_keys=["Saber_Artoria"],
                chronicle_history=[123],  # 不是字符串
                current_snapshot={
                    "Saber_Artoria": CharacterState(),
                },
            )


# ==========================================
# GameTurnRequest
# ==========================================
class TestGameTurnRequest:
    """回合请求模型测试。"""

    def test_valid_request(self):
        req = GameTurnRequest(
            session_id="abc12345",
            player_input="侦察周围环境。",
        )
        assert req.session_id == "abc12345"
        assert req.player_input == "侦察周围环境。"

    def test_empty_player_input_raises(self):
        """空指令应拒绝。"""
        with pytest.raises(ValidationError):
            GameTurnRequest(session_id="abc", player_input="")

    def test_too_long_player_input_raises(self):
        """超过 2000 字符的指令应拒绝。"""
        with pytest.raises(ValidationError):
            GameTurnRequest(session_id="abc", player_input="x" * 2001)

    def test_missing_session_id_raises(self):
        """session_id 必填。"""
        with pytest.raises(ValidationError):
            GameTurnRequest(player_input="test")  # type: ignore


# ==========================================
# GameInitRequest
# ==========================================
class TestGameInitRequest:
    def test_empty_init(self):
        req = GameInitRequest()
        assert req.preferred_servants is None

    def test_with_preferences(self):
        req = GameInitRequest(preferred_servants=["Saber_Artoria"])
        assert req.preferred_servants == ["Saber_Artoria"]

    def test_too_many_preferences_raises(self):
        with pytest.raises(ValidationError):
            GameInitRequest(preferred_servants=[f"S_{i}" for i in range(8)])


# ==========================================
# Other response models
# ==========================================
class TestResponseModels:
    def test_router_assessment(self):
        ra = RouterAssessment(
            complexity_score=7,
            reason="中等复杂度战术",
            router_source="ollama",
        )
        assert ra.complexity_score == 7
        assert ra.router_source == "ollama"

    def test_router_assessment_bounds(self):
        """复杂度必须在 1-10。"""
        with pytest.raises(ValidationError):
            RouterAssessment(complexity_score=0, reason="", router_source="ollama")
        with pytest.raises(ValidationError):
            RouterAssessment(complexity_score=11, reason="", router_source="ollama")

    def test_arbiter_judgment(self, sample_memory):
        aj = ArbiterJudgment(
            judgment_report="判定通过。",
            updated_memory_system=sample_memory,
            arbiter_model="deepseek-chat",
        )
        assert aj.arbiter_model == "deepseek-chat"

    def test_engine_final_response(self, sample_memory):
        resp = EngineFinalResponse(
            narrative="game的血与火之中...",
            memory_system=sample_memory,
            turn_summary={"complexity_score": 5},
        )
        assert "game" in resp.narrative

    def test_health_response(self):
        hr = HealthResponse(
            status="ok",
            ollama_available=True,
            deepseek_configured=True,
            servant_count=15,
        )
        assert hr.ollama_available is True
        assert hr.servant_count == 15

    def test_session_info(self):
        si = SessionInfo(
            session_id="test-01",
            turn_count=5,
            active_servant_keys=["Saber_Artoria"],
            created_at="2026-06-15T10:00:00",
            last_turn_at="2026-06-15T10:05:00",
        )
        assert si.turn_count == 5
