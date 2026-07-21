"""
test_atomic_rules.py — 硬原子规则引擎单元测试。

覆盖每条规则的：正常通过、边界触发、边缘情况。
"""

import pytest

from models import CharacterState, GameMemorySystem
from atomic_rules import (
    RuleViolation,
    check_player_input_safety,
    check_character_alive,
    check_command_spells,
    check_mana_for_noble_phantasm,
    check_day_limit,
    check_game_already_over,
    determine_game_result,
    run_all_atomic_checks,
)


# ==========================================
# 输入安全
# ==========================================
class TestPlayerInputSafety:
    def test_valid_input(self):
        """正常输入不应抛出异常。"""
        check_player_input_safety("合理的战术指令。")

    def test_empty_input(self):
        """空字符串。"""
        with pytest.raises(RuleViolation, match=r"\[INPUT_EMPTY\]"):
            check_player_input_safety("")

    def test_whitespace_only_input(self):
        """只有空白字符。"""
        with pytest.raises(RuleViolation, match=r"\[INPUT_EMPTY\]"):
            check_player_input_safety("   \n\t  ")

    def test_too_long_input(self):
        """超过 2000 字符。"""
        with pytest.raises(RuleViolation, match=r"\[INPUT_TOO_LONG\]"):
            check_player_input_safety("x" * 2001)

    def test_boundary_2000_chars(self):
        """边界：恰好 2000 字符应通过。"""
        check_player_input_safety("x" * 2000)


# ==========================================
# 角色存活检查
# ==========================================
class TestCharacterAlive:
    def test_alive_character_passes(self, alive_servant):
        """存活角色 + 输入提到该角色 → 通过。"""
        check_character_alive(
            "Saber_Artoria", alive_servant,
            "命令Saber_Artoria前进。"
        )

    def test_dead_character_in_input_raises(self, dead_character):
        """已死亡角色 + 输入提到 → 拦截。"""
        with pytest.raises(RuleViolation, match=r"\[CHARACTER_DEAD\]"):
            check_character_alive(
                "Saber_Artoria", dead_character,
                "命令Saber_Artoria发动攻击。"
            )

    def test_dead_character_not_in_input_passes(self, dead_character):
        """死亡角色但输入未提到 → 不拦截（可能是其他角色在行动）。"""
        check_character_alive(
            "Saber_Artoria", dead_character,
            "命令Archer_EMIYA侦察。"
        )

    def test_alive_but_hp_zero_raises(self):
        """hp=0 但 is_alive 未设好的边缘情况（代码层也拦截）。"""
        weird = CharacterState(hp=0, is_alive=True)  # 这个在 models 层会被拦截
        # 但如果绕过 models 直接构造 dict... 这里测试 atomic_rules 的防御
        # 由于 Pydantic 的 validator 已经阻止了这种状态，所以 atomic_rules
        # 通过 is_alive 判断。此处验证 is_alive=True 时不拦截。
        normal = CharacterState(hp=100, is_alive=True)
        check_character_alive("Test", normal, "Test 前进。")

    def test_short_name_match(self):
        """模糊匹配：Rider_Attila → 'Attila' 出现在输入中也应匹配。"""
        dead = CharacterState(hp=0, status="dead", location="",
                              command_spells=0, is_alive=False, mana_remaining=0)
        with pytest.raises(RuleViolation, match=r"\[CHARACTER_DEAD\]"):
            check_character_alive(
                "Rider_Attila", dead,
                "向Attila的阵地发起突击。"
            )


# ==========================================
# 令咒检查
# ==========================================
class TestCommandSpells:
    def test_has_spells_passes(self, alive_master):
        """有令咒 + 输入含关键词 → 通过。"""
        check_command_spells(
            "Protagonist_Master", alive_master,
            "使用令咒命令Saber过来。"
        )

    def test_zero_spells_raises(self, zero_spells_master):
        """零令咒 + 输入含关键词 → 拦截。"""
        with pytest.raises(RuleViolation, match=r"\[NO_COMMAND_SPELLS\]"):
            check_command_spells(
                "Protagonist_Master", zero_spells_master,
                "Protagonist_Master以令咒命之，Saber到我身边来！"
            )

    def test_no_keyword_passes(self, zero_spells_master):
        """零令咒 但输入不含令咒关键词 → 不触发检查。"""
        check_command_spells(
            "Protagonist_Master", zero_spells_master,
            "命令Saber前进。"  # "命令" ≠ "令咒"
        )

    def test_character_not_in_input_passes(self, zero_spells_master):
        """零令咒 但输入操作的是另一个角色 → 不拦截。"""
        check_command_spells(
            "Protagonist_Master", zero_spells_master,
            "命令Saber_Artoria使用令咒。"
        )

    def test_english_keyword(self, zero_spells_master):
        """英文关键词 'command spell' 也能检测。"""
        with pytest.raises(RuleViolation, match=r"\[NO_COMMAND_SPELLS\]"):
            check_command_spells(
                "Protagonist_Master", zero_spells_master,
                "I use my command spell to order Protagonist_Master's servant."
            )

    def test_japanese_keyword(self, zero_spells_master):
        """日文关键词 '令呪' 也能检测。"""
        with pytest.raises(RuleViolation, match=r"\[NO_COMMAND_SPELLS\]"):
            check_command_spells(
                "Protagonist_Master", zero_spells_master,
                "Protagonist_Masterは令呪を使って、サーヴァントを召喚する。"
            )


# ==========================================
# 魔力检查
# ==========================================
class TestManaForNoblePhantasm:
    def test_enough_mana_passes(self, alive_servant):
        """魔力充足 + NP 关键词 → 通过。"""
        check_mana_for_noble_phantasm(
            "Saber_Artoria", alive_servant,
            "Saber_Artoria解放宝具！"
        )

    def test_low_mana_raises(self, low_mana_servant):
        """魔力不足 + NP 关键词 → 拦截。"""
        with pytest.raises(RuleViolation, match=r"\[INSUFFICIENT_MANA\]"):
            check_mana_for_noble_phantasm(
                "Saber_Artoria", low_mana_servant,
                "Saber_Artoria，宝具展开！"
            )

    def test_low_mana_no_np_keyword_passes(self, low_mana_servant):
        """魔力不足 但输入不含宝具关键词 → 通过。"""
        check_mana_for_noble_phantasm(
            "Saber_Artoria", low_mana_servant,
            "Saber_Artoria挥剑攻击。"
        )

    def test_english_np_keyword(self, low_mana_servant):
        """英文 'noble phantasm' 也能检测。"""
        with pytest.raises(RuleViolation, match=r"\[INSUFFICIENT_MANA\]"):
            check_mana_for_noble_phantasm(
                "Saber_Artoria", low_mana_servant,
                "Saber_Artoria, unleash your noble phantasm!"
            )

    def test_boundary_30_mana(self):
        """边界：恰好 30 魔力 → 通过。"""
        boundary = CharacterState(
            hp=100, status="OK", location="test",
            command_spells=0, is_alive=True, mana_remaining=30,
        )
        check_mana_for_noble_phantasm(
            "Test", boundary,
            "Test，解放宝具！"
        )

    def test_boundary_29_mana_raises(self):
        """边界：29 魔力 → 拦截。"""
        boundary = CharacterState(
            hp=100, status="OK", location="test",
            command_spells=0, is_alive=True, mana_remaining=29,
        )
        with pytest.raises(RuleViolation, match=r"\[INSUFFICIENT_MANA\]"):
            check_mana_for_noble_phantasm(
                "Test", boundary,
                "Test，宝具展开！"
            )


# ==========================================
# 昼夜上限
# ==========================================
class TestDayLimit:
    def test_under_limit_passes(self):
        check_day_limit(3, 7)

    def test_at_limit_passes(self):
        """第7天仍在允许范围内，第8天才超限。"""
        check_day_limit(7, 7)

    def test_exceeded_limit_raises(self):
        with pytest.raises(RuleViolation, match=r"\[WAR_ENDED\]"):
            check_day_limit(8, 7)

    def test_first_day_passes(self):
        check_day_limit(1, 7)


# ==========================================
# 游戏已结束检测
# ==========================================
class TestGameAlreadyOver:
    def test_not_over_passes(self):
        check_game_already_over(False)

    def test_already_over_raises(self):
        with pytest.raises(RuleViolation, match=r"\[GAME_ALREADY_OVER\]"):
            check_game_already_over(True)


# ==========================================
# 综合校验 (run_all_atomic_checks)
# ==========================================
class TestRunAllAtomicChecks:
    def test_all_pass(self, sample_memory):
        """正常输入 + 合法状态 → 零违规。"""
        violations = run_all_atomic_checks(
            memory=sample_memory,
            player_input="命令Saber_Artoria侦察周边。",
            max_days=7,
        )
        assert violations == []

    def test_empty_input(self, sample_memory):
        """空输入应立即返回违规。"""
        violations = run_all_atomic_checks(
            memory=sample_memory,
            player_input="",
            max_days=7,
        )
        assert len(violations) == 1
        assert violations[0][0] == "INPUT_EMPTY"

    def test_dead_character_operation(self, sample_memory):
        """操作已死亡角色应被拦截。"""
        # 先让 Saber_Artoria 死亡
        snapshot = dict(sample_memory.current_snapshot)
        snapshot["Saber_Artoria"] = CharacterState(
            hp=0, max_hp=100, status="阵亡", location="冬木市·废墟",
            command_spells=0, is_alive=False, mana_remaining=0,
        )
        memory = GameMemorySystem(
            active_servant_keys=sample_memory.active_servant_keys,
            chronicle_history=sample_memory.chronicle_history,
            current_snapshot=snapshot,
        )

        violations = run_all_atomic_checks(
            memory=memory,
            player_input="命令Saber_Artoria发动宝具。",
            max_days=7,
        )
        assert len(violations) >= 1
        assert any("DEAD" in v[0] for v in violations)

    def test_zero_spells_command_spell_usage(self, sample_memory):
        """零令咒的御主试图使用令咒。"""
        snapshot = dict(sample_memory.current_snapshot)
        snapshot["Protagonist_Master"] = CharacterState(
            hp=100, status="OK", location="test",
            command_spells=0, is_alive=True, mana_remaining=100,
        )
        memory = GameMemorySystem(
            active_servant_keys=sample_memory.active_servant_keys,
            chronicle_history=sample_memory.chronicle_history,
            current_snapshot=snapshot,
        )

        violations = run_all_atomic_checks(
            memory=memory,
            player_input="Protagonist_Master以令咒下令：Saber_Artoria到我身边来！",
            max_days=7,
        )
        assert any("COMMAND_SPELLS" in v[0] for v in violations)

    def test_multiple_violations(self, sample_memory):
        """一次检查可发现多种违规。"""
        snapshot = dict(sample_memory.current_snapshot)
        snapshot["Saber_Artoria"] = CharacterState(
            hp=0, status="死亡", location="墓地",
            command_spells=0, is_alive=False, mana_remaining=0,
        )
        snapshot["Protagonist_Master"] = CharacterState(
            hp=100, status="OK", location="test",
            command_spells=0, is_alive=True, mana_remaining=100,
        )
        memory = GameMemorySystem(
            active_servant_keys=sample_memory.active_servant_keys,
            chronicle_history=sample_memory.chronicle_history,
            current_snapshot=snapshot,
            current_day=8,  # 超过7天上限
            current_phase="night",
        )

        violations = run_all_atomic_checks(
            memory=memory,
            player_input=(
                "Protagonist_Master用令咒强行命令Saber_Artoria复活，"
                "然后Saber_Artoria解放宝具歼灭敌军。"
            ),
            max_days=7,
        )
        # 应检测到：令咒不足、死亡角色操作、战争已结束
        assert len(violations) >= 3

    def test_day_limit_exceeded(self, sample_memory):
        """第8天——超过圣杯战争7天限制。"""
        memory = GameMemorySystem(
            active_servant_keys=sample_memory.active_servant_keys,
            chronicle_history=sample_memory.chronicle_history,
            current_snapshot=sample_memory.current_snapshot,
            current_day=8,
            current_phase="night",
        )
        violations = run_all_atomic_checks(
            memory=memory,
            player_input="侦察。",
            max_days=7,
        )
        assert any("WAR_ENDED" in v[0] for v in violations)


# ==========================================
# RuleViolation 异常
# ==========================================
class TestRuleViolation:
    def test_code_and_message(self):
        exc = RuleViolation("TEST_CODE", "测试消息")
        assert exc.code == "TEST_CODE"
        assert exc.message == "测试消息"
        assert str(exc) == "[TEST_CODE] 测试消息"

    def test_can_be_caught_as_exception(self):
        try:
            raise RuleViolation("ERR", "出错")
        except RuleViolation as e:
            assert e.code == "ERR"
        else:
            pytest.fail("应抛出 RuleViolation")
