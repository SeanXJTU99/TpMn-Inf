"""
测试共享 fixtures 和配置。
"""

import sys
import os
import pytest

# 确保项目根目录在 Python path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    CharacterState,
    GameMemorySystem,
    GameTurnRequest,
    GameInitRequest,
)


# ==========================================
# CharacterState fixtures
# ==========================================
@pytest.fixture
def alive_master() -> CharacterState:
    """一个满状态的御主。"""
    return CharacterState(
        hp=100,
        max_hp=100,
        status="完美健康",
        location="冬木市·安全屋",
        command_spells=3,
        is_alive=True,
        mana_remaining=100,
    )


@pytest.fixture
def alive_servant() -> CharacterState:
    """一个满状态的英灵。"""
    return CharacterState(
        hp=100,
        max_hp=100,
        status="巅峰状态",
        location="冬木市·郊外森林",
        command_spells=0,  # 英灵无令咒
        is_alive=True,
        mana_remaining=100,
    )


@pytest.fixture
def dead_character() -> CharacterState:
    """一个已死亡的角色。"""
    return CharacterState(
        hp=0,
        max_hp=100,
        status="心脏被刺穿，当场死亡",
        location="冬木市·废墟",
        command_spells=0,
        is_alive=False,
        mana_remaining=0,
    )


@pytest.fixture
def low_mana_servant() -> CharacterState:
    """魔力不足的英灵。"""
    return CharacterState(
        hp=80,
        max_hp=100,
        status="轻伤·魔力枯竭",
        location="冬木市·教会",
        command_spells=0,
        is_alive=True,
        mana_remaining=15,
    )


@pytest.fixture
def zero_spells_master() -> CharacterState:
    """令咒耗尽的御主。"""
    return CharacterState(
        hp=90,
        max_hp=100,
        status="轻微擦伤",
        location="冬木市·桥头",
        command_spells=0,
        is_alive=True,
        mana_remaining=80,
    )


# ==========================================
# GameMemorySystem fixtures
# ==========================================
@pytest.fixture
def sample_memory(alive_master, alive_servant) -> GameMemorySystem:
    """一个典型的游戏记忆系统（第1天·夜）。"""
    return GameMemorySystem(
        active_servant_keys=["Saber_Artoria", "Archer_EMIYA"],
        chronicle_history=[
            "游戏开始：第五次冬木圣杯战争开幕。",
        ],
        current_snapshot={
            "Protagonist_Master": alive_master,
            "Saber_Artoria": alive_servant,
            "Archer_EMIYA": CharacterState(
                hp=100, max_hp=100, status="满状态·单独行动中",
                location="冬木市·远坂宅屋顶", command_spells=0,
                is_alive=True, mana_remaining=90,
            ),
        },
        current_day=1,
        current_phase="night",
    )


@pytest.fixture
def empty_memory() -> GameMemorySystem:
    """一个最小化的记忆系统（第1天·夜）。"""
    return GameMemorySystem(
        active_servant_keys=["Saber_Artoria"],
        chronicle_history=["游戏开始。"],
        current_snapshot={
            "Protagonist_Master": CharacterState(
                command_spells=3, location="冬木市·初始点"
            ),
            "Saber_Artoria": CharacterState(
                command_spells=0, location="冬木市·初始点"
            ),
        },
        current_day=1,
        current_phase="night",
    )


# ==========================================
# Request fixtures
# ==========================================
@pytest.fixture
def game_turn_request() -> GameTurnRequest:
    """一个标准的回合请求。"""
    return GameTurnRequest(
        session_id="test-session-001",
        player_input="命令Saber_Artoria侦察周边区域。",
    )


@pytest.fixture
def game_init_request() -> GameInitRequest:
    """一个游戏初始化请求（无偏好）。"""
    return GameInitRequest(preferred_servants=None)


@pytest.fixture
def game_init_request_preferred() -> GameInitRequest:
    """一个带偏好的初始化请求。"""
    return GameInitRequest(
        preferred_servants=["Saber_Artoria", "Archer_Gilgamesh"]
    )
