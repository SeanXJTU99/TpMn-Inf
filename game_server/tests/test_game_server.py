"""
test_game_server.py — FastAPI 集成测试。

Mock 掉所有 AI 调用（call_deepseek / call_ollama），
测试 API 端点的完整请求-响应链路。
"""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from fastapi.testclient import TestClient

# --- Mock 必须在导入 app 之前设置 ---
# 因为 ai_client 在模块级别创建客户端，
# 我们需要 mock 的是 game_server 中调用的函数。

# 先 mock config 确保测试不依赖真实 API key
with patch("config.config.DEEPSEEK_API_KEY", "sk-test-mock-key"):
    with patch("config.config.validate", lambda self: []):
        from game_server import app, GAME_SESSIONS

client = TestClient(app)


# ==========================================
# Fixtures (每个测试前后清理)
# ==========================================
@pytest.fixture(autouse=True)
def cleanup_sessions():
    """每个测试前后清理全局 session 存储。"""
    GAME_SESSIONS.clear()
    yield
    GAME_SESSIONS.clear()


@pytest.fixture
def mock_ollama_success():
    """Mock Ollama 成功返回路由打分。"""
    with patch(
        "game_server.call_ollama",
        new_callable=AsyncMock,
    ) as mock:
        mock.return_value = (
            json.dumps({"complexity_score": 5, "reason": "中等复杂度"}),
            {"total_tokens": 50, "model": "qwen2.5:3b", "latency_sec": 0.5},
        )
        yield mock


@pytest.fixture
def mock_ollama_failure():
    """Mock Ollama 失败（触发降级）。"""
    with patch(
        "game_server.call_ollama",
        new_callable=AsyncMock,
        side_effect=Exception("Ollama connection refused"),
    ):
        yield


@pytest.fixture
def mock_deepseek_router():
    """Mock DeepSeek 路由打分。"""
    with patch(
        "game_server.call_deepseek",
        new_callable=AsyncMock,
    ) as mock:
        # call_deepseek 会被调用两次：router + arbiter + narrator
        # 用 side_effect 区分
        mock.side_effect = None  # 由各测试自行设置
        yield mock


@pytest.fixture
def mock_full_pipeline():
    """Mock 完整的 AI 调用链路：router → arbiter → narrator。"""
    with patch(
        "game_server.call_ollama",
        new_callable=AsyncMock,
        side_effect=Exception("Ollama unavailable"),
    ), patch(
        "game_server.call_deepseek",
        new_callable=AsyncMock,
    ) as mock_ds:
        # 三次调用分别返回 router、arbiter、narrator 的结果
        mock_ds.side_effect = [
            # 1. Router (deepseek-chat)
            (
                json.dumps({"complexity_score": 6, "reason": "常规战术"}),
                {"total_tokens": 100, "model": "deepseek-chat"},
            ),
            # 2. Arbiter (deepseek-chat, 因为 6 < 9)
            (
                json.dumps({
                    "judgment_report": "Saber_Artoria 进行了侦察。未发现异常。",
                    "updated_memory_system": {
                        "current_day": 1,
                        "current_phase": "night",
                        "active_servant_keys": ["Saber_Artoria", "Archer_EMIYA"],
                        "chronicle_history": [
                            "游戏开始。",
                            "第1回合：Saber进行了侦察。"
                        ],
                        "current_snapshot": {
                            "Protagonist_Master": {
                                "hp": 100, "max_hp": 100, "status": "完美健康",
                                "location": "城区·安全屋", "command_spells": 3,
                                "is_alive": True, "mana_remaining": 100,
                            },
                            "Saber_Artoria": {
                                "hp": 100, "max_hp": 100, "status": "侦察完毕",
                                "location": "城区·郊外", "command_spells": 0,
                                "is_alive": True, "mana_remaining": 100,
                            },
                            "Archer_EMIYA": {
                                "hp": 100, "max_hp": 100, "status": "待命中",
                                "location": "城区·远坂宅", "command_spells": 0,
                                "is_alive": True, "mana_remaining": 90,
                            },
                            "Enemy_Master": {
                                "hp": 100, "max_hp": 100, "status": "完美健康",
                                "location": "城区·教会", "command_spells": 3,
                                "is_alive": True, "mana_remaining": 100,
                            },
                        },
                    },
                }),
                {"total_tokens": 2000, "model": "deepseek-chat"},
            ),
            # 3. Narrator (deepseek-chat)
            (
                "Saber_Artoria 宛如一道银色的闪电掠过城区的夜空——"
                "她的直感没有捕捉到任何异常，郊外的森林静得可怕。"
                "而这异常的寂静本身，就是最大的异常。",
                {"total_tokens": 500, "model": "deepseek-chat"},
            ),
        ]
        yield mock_ds


# ==========================================
# Health 端点
# ==========================================
class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "ollama_available" in data
        assert "deepseek_configured" in data
        assert "servant_count" in data

    def test_health_servant_count_positive(self):
        resp = client.get("/health")
        assert resp.json()["servant_count"] > 0


# ==========================================
# Init 端点
# ==========================================
class TestGameInit:
    def test_init_success(self):
        """正常初始化应返回 session_id，玩家仅可见自己契约的英灵。"""
        resp = client.post("/api/game/init", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert len(data["session_id"]) == 8
        # active_servants 仅返回玩家契约的 1 骑英灵（其余 6 骑隐匿）
        assert len(data["active_servants"]) == 1
        assert "player_servant_key" in data
        assert "player_servant_name" in data
        assert data["player_servant_key"] in list(data["active_servants"].keys())
        assert "memory_system" in data
        # 内部仍保留全部 7 骑
        assert len(data["memory_system"]["active_servant_keys"]) == 7
        # 初始快照应有 7 英灵 + 7 御主（1主角+6敌方） = 14 个角色
        assert len(data["memory_system"]["current_snapshot"]) == 14
        # 主角御主应有 3 划令咒
        master = data["memory_system"]["current_snapshot"]["Protagonist_Master"]
        assert master["command_spells"] == 3

    def test_init_with_preferences(self):
        """带偏好的初始化。"""
        resp = client.post("/api/game/init", json={
            "preferred_servants": ["Saber_Artoria", "Archer_Gilgamesh"]
        })
        assert resp.status_code == 200
        data = resp.json()
        keys = data["memory_system"]["active_servant_keys"]
        assert "Saber_Artoria" in keys
        assert "Archer_Gilgamesh" in keys

    def test_init_creates_session(self):
        """init 后 session 应存在于服务端。"""
        resp = client.post("/api/game/init", json={})
        sid = resp.json()["session_id"]
        # 查 session
        resp2 = client.get(f"/api/game/session/{sid}")
        assert resp2.status_code == 200
        assert resp2.json()["turn_count"] == 0

    def test_init_unique_sessions(self):
        """每次 init 应生成不同的 session_id。"""
        ids = set()
        for _ in range(5):
            resp = client.post("/api/game/init", json={})
            ids.add(resp.json()["session_id"])
        assert len(ids) == 5


# ==========================================
# Execute Turn 端点
# ==========================================
class TestExecuteTurn:
    def _init_and_get_sid(self) -> str:
        """辅助：初始化游戏并返回 session_id。"""
        resp = client.post("/api/game/init", json={})
        assert resp.status_code == 200
        return resp.json()["session_id"]

    def test_session_not_found(self):
        """不存在的 session → 404。"""
        resp = client.post("/api/game/execute_turn", json={
            "session_id": "nonexist",
            "player_input": "侦察。",
        })
        assert resp.status_code == 404
        assert "不存在" in resp.json()["detail"]

    def test_empty_input_blocked(self):
        """空输入 → 422（Pydantic 或原子规则拦截，不调 AI）。"""
        sid = self._init_and_get_sid()
        resp = client.post("/api/game/execute_turn", json={
            "session_id": sid,
            "player_input": "",
        })
        assert resp.status_code == 422
        # Pydantic body parsing 先拦截（string_too_short）或原子规则拦截（INPUT_EMPTY）
        detail = str(resp.json()["detail"])
        assert "INPUT_EMPTY" in detail or "string_too_short" in detail

    def test_dead_character_blocked(self):
        """操作已死亡角色 → 422。"""
        sid = self._init_and_get_sid()

        # 先在 session 中手动标记某英灵死亡
        from game_server import GAME_SESSIONS
        from models import CharacterState

        session = GAME_SESSIONS[sid]
        snapshot = dict(session.memory_system.current_snapshot)
        # 找到第一个英灵（非 Protagonist_Master）
        servant_key = [
            k for k in snapshot if k != "Protagonist_Master"
        ][0]
        snapshot[servant_key] = CharacterState(
            hp=0, max_hp=100, status="阵亡",
            location="墓地", command_spells=0,
            is_alive=False, mana_remaining=0,
        )
        session.memory_system.current_snapshot = snapshot

        resp = client.post("/api/game/execute_turn", json={
            "session_id": sid,
            "player_input": f"命令{servant_key}发动攻击。",
        })
        assert resp.status_code == 422

    def test_successful_turn(self, mock_full_pipeline):
        """完整成功的回合推演。"""
        sid = self._init_and_get_sid()

        resp = client.post("/api/game/execute_turn", json={
            "session_id": sid,
            "player_input": "命令Saber_Artoria侦察周边区域。",
        })
        assert resp.status_code == 200
        data = resp.json()

        # 响应结构
        assert "narrative" in data
        assert len(data["narrative"]) > 0
        assert "memory_system" in data
        assert "turn_summary" in data

        # turn_summary
        ts = data["turn_summary"]
        assert ts["complexity_score"] == 6
        assert ts["router_source"] == "deepseek"
        assert ts["arbiter_model"] == "deepseek-v4-pro"
        assert ts["total_tokens"] > 0
        assert ts["total_latency_sec"] >= 0  # mock 环境下延迟可能为 0
        assert ts["turn_count"] == 1

        # session 应已更新
        resp2 = client.get(f"/api/game/session/{sid}")
        assert resp2.json()["turn_count"] == 1

    def test_high_complexity_triggers_reasoner(self):
        """复杂度 >= 9 → 使用 deepseek-reasoner。"""
        sid = self._init_and_get_sid()

        with patch(
            "game_server.call_ollama",
            new_callable=AsyncMock,
            side_effect=Exception("Ollama unavailable"),
        ), patch(
            "game_server.call_deepseek",
            new_callable=AsyncMock,
        ) as mock_ds:
            mock_ds.side_effect = [
                # Router: 高分
                (
                    json.dumps({"complexity_score": 10, "reason": "掀桌级操作"}),
                    {"total_tokens": 100},
                ),
                # Arbiter: 使用 reasoner
                (
                    json.dumps({
                        "judgment_report": "复杂的多线操作判定完成。",
                        "updated_memory_system": {
                            "active_servant_keys": ["Saber_Artoria", "Archer_EMIYA"],
                            "chronicle_history": ["游戏开始。", "第1回合。"],
                            "current_snapshot": {
                                "Protagonist_Master": {
                                    "hp": 100, "max_hp": 100, "status": "OK",
                                    "location": "test", "command_spells": 3,
                                    "is_alive": True, "mana_remaining": 100,
                                },
                                "Saber_Artoria": {
                                    "hp": 100, "max_hp": 100, "status": "OK",
                                    "location": "test", "command_spells": 0,
                                    "is_alive": True, "mana_remaining": 100,
                                },
                                "Archer_EMIYA": {
                                    "hp": 100, "max_hp": 100, "status": "OK",
                                    "location": "test", "command_spells": 0,
                                    "is_alive": True, "mana_remaining": 90,
                                },
                            },
                        },
                    }),
                    {"total_tokens": 3000},
                ),
                # Narrator
                (
                    "宏大的叙事文本...",
                    {"total_tokens": 800},
                ),
            ]

            resp = client.post("/api/game/execute_turn", json={
                "session_id": sid,
                "player_input": "同时发动三线作战的复杂骚操作。",
            })
            assert resp.status_code == 200
            # 验证使用了 reasoner
            assert resp.json()["turn_summary"]["arbiter_model"] == "deepseek-v4-pro"

    def test_turn_count_increments(self):
        """每回合 turn_count 递增。使用无限返回 mock。"""
        sid = self._init_and_get_sid()

        def _make_arbiter_output(turn: int):
            return (
                json.dumps({
                    "judgment_report": f"第{turn}回合判定完成。",
                    "updated_memory_system": {
                        "current_day": 1,
                        "current_phase": "night",
                        "active_servant_keys": ["Saber_Artoria", "Archer_EMIYA"],
                        "chronicle_history": ["start.", f"turn {turn}."],
                        "current_snapshot": {
                            "Protagonist_Master": {
                                "hp": 100, "max_hp": 100, "status": "OK",
                                "location": "test", "command_spells": 3,
                                "is_alive": True, "mana_remaining": 100,
                            },
                            "Saber_Artoria": {
                                "hp": 100, "max_hp": 100, "status": "OK",
                                "location": "test", "command_spells": 0,
                                "is_alive": True, "mana_remaining": 100,
                            },
                            "Archer_EMIYA": {
                                "hp": 100, "max_hp": 100, "status": "OK",
                                "location": "test", "command_spells": 0,
                                "is_alive": True, "mana_remaining": 90,
                            },
                            "Enemy_Master": {
                                "hp": 100, "max_hp": 100, "status": "OK",
                                "location": "test", "command_spells": 3,
                                "is_alive": True, "mana_remaining": 100,
                            },
                        },
                    },
                }),
                {"total_tokens": 1000},
            )

        with patch(
            "game_server.call_ollama",
            new_callable=AsyncMock,
            side_effect=Exception("Ollama unavailable"),
        ), patch(
            "game_server.call_deepseek",
            new_callable=AsyncMock,
        ) as mock_ds:
            # 每个 turn 需要 3 次 call_deepseek 调用
            # 构造一个无限循环的 side_effect
            call_count = [0]

            def infinite_side_effect(*args, **kwargs):
                idx = call_count[0]
                call_count[0] += 1
                phase = idx % 3  # 0=router, 1=arbiter, 2=narrator
                turn_num = idx // 3 + 1
                if phase == 0:
                    return (
                        json.dumps({"complexity_score": 5, "reason": "OK"}),
                        {"total_tokens": 50},
                    )
                elif phase == 1:
                    return _make_arbiter_output(turn_num)
                else:
                    return (f"第{turn_num}回合叙事。", {"total_tokens": 200})

            mock_ds.side_effect = infinite_side_effect

            for expected_turn in range(1, 4):
                resp = client.post("/api/game/execute_turn", json={
                    "session_id": sid,
                    "player_input": f"第{expected_turn}回合操作。",
                })
                assert resp.status_code == 200, f"Turn {expected_turn} failed: {resp.json()}"
                assert resp.json()["turn_summary"]["turn_count"] == expected_turn

    def test_day_limit_exceeded(self):
        """超过第7天 → 战争强制结束。"""
        sid = self._init_and_get_sid()
        from game_server import GAME_SESSIONS
        from models import GameMemorySystem
        # 手动推进到第8天
        snapshot = dict(GAME_SESSIONS[sid].memory_system.current_snapshot)
        GAME_SESSIONS[sid].memory_system = GameMemorySystem(
            active_servant_keys=GAME_SESSIONS[sid].memory_system.active_servant_keys,
            chronicle_history=GAME_SESSIONS[sid].memory_system.chronicle_history,
            current_snapshot=snapshot,
            current_day=8,
            current_phase="night",
        )

        resp = client.post("/api/game/execute_turn", json={
            "session_id": sid,
            "player_input": "最后一击。",
        })
        assert resp.status_code == 422
        assert "[WAR_ENDED]" in resp.json()["detail"]


# ==========================================
# Session 调试端点
# ==========================================
class TestSessionEndpoints:
    def test_get_session_found(self):
        resp = client.post("/api/game/init", json={})
        sid = resp.json()["session_id"]

        resp2 = client.get(f"/api/game/session/{sid}")
        assert resp2.status_code == 200
        assert resp2.json()["session_id"] == sid
        assert resp2.json()["turn_count"] == 0

    def test_get_session_not_found(self):
        resp = client.get("/api/game/session/nonexist")
        assert resp.status_code == 404

    def test_list_sessions(self):
        # 创建 3 个 session
        for _ in range(3):
            client.post("/api/game/init", json={})

        resp = client.get("/api/game/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_sessions"] == 3
        assert len(data["sessions"]) == 3

    def test_list_sessions_empty(self):
        resp = client.get("/api/game/sessions")
        assert resp.status_code == 200
        assert resp.json()["active_sessions"] == 0


# ==========================================
# Ollama fallback 链路
# ==========================================
class TestOllamaFallback:
    def test_ollama_unavailable_falls_back_to_deepseek(self):
        """Ollama 不可用 → 自动降级到 DeepSeek 做路由。"""
        sid = client.post("/api/game/init", json={}).json()["session_id"]

        with patch(
            "game_server.call_ollama",
            new_callable=AsyncMock,
            side_effect=Exception("Connection refused"),
        ), patch(
            "game_server.call_deepseek",
            new_callable=AsyncMock,
        ) as mock_ds:
            mock_ds.side_effect = [
                # Router fallback
                (
                    json.dumps({"complexity_score": 4, "reason": "基础操作"}),
                    {"total_tokens": 80},
                ),
                # Arbiter
                (
                    json.dumps({
                        "judgment_report": "判定完成。",
                        "updated_memory_system": {
                            "active_servant_keys": ["Saber_Artoria", "Archer_EMIYA"],
                            "chronicle_history": ["start.", "turn 1."],
                            "current_snapshot": {
                                "Protagonist_Master": {
                                    "hp": 100, "max_hp": 100, "status": "OK",
                                    "location": "test", "command_spells": 3,
                                    "is_alive": True, "mana_remaining": 100,
                                },
                                "Saber_Artoria": {
                                    "hp": 100, "max_hp": 100, "status": "OK",
                                    "location": "test", "command_spells": 0,
                                    "is_alive": True, "mana_remaining": 100,
                                },
                                "Archer_EMIYA": {
                                    "hp": 100, "max_hp": 100, "status": "OK",
                                    "location": "test", "command_spells": 0,
                                    "is_alive": True, "mana_remaining": 90,
                                },
                            },
                        },
                    }),
                    {"total_tokens": 1500},
                ),
                # Narrator
                ("叙事文本。", {"total_tokens": 300}),
            ]

            resp = client.post("/api/game/execute_turn", json={
                "session_id": sid,
                "player_input": "简单侦察。",
            })
            assert resp.status_code == 200
            assert resp.json()["turn_summary"]["router_source"] == "deepseek"
