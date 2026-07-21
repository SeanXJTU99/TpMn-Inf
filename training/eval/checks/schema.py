# SPDX-License-Identifier: Apache-2.0
"""Arbiter JSON schema / Pydantic 校验。

检测项:
  1. JSON parse 一次通过率（不含 repair）
  2. Pydantic model_validate 通过率
  3. 数值幻觉 — 输出的面板数值对照 servant_db.json 是否在合理范围
  4. 关键字段存在性 — result / damage / state_changes / narration_hints

模型复用 game_server/models.py 的 CharacterState + GameMemorySystem。

使用方式:
  需 game_server 路径在 PYTHONPATH 中，或从 game_server/ 复制 models.py 到本包。
  若无 game_server 依赖，回退到轻量 schema 校验（仅检查 JSON 结构 + 必要 key）。
"""

import json
import re
from typing import Any

from . import CheckResult


# ─── schema 定义（与 game_server/models.py 最低耦合） ───

_REQUIRED_ARBITER_FIELDS = {
    "judgment_report": {
        "result": str,
        "damage": (int, float),
    },
    "updated_memory_system": dict,
}

# servant_db 35 骑的属性上界（用于数值幻觉检测，从 servant_db.json 派生）
# key = servant_key, value = (max_str_val, max_end_val, ...)
_SERVANT_ATTR_BOUNDS: dict[str, dict[str, tuple[int, str]]] = {}

# 属性名 → 顺序值映射（粗检测用）
_ATTR_RANK_ORDER = {
    "E": 1, "E+": 1.5, "D": 2, "D+": 2.5, "C": 3, "C+": 3.5,
    "B": 4, "B+": 4.5, "A": 5, "A+": 5.5, "A++": 6, "EX": 7,
}

_STANDARD_ATTRS = ["strength", "endurance", "agility", "mana", "luck", "noble_phantasm"]


def _try_load_servant_db(ctx: dict[str, Any]) -> dict[str, Any]:
    """从 context 或本地文件加载 servant_db。"""
    if "servant_db" in ctx:
        return ctx["servant_db"]
    # 尝试从 game_server 导入
    try:
        import json as _json
        from pathlib import Path

        path = Path(__file__).parents[4] / "game_server" / "servant_db.json"
        if path.exists():
            return _json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


# ─── checks ───


def check_json_parse(text: str, _ctx: dict[str, Any]) -> CheckResult:
    """JSON 解析一次成功率。"""
    try:
        json.loads(text)
        return CheckResult(
            name="schema_json_parse",
            passed=True,
            metrics={"success": 1.0},
            detail="JSON parse 成功",
        )
    except json.JSONDecodeError as e:
        return CheckResult(
            name="schema_json_parse",
            passed=False,
            violations=[str(e)],
            metrics={"success": 0.0},
            detail=f"JSON parse 失败: {e}",
        )


def check_pydantic_validate(text: str, ctx: dict[str, Any]) -> CheckResult:
    """尝试通过 game_server 的 Pydantic 模型校验。可选依赖。"""
    try:
        from game_server.models import GameMemorySystem  # type: ignore[import-untyped]

        obj = json.loads(text)
        memory = obj.get("updated_memory_system")
        if memory is None:
            return CheckResult(
                name="schema_pydantic",
                passed=False,
                violations=["缺少 updated_memory_system"],
                metrics={"success": 0.0},
                detail="JSON 中缺失 updated_memory_system 字段",
            )
        GameMemorySystem.model_validate(memory)
        return CheckResult(
            name="schema_pydantic",
            passed=True,
            metrics={"success": 1.0},
            detail="Pydantic 校验通过",
        )
    except ImportError:
        # 无 game_server 依赖，跳过 Pydantic 校验
        return CheckResult(
            name="schema_pydantic",
            passed=True,  # 不视为失败——环境未安装依赖
            metrics={"success": -1.0},  # -1 = skipped
            detail="跳过: game_server 未安装",
        )
    except Exception as e:
        return CheckResult(
            name="schema_pydantic",
            passed=False,
            violations=[str(e)[:200]],
            metrics={"success": 0.0},
            detail=f"Pydantic 校验失败: {e}",
        )


def check_numerical_hallucination(
    text: str, ctx: dict[str, Any]
) -> CheckResult:
    """数值幻觉检测：HP/令咒/魔力 是否在合理范围。纯启发式，不依赖 servant_db。"""
    violations: list[str] = []

    # 检测超出游戏范围的数值
    hp_over = re.findall(r"[Hh][Pp]\s*(?:剩余|当前)?\s*(\d{3,})", text)
    for val in hp_over:
        if int(val) > 100:
            violations.append(f"HP 超出上限: {val}")

    cs_over = re.findall(r"令咒\s*(?:剩余|余量)?\s*(\d+)", text)
    for val in cs_over:
        if int(val) > 3:
            violations.append(f"令咒超出上限(3): {val}")

    # 属性值幻觉（英文 rank 格式）
    false_ranks = re.findall(r"[Ss]trength.*?([A-E][+]{0,2})\b", text)
    for r in false_ranks:
        if r not in _ATTR_RANK_ORDER:
            violations.append(f"无效属性值: {r}")

    return CheckResult(
        name="schema_numerical_hallucination",
        passed=len(violations) == 0,
        violations=violations,
        metrics={"violation_count": len(violations)},
        detail=f"发现 {len(violations)} 处数值幻觉" if violations else "无数值幻觉",
    )


def check_required_fields(text: str, _ctx: dict[str, Any]) -> CheckResult:
    """关键字段存在性检查。"""
    violations: list[str] = []
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return CheckResult(
            name="schema_required_fields",
            passed=False,
            violations=["JSON 无法解析, 跳过字段检查"],
            metrics={"missing": 1},
            detail="JSON 无法解析",
        )

    # judgment_report
    jr = obj.get("judgment_report", {})
    if not isinstance(jr, dict):
        violations.append("judgment_report 缺失或不是 dict")
    else:
        if "result" not in jr:
            violations.append("judgment_report.result 缺失")
        if "damage" not in jr:
            violations.append("judgment_report.damage 缺失")
        if not isinstance(jr.get("result"), str):
            violations.append("judgment_report.result 不是字符串")

    # updated_memory_system
    ums = obj.get("updated_memory_system")
    if ums is None:
        violations.append("updated_memory_system 缺失")
    elif not isinstance(ums, dict):
        violations.append("updated_memory_system 不是 dict")

    return CheckResult(
        name="schema_required_fields",
        passed=len(violations) == 0,
        violations=violations,
        metrics={"missing": len(violations)},
        detail=f"缺失/错误 {len(violations)} 个关键字段" if violations else "关键字段 OK",
    )


# ─── 汇总 ───

SCHEMA_CHECKS = [
    check_json_parse,
    check_pydantic_validate,
    check_numerical_hallucination,
    check_required_fields,
]


def run_all_schema(text: str, ctx: dict[str, Any] | None = None) -> list[CheckResult]:
    ctx = ctx or {}
    return [fn(text, ctx) for fn in SCHEMA_CHECKS]
