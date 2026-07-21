# SPDX-License-Identifier: Apache-2.0
"""信息泄漏检测 — Narrator 输出中是否泄露了尚未揭示的信息。

核心：对照 reveal_state（游戏运行时已揭示信息）与 servant_db（全部秘密），
检测叙事文本里是否出现了不应出现的真名 / 宝具名。

检测项:
  1. 真名泄漏 — 未揭示英灵的真名出现在叙事中
  2. 宝具名泄漏 — 未揭示宝具名出现在叙事中（含英文名）
  3. 令咒余量泄漏 — 玩家未主动查看时不应暴露精确数字
"""

from typing import Any

from . import CheckResult


def _build_secret_registry(ctx: dict[str, Any]) -> dict[str, set[str]]:
    """从 context 构建秘密登记表。

    ctx 期望字段:
      servant_db: dict  — 原始 servant_db.json
      revealed_true_names: set[str]  — 已揭示真名的 servant_key 集合
      revealed_np_names: set[str]    — 已揭示宝具名的 servant_key 集合
    """
    db = ctx.get("servant_db", {})
    revealed_names = ctx.get("revealed_true_names", set())
    revealed_nps = ctx.get("revealed_np_names", set())

    secrets: dict[str, set[str]] = {"true_names": set(), "np_names": set()}

    for key, card in db.items():
        if key not in revealed_names:
            tn = card.get("true_name", "")
            if tn:
                # 分段拆词，取关键片段（中文名）
                secrets["true_names"].add(tn)
                # 也收录不带括号的版本
                clean = tn.split("(")[0].strip()
                if clean and clean != tn:
                    secrets["true_names"].add(clean)

        if key not in revealed_nps:
            np_info = card.get("noble_phantasm", {})
            np_name = np_info.get("name", "")
            if np_name:
                # 《XXX》 内的中文名 + 英文名
                secrets["np_names"].add(np_name)
                # 提取英文名（括号内）
                import re
                eng = re.findall(r"[（(]([^）)]+)[）)]", np_name)
                for e in eng:
                    secrets["np_names"].add(e.strip())

    return secrets


def check_true_name_leak(text: str, ctx: dict[str, Any]) -> CheckResult:
    """检测未揭示真名是否出现在叙事中。"""
    secrets = _build_secret_registry(ctx)
    hits = [name for name in secrets["true_names"] if name and name in text]

    return CheckResult(
        name="leak_true_name",
        passed=len(hits) == 0,
        violations=hits,
        metrics={"leak_count": len(hits)},
        detail=f"泄漏 {len(hits)} 个未揭示真名: {hits[:5]}" if hits else "无真名泄漏",
    )


def check_np_name_leak(text: str, ctx: dict[str, Any]) -> CheckResult:
    """检测未揭示宝具名是否泄漏。"""
    secrets = _build_secret_registry(ctx)
    hits = [name for name in secrets["np_names"] if name and len(name) >= 4 and name in text]

    return CheckResult(
        name="leak_noble_phantasm",
        passed=len(hits) == 0,
        violations=hits,
        metrics={"leak_count": len(hits)},
        detail=f"泄漏 {len(hits)} 个未揭示宝具名" if hits else "无宝具泄漏",
    )


def check_command_spell_leak(text: str, ctx: dict[str, Any]) -> CheckResult:
    """令咒余量精确数值不应在叙事中被动暴露（仅玩家主动查看时允许）。"""
    cs_pattern = r"令咒[^\d]*(\d)\s*[划枚次]"
    import re
    matches = re.findall(cs_pattern, text)

    # 如果 context 表示玩家主动查看了，豁免
    if ctx.get("player_checked_status", False):
        return CheckResult(
            name="leak_command_spells",
            passed=True,
            metrics={"leak_count": 0},
            detail="玩家主动查看，豁免",
        )

    return CheckResult(
        name="leak_command_spells",
        passed=len(matches) == 0,
        violations=[f"令咒余量 {m}" for m in matches],
        metrics={"leak_count": len(matches)},
        detail=f"被动暴露令咒余量 {len(matches)} 次" if matches else "无被动暴露",
    )


# ─── 汇总 ───

LEAK_CHECKS = [
    check_true_name_leak,
    check_np_name_leak,
    check_command_spell_leak,
]


def run_all_leak(text: str, ctx: dict[str, Any] | None = None) -> list[CheckResult]:
    ctx = ctx or {}
    ctx.setdefault("servant_db", {})
    return [fn(text, ctx) for fn in LEAK_CHECKS]
