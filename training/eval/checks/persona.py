# SPDX-License-Identifier: Apache-2.0
"""Narrator 人称 / 不出戏检查。

检测项:
  1. 第一人称叙述者   — 不得使用 “我认为” “我觉得” “我” 作为叙事者
  2. 第三人称指代玩家 — 不得用 “玩家” 等称呼替代 “你”
  3. 出戏元语言       — “作为AI” “根据GM报告” “以下是” 等 break character
  4. markdown 泄漏    — `**xxx**` `#` 等格式标记不该出现在游戏叙事中
  5. 英文混入         — 显著英文词比例（宝具英文名除外）

每项按阈值判定 passed / failed，阈值均为可配参数。
"""

import re
from typing import Any

from . import CheckResult

# ---------------------------------------------------------------------------
# 词表
# ---------------------------------------------------------------------------

_VIOLATION_FIRST_PERSON = re.compile(
    r"我认为|我觉得|我想|我感到|我明白|我注意到|我发现|我确定|我怀疑"
)

_VIOLATION_THIRD_PERSON = re.compile(
    r"(那个|这位|该)\s*玩家|玩家[你您他她]|玩家角色|玩家扮演|作为玩家"
)

_META_WORDS = [
    "作为AI", "作为一个AI", "根据GM报告", "以下是", "如下所示",
    "根据以上信息", "根据上述", "根据系统提示", "根据指令", "请让我",
    "我将叙述", "我将描述", "让我来叙述",
]

_MARKDOWN_PATTERNS = re.compile(
    r"\*\*|\#\s|__[^_]+__|\`[^`]+\`|\[\s*\]|\(\s*\)"
)

# 宝具英文名豁免（不在英文比例计算内）
_ENGLISH_EXEMPT = re.compile(
    r"Saber|Archer|Lancer|Rider|Caster|Assassin|Berserker|"
    r"Noble\s*Phantasm|Master|Servant|Command\s*Spell|"
    r"HP|MP|NP|Berserker"
)

# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def check_first_person(text: str, _ctx: dict[str, Any]) -> CheckResult:
    """叙事者不得使用第一人称。"""
    hits = _VIOLATION_FIRST_PERSON.findall(text)
    return CheckResult(
        name="persona_first_person",
        passed=len(hits) == 0,
        violations=hits[:20],
        metrics={"hit_count": len(hits)},
        detail=f"发现 {len(hits)} 处第一人称叙事：{hits[:5]}" if hits else "无违规",
    )


def check_third_person_pronoun(text: str, _ctx: dict[str, Any]) -> CheckResult:
    """不得用第三人称指代玩家。"""
    hits = _VIOLATION_THIRD_PERSON.findall(text)
    return CheckResult(
        name="persona_third_person",
        passed=len(hits) == 0,
        violations=hits[:20],
        metrics={"hit_count": len(hits)},
        detail=f"发现 {len(hits)} 处第三人称指代玩家" if hits else "无违规",
    )


def check_meta_language(text: str, _ctx: dict[str, Any]) -> CheckResult:
    """不得出现 break character 元语言。"""
    hits = [w for w in _META_WORDS if w in text]
    return CheckResult(
        name="persona_meta_language",
        passed=len(hits) == 0,
        violations=hits,
        metrics={"hit_count": len(hits)},
        detail=f"发现 {len(hits)} 处元语言泄出：{hits}" if hits else "无违规",
    )


def check_markdown_leak(text: str, _ctx: dict[str, Any]) -> CheckResult:
    """不得出现 markdown 格式标记。"""
    hits = _MARKDOWN_PATTERNS.findall(text)
    return CheckResult(
        name="persona_markdown_leak",
        passed=len(hits) == 0,
        violations=hits[:20],
        metrics={"hit_count": len(hits)},
        detail=f"发现 {len(hits)} 处 markdown 泄漏" if hits else "无违规",
    )


def check_english_intrusion(
    text: str, _ctx: dict[str, Any], threshold: float = 0.05
) -> CheckResult:
    """英文词占比过高检测（豁免英灵类名/专有名词）。"""
    cleaned = _ENGLISH_EXEMPT.sub("", text)
    # 简单词分割（中英文混合时有效的近似）
    words = cleaned.split()
    if not words:
        return CheckResult(
            name="persona_english_intrusion",
            passed=True,
            metrics={"ratio": 0.0},
            detail="无英文词",
        )
    english_words = [w for w in words if re.fullmatch(r"[a-zA-Z]+", w)]
    ratio = len(english_words) / len(words)
    return CheckResult(
        name="persona_english_intrusion",
        passed=ratio <= threshold,
        violations=english_words[:20] if ratio > threshold else [],
        metrics={"ratio": round(ratio, 4), "threshold": threshold},
        detail=f"英文比例 {ratio:.2%}（阈值 {threshold:.0%}）",
    )


# ---------------------------------------------------------------------------
# 汇总：所有 persona check
# ---------------------------------------------------------------------------

PERSONA_CHECKS = [
    check_first_person,
    check_third_person_pronoun,
    check_meta_language,
    check_markdown_leak,
    check_english_intrusion,
]


def run_all_persona(text: str, ctx: dict[str, Any] | None = None) -> list[CheckResult]:
    ctx = ctx or {}
    return [fn(text, ctx) for fn in PERSONA_CHECKS]
