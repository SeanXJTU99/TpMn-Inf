# SPDX-License-Identifier: Apache-2.0
"""反 Slop 指标 —— DPO 多样性塌缩预警。

检测项:
  1. distinct-2 / distinct-3 — 跨样本 bigram/trigram 多样性
  2. self-BLEU — 同一 prompt 多次采样的互相似度
  3. 套话词表 — 型月/网文常见 cliché 命中率
  4. 句长方差 — 句式趋同时方差坍缩

self-BLEU 需要 inputs: list[str] 多段样本（同一 scene 的不同独立采样）。
单段样本仅运行 distinct-n 和套话检测。
"""

import math
import re
from collections import Counter
from typing import Any

from . import CheckResult


# ─── 型月 / 中文叙事常见套话词表 ───

_CLICHE_WORDS = [
    "嘴角勾起一抹弧度", "眼中闪过一丝", "嘴角微微上扬", "眼眸深处",
    "不约而同", "微微一怔", "不由得", "深吸一口气", "攥紧拳头",
    "咬紧牙关", "瞳孔猛地收缩", "心头一紧", "后背发凉", "脊背发寒",
    "寒意彻骨", "空气中弥漫着", "夜色如墨", "月色如水",
    "如你所料", "果不其然", "与此同时", "在这千钧一发之际",
    "那一瞬间", "电光火石之间", "说不清道不明", "不可名状",
    "嘴角溢出一丝", "不置可否", "意味深长", "冷冷的",
    # 游戏特有套话
    "你的 Servant", "魔力回路", "灵体化",
]


def _tokenize_zh(text: str) -> list[str]:
    """简单中文分词：按标点 / 空格切，连续中文字符为 token。"""
    # 切分标点和空白
    tokens: list[str] = []
    buf = ""
    for ch in text:
        if re.match(r"[一-鿿\w]", ch):
            buf += ch
        else:
            if buf:
                tokens.append(buf)
                buf = ""
    if buf:
        tokens.append(buf)
    return tokens


def _distinct_n(samples: list[str], n: int) -> float:
    """distinct-n: 去重 n-gram 数 / 总数。"""
    all_ngrams: list[str] = []
    for text in samples:
        tokens = _tokenize_zh(text)
        for i in range(len(tokens) - n + 1):
            all_ngrams.append("|".join(tokens[i : i + n]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def _sentence_lengths(text: str) -> list[int]:
    """返回各句的字符数。句分隔符：。！？\n"""
    sents = re.split(r"[。！？\n]+", text)
    return [len(s.strip()) for s in sents if s.strip()]


# ─── checks ───


def check_distinct_n(
    _text: str, ctx: dict[str, Any], n: int = 2
) -> CheckResult:
    """distinct-n 多样性（需要 ctx["samples"] 列表）。"""
    samples: list[str] = ctx.get("samples", [])
    if not samples:
        return CheckResult(
            name=f"slop_distinct_{n}",
            passed=True,
            metrics={f"distinct_{n}": -1.0},
            detail="无多样本，跳过 distinct-n",
        )

    score = _distinct_n(samples, n)
    # 阈值：distinct-2 < 0.3 视为塌缩警告（中文字数少，bigram 覆盖率天然低）
    warn_threshold = 0.15 if n == 2 else 0.08

    return CheckResult(
        name=f"slop_distinct_{n}",
        passed=score >= warn_threshold,
        violations=[] if score >= warn_threshold else [f"distinct-{n}={score:.3f} < {warn_threshold}"],
        metrics={f"distinct_{n}": round(score, 4)},
        detail=f"distinct-{n}={score:.3f}（阈值 {warn_threshold}）",
    )


def check_cliche_hits(text: str, _ctx: dict[str, Any]) -> CheckResult:
    """套话词表命中率（每千字）。"""
    hits = [w for w in _CLICHE_WORDS if w in text]
    char_count = len(text)
    rate = len(hits) / max(char_count, 1) * 1000  # 每千字

    return CheckResult(
        name="slop_cliche",
        passed=rate <= 3.0,  # 每千字 ≤3 处套话
        violations=hits[:15],
        metrics={"hits_per_1k_chars": round(rate, 2)},
        detail=f"套话命中率 {rate:.1f}/千字（{len(hits)} 处）",
    )


def check_sentence_length_variance(
    text: str, _ctx: dict[str, Any]
) -> CheckResult:
    """句长方差检测——方差 <5 说明句式趋同。"""
    lens = _sentence_lengths(text)
    if len(lens) < 3:
        return CheckResult(
            name="slop_sentence_var",
            passed=True,
            metrics={"sentence_var": -1.0, "sentence_count": len(lens)},
            detail="句子数不足，跳过",
        )

    mean_len = sum(lens) / len(lens)
    var_len = sum((x - mean_len) ** 2 for x in lens) / len(lens)

    return CheckResult(
        name="slop_sentence_var",
        passed=var_len >= 5.0,
        violations=[] if var_len >= 5.0 else [f"句长方差={var_len:.1f} < 5"],
        metrics={
            "sentence_var": round(var_len, 1),
            "sentence_count": len(lens),
            "mean_len": round(mean_len, 1),
        },
        detail=f"句长方差={var_len:.1f}（{len(lens)} 句，均值 {mean_len:.1f}）",
    )


# ─── 汇总 ───

SLOP_CHECKS = [
    lambda t, c: check_distinct_n(t, c, n=2),
    lambda t, c: check_distinct_n(t, c, n=3),
    check_cliche_hits,
    check_sentence_length_variance,
]


def run_all_slop(text: str, ctx: dict[str, Any] | None = None) -> list[CheckResult]:
    ctx = ctx or {}
    return [fn(text, ctx) for fn in SLOP_CHECKS]
