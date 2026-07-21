# SPDX-License-Identifier: Apache-2.0
"""成对比较 judge — 带 position swap 的 win/tie/loss 判定。

用法:
  import asyncio
  results = asyncio.run(run_pairwise(samples, judge_fn))
"""

import asyncio
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_PROMPT_TEMPLATE = (
    Path(__file__).parent / "prompts" / "pairwise_v1.md"
).read_text(encoding="utf-8")


@dataclass
class PairResult:
    gm_report: str
    winner: str  # "A" | "B" | "tie"
    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    reason: str = ""
    swap_winner: str = ""  # 位置互换后的赢家
    swap_reason: str = ""

    @property
    def resolved_winner(self) -> str:
        """position swap 后解析: A/B/tie。两次不一致 → tie。"""
        if self.winner == self.swap_winner:
            if self.winner == "A":
                # 两边都选 A（第一个位置），即都选同一个模型 → tie
                return "tie"
            return self.winner
        if self.winner == "A" and self.swap_winner == "B":
            # swap 后赢家互换 → 说明该模型在两轮都胜了 → 解码真实胜者
            # 第一轮 A=model1, B=model2, 选 A=model1
            # 第二轮 A=model2, B=model1, 选 B=model2 → swap_winner=B=model1
            return "swap_conflict"  # 不应出现，进人工复核
        return "tie"


@dataclass
class PairwiseStats:
    win_rate: float
    loss_rate: float
    tie_rate: float
    ci_lower: float  # 95% bootstrap CI 下界
    ci_upper: float
    total_pairs: int
    details: list[PairResult] = field(default_factory=list)


def _parse_pair_response(text: str) -> tuple[str, dict, str]:
    """解析 judge 返回的 JSON → (winner, scores_dict, reason)。"""
    try:
        # 提取 JSON（可能被 markdown 包裹）
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return "tie", {}, "JSON not found"
        obj = json.loads(m.group(0))
        return (
            obj.get("winner", "tie"),
            obj.get("scores", {}),
            obj.get("reason", ""),
        )
    except Exception:
        return "tie", {}, "parse error"


def build_pairwise_prompt(gm_report: str, out_a: str, out_b: str) -> str:
    return _PROMPT_TEMPLATE.format(
        gm_report=gm_report, output_a=out_a, output_b=out_b
    )


async def run_pairwise(
    samples: list[dict[str, str]],
    judge_fn: Callable[..., Any],
    swap: bool = True,
) -> PairwiseStats:
    """运行成对比较。

    samples: list of {"gm_report": ..., "model_out": ..., "deepseek_out": ...}
    judge_fn: async (prompt: str) -> str
    """
    results: list[PairResult] = []

    for s in samples:
        gm = s["gm_report"]
        a_out = s["model_out"]
        b_out = s["deepseek_out"]

        # 第一轮: model=A, deepseek=B
        p1 = build_pairwise_prompt(gm, a_out, b_out)
        r1 = await judge_fn(p1)
        w1, sc1, reason1 = _parse_pair_response(r1)

        swap_w = ""
        swap_reason = ""
        if swap:
            # 第二轮：位置互换
            p2 = build_pairwise_prompt(gm, b_out, a_out)
            r2 = await judge_fn(p2)
            w2, _, reason2 = _parse_pair_response(r2)
            # w2 里的 A 对应第二轮 output_a=b_out=deepseek, B 对应 output_b=a_out=model
            # 我们要映射回 model 视角
            if w2 == "A":
                swap_w = "B"  # deepseek 赢 = model 视角的 B
            elif w2 == "B":
                swap_w = "A"  # model 赢
            else:
                swap_w = w2  # tie
            swap_reason = reason2

        results.append(PairResult(
            gm_report=gm[:200],
            winner=w1,
            scores=sc1,
            reason=reason1,
            swap_winner=swap_w,
            swap_reason=swap_reason,
        ))

    # 汇总
    resolved = [r.resolved_winner for r in results]
    n = len(resolved)
    wins = resolved.count("A")
    losses = resolved.count("B")
    ties = resolved.count("tie")
    conflicts = resolved.count("swap_conflict")

    # 将 conflicts 归入 tie
    ties += conflicts
    effective = n

    wr = wins / effective if effective > 0 else 0.0
    lr = losses / effective if effective > 0 else 0.0
    tr = ties / effective if effective > 0 else 0.0

    # Bootstrap 95% CI（简化：正态近似）
    se = math.sqrt(wr * (1 - wr) / effective) if effective > 0 else 1.0
    ci_low = max(0.0, wr - 1.96 * se)
    ci_high = min(1.0, wr + 1.96 * se)

    return PairwiseStats(
        win_rate=wr,
        loss_rate=lr,
        tie_rate=tr,
        ci_lower=ci_low,
        ci_upper=ci_high,
        total_pairs=effective,
        details=results,
    )
