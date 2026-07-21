# SPDX-License-Identifier: Apache-2.0
"""Rubric judge — 四维锚定评分（诊断用，不与 pairwise 冲突）。

prompt 措辞与 RLAIF reward prompt 完全不同（eval_plan.md §5 防 Goodhart）。

用法:
  scores = asyncio.run(run_rubric(samples, judge_fn))
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_PROMPT_TEMPLATE = (
    Path(__file__).parent / "prompts" / "rubric_v1.md"
).read_text(encoding="utf-8")


@dataclass
class RubricScores:
    style: float
    consistency: float
    hint: float
    you: float
    comment: str = ""

    @property
    def weighted(self) -> float:
        """eval_plan.md 权重: 0.40×style + 0.30×consistency + 0.20×hint + 0.10×you"""
        return 0.40 * self.style + 0.30 * self.consistency + 0.20 * self.hint + 0.10 * self.you


@dataclass
class RubricResult:
    gm_report: str
    model_scores: RubricScore
    baseline_scores: RubricScore  # DeepSeek reference
    delta_weighted: float  # model - baseline


def _parse_rubric_response(text: str) -> RubricScores:
    try:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return RubricScores(0, 0, 0, 0, "parse error")
        obj = json.loads(m.group(0))
        return RubricScores(
            style=float(obj.get("style", 0)),
            consistency=float(obj.get("consistency", 0)),
            hint=float(obj.get("hint", 0)),
            you=float(obj.get("you", 0)),
            comment=obj.get("comment", ""),
        )
    except Exception:
        return RubricScores(0, 0, 0, 0, "parse error")


def build_rubric_prompt(gm_report: str, output: str) -> str:
    return _PROMPT_TEMPLATE.format(gm_report=gm_report, output=output)


async def run_rubric(
    samples: list[dict[str, str]],
    judge_fn: Callable[..., Any],
) -> list[RubricResult]:
    results: list[RubricResult] = []

    for s in samples:
        gm = s["gm_report"]
        model_out = s["model_out"]
        baseline_out = s.get("deepseek_out", "")

        tasks = [
            judge_fn(build_rubric_prompt(gm, model_out)),
            judge_fn(build_rubric_prompt(gm, baseline_out)),
        ]
        r_model, r_baseline = await asyncio.gather(*tasks)

        model_sc = _parse_rubric_response(r_model)
        baseline_sc = _parse_rubric_response(r_baseline)

        results.append(RubricResult(
            gm_report=gm[:200],
            model_scores=model_sc,
            baseline_scores=baseline_sc,
            delta_weighted=model_sc.weighted - baseline_sc.weighted,
        ))

    return results
