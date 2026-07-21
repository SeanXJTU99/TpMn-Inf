#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""评估报告生成 —— scorecard.md + history.csv。

汇总 gate 通过状态、composite score、各轴指标。
"""

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class EvalReport:
    """一次完整评估 run 的汇总。"""
    tag: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Gate 状态
    gate_arbiter_json_pass: bool = True
    gate_rules_correct: bool = True
    gate_info_leak: bool = True
    gate_resurrection: bool = True
    gate_persona_pronoun: bool = True
    gate_slop_degrade: bool = True

    # 分项指标
    pairwise_win_rate: float = 0.0
    pairwise_ci_low: float = 0.0
    pairwise_ci_high: float = 0.0
    consistency_rate: float = 0.0       # 1 - 归一化矛盾率
    instruction_follow_rate: float = 0.0  # I-Set + A-Set 合规率
    needle_recall_16k: float = 0.0        # 探针召回 @16k

    # 硬校验汇总
    persona_violations: dict[str, int] = field(default_factory=dict)
    leak_violations: dict[str, int] = field(default_factory=dict)
    schema_violations: dict[str, int] = field(default_factory=dict)
    slop_metrics: dict[str, float] = field(default_factory=dict)

    # Composite（eval_plan.md 权重）
    composite: float = 0.0

    def calc_composite(self):
        self.composite = round(
            0.35 * self.pairwise_win_rate
            + 0.25 * self.consistency_rate
            + 0.25 * self.instruction_follow_rate
            + 0.15 * self.needle_recall_16k,
            4,
        )

    @property
    def all_gates_passed(self) -> bool:
        return all([
            self.gate_arbiter_json_pass,
            self.gate_rules_correct,
            self.gate_info_leak,
            self.gate_resurrection,
            self.gate_persona_pronoun,
            self.gate_slop_degrade,
        ])

    @property
    def gate_summary(self) -> str:
        gates = {
            "JSON≥99%": self.gate_arbiter_json_pass,
            "规则≥95%": self.gate_rules_correct,
            "信息泄漏=0": self.gate_info_leak,
            "复活=0": self.gate_resurrection,
            "人称≤1%": self.gate_persona_pronoun,
            "slop≤15%恶化": self.gate_slop_degrade,
        }
        return " | ".join(f"{'✅' if v else '❌'}{k}" for k, v in gates.items())


def to_scorecard(report: EvalReport, path: Path | None = None) -> str:
    """生成 scorecard.md 内容。"""
    lines = [
        f"# Eval Scorecard — {report.tag}",
        f"Timestamp: {report.timestamp}",
        "",
        "## Gates",
        f"{report.gate_summary}",
        f"**{'✅ 全部通过' if report.all_gates_passed else '❌ 未通过 — 不晋升'}**",
        "",
        "## Composite Score",
        f"| 指标 | 值 |",
        f"|---|---|",
        f"| composite | {report.composite:.4f} |",
        f"| pairwise win-rate | {report.pairwise_win_rate:.3f} (95% CI [{report.pairwise_ci_low:.3f}, {report.pairwise_ci_high:.3f}]) |",
        f"| consistency | {report.consistency_rate:.3f} |",
        f"| instruction_follow | {report.instruction_follow_rate:.3f} |",
        f"| needle_recall@16k | {report.needle_recall_16k:.3f} |",
        "",
        "## Checks Detail",
        f"| Persona | Leak | Schema | Slop |",
        f"|---|---|---|---|",
        f"| {_fmt_dict(report.persona_violations)} | {_fmt_dict(report.leak_violations)} | {_fmt_dict(report.schema_violations)} | {_fmt_dict(report.slop_metrics)} |",
    ]
    content = "\n".join(lines)
    if path:
        path.write_text(content, encoding="utf-8")
    return content


def _fmt_dict(d: dict) -> str:
    if not d:
        return "—"
    return ", ".join(f"{k}={v}" for k, v in list(d.items())[:5])


def add_to_history(report: EvalReport, path: Path | None = None):
    """追加一行到 history.csv。"""
    if path is None:
        path = Path(__file__).parent / "history.csv"
    row = {
        "tag": report.tag,
        "timestamp": report.timestamp,
        "composite": report.composite,
        "pairwise_win_rate": report.pairwise_win_rate,
        "gates_passed": report.all_gates_passed,
        "consistency": report.consistency_rate,
        "instruction_follow": report.instruction_follow_rate,
        "needle_16k": report.needle_recall_16k,
    }
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
