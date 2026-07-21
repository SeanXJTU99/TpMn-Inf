#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""评估主入口 —— 一键跑全量 checks + 可选 pairwise/rubric judge + 出 report。

用法:
  # 仅程序化硬校验（零 API 调用）
  python3 training/eval/runners/run_eval.py --input results.jsonl --tag sft_v1

  # 含 pairwise judge（需要 GEMINI_API_KEY / DEEPSEEK_API_KEY）
  python3 training/eval/runners/run_eval.py --input results.jsonl --tag sft_v1 --judge pairwise

  # 全量
  python3 training/eval/runners/run_eval.py --input results.jsonl --tag sft_v1 \\
      --judge pairwise --judge rubric --human-votes human.csv
"""

import argparse
import json
from pathlib import Path
from typing import Any

from training.eval.checks.persona import run_all_persona
from training.eval.checks.schema import run_all_schema
from training.eval.checks.leak import run_all_leak
from training.eval.checks.slop import run_all_slop
from training.eval.report import EvalReport, to_scorecard, add_to_history


def load_outputs(path: str) -> list[dict[str, Any]]:
    """加载 replay 产出的 JSONL。"""
    lines = [
        l for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip()
    ]
    return [json.loads(l) for l in lines]


def extract_texts(sessions: list[dict]) -> list[str]:
    """从 replay 输出提取所有轮次的 model_output。"""
    texts = []
    for sess in sessions:
        for turn in sess.get("turns", []):
            out = turn.get("model_output", "").strip()
            if out:
                texts.append(out)
    return texts


def run_programmatic_checks(
    texts: list[str], ctx: dict[str, Any]
) -> dict[str, Any]:
    """运行全部程序化硬校验，返回汇总字典。"""
    all_persona = []
    all_schema = []
    all_leak = []
    all_slop = []
    for t in texts:
        all_persona.extend(run_all_persona(t, ctx))
        all_schema.extend(run_all_schema(t, ctx))
        all_leak.extend(run_all_leak(t, ctx))
    # slop 拿全部 texts 做跨样本分析
    all_slop.extend(run_all_slop("", {**ctx, "samples": texts}))

    def _summarize(results, key) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in results:
            name = getattr(r, key)
            if name:
                counts[name] = counts.get(name, 0) + 1
        return counts

    return {
        "persona": {
            "total": len(all_persona),
            "failed": sum(1 for r in all_persona if not r.passed),
            "by_check": {r.name: r.passed for r in all_persona},
        },
        "schema": {
            "total": len(all_schema),
            "failed": sum(1 for r in all_schema if not r.passed),
            "by_check": {r.name: r.passed for r in all_schema},
        },
        "leak": {
            "total": len(all_leak),
            "failed": sum(1 for r in all_leak if not r.passed),
            "by_check": {r.name: r.passed for r in all_leak},
        },
        "slop": {
            "total": len(all_slop),
            "failed": sum(1 for r in all_slop if not r.passed),
            "by_check": {r.name: r.passed for r in all_slop},
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="replay 输出的 JSONL")
    ap.add_argument("--tag", required=True, help="e.g. sft_v1 or dpo_r2")
    ap.add_argument("--judge", action="append", default=[], choices=["pairwise", "rubric"])
    ap.add_argument("--human-votes", help="盲测 CSV 路径")
    ap.add_argument("--output-dir", default="training/eval/reports")
    args = ap.parse_args()

    sessions = load_outputs(args.input)
    texts = extract_texts(sessions)
    ctx: dict[str, Any] = {}
    if sessions:
        first = sessions[0]
        ctx["servant_db"] = first.get("servant_db", {})
        ctx["revealed_true_names"] = set(first.get("revealed_true_names", []))
        ctx["revealed_np_names"] = set(first.get("revealed_np_names", []))

    print(f"[硬校验] {len(texts)} 段文本 ...")
    prog_results = run_programmatic_checks(texts, ctx)

    # Gate 判定
    report = EvalReport(tag=args.tag)

    # persona gates
    persona = prog_results["persona"]
    report.gate_persona_pronoun = persona["by_check"].get("persona_first_person", True)

    # schema gates
    schema = prog_results["schema"]
    report.gate_arbiter_json_pass = schema["by_check"].get("schema_json_parse", True)

    # leak gates
    leak = prog_results["leak"]
    report.gate_info_leak = leak["by_check"].get("leak_true_name", True)

    report.persona_violations = {
        k: 1 for k, v in persona["by_check"].items() if not v
    }
    report.schema_violations = {
        k: 1 for k, v in schema["by_check"].items() if not v
    }
    report.leak_violations = {
        k: 1 for k, v in leak["by_check"].items() if not v
    }

    # Pairwise judge（需要 API key 时先占位）
    if "pairwise" in args.judge:
        print("[pairwise judge] 需要 API key，跳过（请使用 --judge pairwise 并设置 env）")
        # TODO: 接入 judge_fn
    if "rubric" in args.judge:
        print("[rubric judge] 跳过（同上）")

    # 人工盲测统计
    if args.human_votes:
        import csv
        with open(args.human_votes, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        wins = sum(1 for r in rows if r.get("choice") == "a")
        ties = sum(1 for r in rows if r.get("choice") == "tie")
        total = len(rows)
        report.pairwise_win_rate = wins / total if total > 0 else 0.0
        report.pairwise_ci_low = max(0, report.pairwise_win_rate - 0.1)
        report.pairwise_ci_high = min(1, report.pairwise_win_rate + 0.1)
        print(f"[human] {wins}W / {ties}T / {total-wins-ties}L ({total} pairs)")

    report.calc_composite()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"scorecard_{args.tag}.md"

    scorecard = to_scorecard(report, path)
    add_to_history(report, out_dir / "history.csv")
    print(f"\n{scorecard}")
    print(f"\n报告已存: {path}")
    print(f"历史追加: {out_dir / 'history.csv'}")


if __name__ == "__main__":
    main()
