#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""盲测工具 —— 终端 A/B 投票，落 CSV（人工 ground truth）。

每次展示两段叙事（随机打乱来源），用户盲投 1/2/=。
结果落在 training/eval/human/results/<tag>_<timestamp>.csv。

用法:
  python3 training/eval/human/blind_ab.py --tag r1_sft_v1 --prompts input.jsonl
  python3 training/eval/human/blind_ab.py --tag r1_sft_v1 --interleaved story_a.txt story_b.txt
"""

import argparse
import csv
import random
import sys
from datetime import datetime
from pathlib import Path

OUT_DIR = Path(__file__).parent / "results"


def _present_pair(idx: int, total: int, a: str, b: str, gm: str) -> str:
    """在终端展示 A/B 对，返回用户选择。"""
    print(f"\n{'='*60}")
    print(f"  第 {idx+1}/{total} 对")
    print(f"{'='*60}")
    print(f"\n【GM 报告】{gm[:300]}")
    print(f"\n--- 叙事 A ---")
    print(a[:1500])
    print(f"\n--- 叙事 B ---")
    print(b[:1500])
    print(f"\n你的判断: [1] A 更好  [2] B 更好  [=] 差不多  [s] 跳过  [q] 退出并保存")
    while True:
        choice = input("> ").strip().lower()
        if choice in ("1", "2", "=", "s", "q"):
            return choice
        print("请输入 1 / 2 / = / s / q")


def run_interleaved(tag: str, file_a: str, file_b: str):
    """两段独立文本交替展示（每条 text 长度相等时适用）。"""
    lines_a = Path(file_a).read_text(encoding="utf-8").strip().split("\n")
    lines_b = Path(file_b).read_text(encoding="utf-8").strip().split("\n")
    max_len = max(len(lines_a), len(lines_b))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"{tag}_{stamp}.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "choice", "real_a_is", "gm_snippet"])

        for i in range(max_len):
            a_line = lines_a[i] if i < len(lines_a) else ""
            b_line = lines_b[i] if i < len(lines_b) else ""
            swap = random.choice([True, False])
            shown_a = b_line if swap else a_line
            shown_b = a_line if swap else b_line
            real_a = "b" if swap else "a"

            choice = _present_pair(i, max_len, shown_a, shown_b, f"第{i+1}组")
            if choice == "q":
                break
            if choice == "s":
                continue

            # 解码盲测结果
            if choice == "1":
                real_winner = real_a
            elif choice == "2":
                real_winner = "b" if real_a == "a" else "a"
            else:
                real_winner = "tie"

            writer.writerow([i, real_winner, real_a, f"组{i+1}"])
            print(f"  → 记录: {real_winner}")

    print(f"\n结果已保存: {out_path}")


def run_prompt_pairs(tag: str, prompt_path: str):
    """标准格式 JSONL: {"gm_report": "...", "output_a": "...", "output_b": "..."}"""
    import json

    pairs = [
        json.loads(line)
        for line in Path(prompt_path).read_text(encoding="utf-8").strip().split("\n")
        if line.strip()
    ]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"{tag}_{stamp}.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "choice", "real_a_is", "gm_snippet"])

        for i, pair in enumerate(pairs):
            gm = pair.get("gm_report", "")
            a = pair["output_a"]
            b = pair["output_b"]
            swap = random.choice([True, False])
            shown_a = b if swap else a
            shown_b = a if swap else b
            real_a = "b" if swap else "a"

            choice = _present_pair(i, len(pairs), shown_a, shown_b, gm)
            if choice == "q":
                break
            if choice == "s":
                continue

            if choice == "1":
                real_winner = real_a
            elif choice == "2":
                real_winner = "b" if real_a == "a" else "a"
            else:
                real_winner = "tie"

            writer.writerow([i, real_winner, real_a, gm[:100]])
            print(f"  → 记录: {real_winner}")

    print(f"\n结果已保存: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="标签，如 r1_sft_v1")
    ap.add_argument("--interleaved", nargs=2, metavar=("FILE_A", "FILE_B"))
    ap.add_argument("--prompts", help="JSONL 文件路径")
    args = ap.parse_args()

    if args.interleaved:
        run_interleaved(args.tag, args.interleaved[0], args.interleaved[1])
    elif args.prompts:
        run_prompt_pairs(args.tag, args.prompts)
    else:
        print("请指定 --interleaved 或 --prompts")
        sys.exit(1)


if __name__ == "__main__":
    main()
