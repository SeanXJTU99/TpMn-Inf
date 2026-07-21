#!/usr/bin/env python3
"""Phase 1 基线 benchmark — 游戏真实负载形态。

对 vLLM OpenAI-compatible endpoint 测三类场景（与 game_server 实际调用形态一致）：
  narrator       中等上下文流式叙事（TTFT 敏感）
  narrator_long  深上下文叙事（长上下文退化观察）
  arbiter        长上下文 JSON 裁定（总延迟敏感）

指标：TTFT / E2E 延迟（P50/P95）、decode tok/s、实际 prompt/completion tokens。
结果落盘 results/baseline_<tag>.json，供 Phase 2 每个 Triton kernel 替换后对比。

用法:
  python3 baseline_bench.py --runs 5
  python3 baseline_bench.py --scenario narrator --runs 10 --tag after_triton_pa
"""

import argparse
import asyncio
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# 游戏风格合成负载（正式 G-Set 就绪后可替换为真实样本）
# ---------------------------------------------------------------------------

NARRATOR_SYSTEM = (
    "你是圣杯战争的叙事者，以虚渊玄的冷峻文风、第二人称「你」向玩家叙述战局。"
    "只做战术暗示，不直述裁定结果，不使用 markdown。"
)

ARBITER_SYSTEM = (
    "你是圣杯战争的裁定引擎。根据规则与双方面板裁定行动结果，"
    "严格输出 JSON：{\"result\": str, \"damage\": int, \"state_changes\": [...], \"narration_hints\": [...]}。"
)

# 游戏味填充段（约 120 字/段），按目标字数重复拼接模拟历史上下文
_FILLER = (
    "第{i}回合：未远川的雾气尚未散尽，Lancer 的枪尖在桥面划出火星。你的 Servant 灵体化后撤，"
    "魔力回路的灼痛提醒你今夜已消耗过半。教会方向传来钟声，监督者对昨夜仓库街的爆炸保持沉默。"
    "Assassin 的气配遮断仍未解除，河岸边残留的魔力痕迹指向柳洞寺。"
)

GM_REPORT = (
    "【GM报告】回合17夜间：玩家Master(远坂)与Saber于新都大桥遭遇Berserker。"
    "Saber HP 62/100，魔力储备40%；Berserker狂化Rank B，其Master藏于300m外废楼。"
    "玩家剩余令咒2划。上回合玩家使用了宝具真名解放，已被敌方阵营观测。"
    "环境：暴雨，视野受限，桥面民众已疏散。请生成本回合叙事。"
)


def build_prompt_chars(n_chars: int) -> str:
    """拼出约 n_chars 字的历史上下文（Qwen 中文 ≈ 0.8-1 token/字）。"""
    parts, i = [], 1
    while sum(len(p) for p in parts) < n_chars:
        parts.append(_FILLER.format(i=i))
        i += 1
    return "".join(parts)[:n_chars]


SCENARIOS = {
    # name: (system, 上下文目标字数, max_tokens, temperature, json_mode)
    "narrator": (NARRATOR_SYSTEM, 3000, 800, 0.8, False),
    "narrator_long": (NARRATOR_SYSTEM, 12000, 800, 0.8, False),
    "arbiter": (ARBITER_SYSTEM, 8000, 1500, 0.3, True),
}

# ---------------------------------------------------------------------------


async def run_once(client: AsyncOpenAI, model: str, scenario: str) -> dict:
    system, ctx_chars, max_tokens, temp, json_mode = SCENARIOS[scenario]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": build_prompt_chars(ctx_chars) + "\n\n" + GM_REPORT},
    ]
    kwargs = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temp,
        stream=True,
        stream_options={"include_usage": True},
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    t0 = time.perf_counter()
    ttft = None
    usage = None
    n_chunks = 0
    stream = await client.chat.completions.create(**kwargs)
    async for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - t0
            n_chunks += 1
    t_end = time.perf_counter()

    e2e = t_end - t0
    completion_tokens = usage.completion_tokens if usage else n_chunks
    prompt_tokens = usage.prompt_tokens if usage else -1
    decode_time = e2e - (ttft or 0.0)
    return {
        "ttft_s": ttft,
        "e2e_s": e2e,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "decode_tok_s": completion_tokens / decode_time if decode_time > 0 else 0.0,
    }


def pctl(values: list, p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return s[k]


async def bench_scenario(client, model, scenario, runs, warmup) -> dict:
    for _ in range(warmup):
        await run_once(client, model, scenario)
    samples = [await run_once(client, model, scenario) for _ in range(runs)]

    ttfts = [s["ttft_s"] for s in samples if s["ttft_s"] is not None]
    e2es = [s["e2e_s"] for s in samples]
    toks = [s["decode_tok_s"] for s in samples]
    return {
        "runs": runs,
        "prompt_tokens": samples[0]["prompt_tokens"],
        "ttft_p50_s": round(pctl(ttfts, 50), 3),
        "ttft_p95_s": round(pctl(ttfts, 95), 3),
        "e2e_p50_s": round(pctl(e2es, 50), 3),
        "e2e_p95_s": round(pctl(e2es, 95), 3),
        "decode_tok_s_mean": round(statistics.mean(toks), 1) if toks else 0.0,
        "samples": samples,
    }


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--model", default="qwen2.5-7b-baseline")
    ap.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--tag", default="baseline", help="结果文件名标签，如 after_triton_pa")
    args = ap.parse_args()

    client = AsyncOpenAI(api_key="not-needed", base_url=args.base_url)
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]

    report = {
        "tag": args.tag,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "base_url": args.base_url,
        "model": args.model,
        "scenarios": {},
    }
    for name in names:
        print(f"[{name}] warmup={args.warmup} runs={args.runs} ...", flush=True)
        report["scenarios"][name] = await bench_scenario(
            client, args.model, name, args.runs, args.warmup
        )

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{args.tag}_{stamp}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 终端汇总表
    print(f"\n| scenario | prompt_tok | TTFT p50/p95 | E2E p50/p95 | decode tok/s |")
    print(f"|---|---|---|---|---|")
    for name, r in report["scenarios"].items():
        print(
            f"| {name} | {r['prompt_tokens']} "
            f"| {r['ttft_p50_s']}s / {r['ttft_p95_s']}s "
            f"| {r['e2e_p50_s']}s / {r['e2e_p95_s']}s "
            f"| {r['decode_tok_s_mean']} |"
        )
    print(f"\n结果已存: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
