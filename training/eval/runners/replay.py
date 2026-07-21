# SPDX-License-Identifier: Apache-2.0
"""会话回放器 — 从 JSONL 数据集逐轮调用模型生成输出。

JSONL 格式（每行一个完整会话）:
  {
    "session_id": "xxx",
    "turns": [
      {"turn": 1, "system_prompt": "...", "user_prompt": "...", "max_tokens": 800},
      ...
    ]
  }

输出: 每轮增加 "model_output" 字段，回写到 output_path。

用法:
  python3 training/eval/runners/replay.py --input g_set.jsonl --output results_triton_pa.jsonl
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


async def replay_session(
    client: AsyncOpenAI,
    model: str,
    session: dict[str, Any],
    verbose: bool = False,
) -> dict[str, Any]:
    """回放一个会话的所有轮次，返回带 model_output 的 session 副本。"""
    turns = session.get("turns", [])
    out_turns: list[dict[str, Any]] = []

    for t in turns:
        t0 = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": t["system_prompt"]},
                    {"role": "user", "content": t["user_prompt"]},
                ],
                max_tokens=t.get("max_tokens", 800),
                temperature=t.get("temperature", 0.8),
            )
            text = resp.choices[0].message.content or ""
            usage = resp.usage
            elapsed = time.perf_counter() - t0
            out_turns.append({
                **t,
                "model_output": text,
                "ttft_s": round(elapsed, 3),
                "usage": {
                    "prompt_tokens": usage.prompt_tokens if usage else -1,
                    "completion_tokens": usage.completion_tokens if usage else -1,
                },
            })
            if verbose:
                print(f"  turn {t['turn']} ok {elapsed:.1f}s ({usage.completion_tokens if usage else '?'} tok)")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            out_turns.append({
                **t,
                "model_output": "",
                "ttft_s": round(elapsed, 3),
                "error": str(e),
            })
            if verbose:
                print(f"  turn {t['turn']} ERROR {e}")

    return {**session, "turns": out_turns}


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="JSONL 数据集")
    ap.add_argument("--output", required=True, help="输出 JSONL")
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--model", default="qwen2.5-7b-baseline")
    ap.add_argument("--max-samples", type=int, default=0, help="限制会话数, 0=全部")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    lines = [
        l for l in Path(args.input).read_text(encoding="utf-8").split("\n") if l.strip()
    ]
    sessions = [json.loads(l) for l in lines]
    if args.max_samples > 0:
        sessions = sessions[: args.max_samples]

    client = AsyncOpenAI(api_key="not-needed", base_url=args.base_url)

    results = []
    for i, sess in enumerate(sessions):
        if args.verbose:
            print(f"[{i+1}/{len(sessions)}] session {sess.get('session_id', i)}")
        result = await replay_session(client, args.model, sess, verbose=args.verbose)
        results.append(result)

    Path(args.output).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results),
        encoding="utf-8",
    )
    print(f"完成: {len(results)} sessions → {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
