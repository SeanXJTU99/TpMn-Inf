#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""RDNA3 kernel 自动调参 —— 参数网格搜索 + 最优组合写入 amd_tune.json。

策略: 两阶段分治，避免组合爆炸：
  1. coarse: tile_size × num_warps  (最大搜索空间，决定占用率)
  2. fine:   固定上述最优后 sweep num_stages × waves_per_eu × segments

每组合跑 baseline_bench.py --scenario narrator（TTFT 敏感），
以 TTFT p50 为主指标，E2E p50 为副指标，decode tok/s 为参考。

用法:
  # 完整 sweep（需要 vLLM server 在 localhost:8080）
  python3 tune_attention.py --kernel decode

  # 仅 coarse pass
  python3 tune_attention.py --kernel decode --coarse-only

  # resume 上次中断的 sweep
  python3 tune_attention.py --kernel decode --resume /tmp/tune_checkpoint.json

产出:
  infer/amdk/amd_tune.json  — 最优配置（vLLM 重启后自动加载）
  infer/bench/results/tune_*.json  — 各组合 benchmark 结果
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 参数搜索空间
# ---------------------------------------------------------------------------
# tile_size: RDNA3 wave32 → 32 为 natural wave 宽度，64 摊薄地址计算
# num_warps: 2=轻量(高 occupancy), 4=中等, 8=重(register pressure, LDS 竞争)
# num_stages: 1=RDNA3 无软件流水(推荐), 2=试一下
# waves_per_eu: AMD 专属 occupancy 提示，None=不传，2/4=不同 CU 占用策略
# segments: decode 3D kernel 的 softmax 分段数

COARSE_GRID_DECODE = {
    "tile_size": [32, 48, 64, 96],
    "num_warps": [2, 4, 8],
}

COARSE_GRID_PREFILL = {
    "tile_size": [32, 64, 96, 128],
    "num_warps": [2, 4, 8],
}

FINE_GRID = {
    "num_stages": [1, 2],
    "waves_per_eu": [None, 1, 2, 4],
    "num_softmax_segments": [8, 16, 24, 32],  # 仅 decode
}

# 基准 tag — 扫参期间固定
BENCH_SCENARIO = "narrator"
BENCH_RUNS = 3
WARMUP = 1

BENCH_SCRIPT = Path(__file__).parents[1] / "bench" / "baseline_bench.py"
TUNE_CFG_OUT = Path(__file__).parent / "amd_tune.json"


@dataclass
class TrialResult:
    cfg: dict
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    e2e_p50: float = 0.0
    decode_tok_s: float = 0.0
    error: str = ""

    @property
    def score(self) -> float:
        """越小越好 = TTFT p50 (主指标)。"""
        return self.ttft_p50

    def is_valid(self) -> bool:
        return self.error == "" and self.ttft_p50 > 0


def _check_server(base_url: str) -> bool:
    """确认 vLLM server 存活。"""
    import urllib.request
    try:
        urllib.request.urlopen(f"{base_url}/models", timeout=5)
        return True
    except Exception:
        return False


def _run_benchmark(tag: str, base_url: str, model: str) -> TrialResult:
    """调用 baseline_bench.py，解析 JSON → TrialResult。"""
    cmd = [
        sys.executable, str(BENCH_SCRIPT),
        "--base-url", base_url,
        "--model", model,
        "--scenario", BENCH_SCENARIO,
        "--runs", str(BENCH_RUNS),
        "--warmup", str(WARMUP),
        "--tag", tag,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            return TrialResult(cfg={}, error=proc.stderr[:500])

        # 找到最新产出的 JSON
        out_dir = Path(__file__).parents[1] / "bench" / "results"
        jsons = sorted(out_dir.glob(f"{tag}_*.json"), key=lambda p: p.stat().st_mtime)
        if not jsons:
            return TrialResult(cfg={}, error="no JSON output found")

        data = json.loads(jsons[-1].read_text(encoding="utf-8"))
        sc = data["scenarios"].get(BENCH_SCENARIO, {})
        return TrialResult(
            cfg={},
            ttft_p50=sc.get("ttft_p50_s", 0),
            ttft_p95=sc.get("ttft_p95_s", 0),
            e2e_p50=sc.get("e2e_p50_s", 0),
            decode_tok_s=sc.get("decode_tok_s_mean", 0),
        )
    except subprocess.TimeoutExpired:
        return TrialResult(cfg={}, error="benchmark timeout")
    except Exception as e:
        return TrialResult(cfg={}, error=str(e))


def _write_tune_cfg(cfg: dict, path: Path):
    """写临时 amd_tune.json（vLLM 不重启不生效，但 baseline_bench.py 调用时
    可通过 AMDK_TUNE_CONFIG env 指向该文件让 python 进程内生效）。

    实际扫参每轮需要重启 vLLM server —— 简化：只写最终结果到 amd_tune.json，
    中间扫参时通过 --extra-args 传给 vLLM（或直接改共享内存 env）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def _checkpoint_path(kernel: str) -> Path:
    return Path(f"/tmp/amdk_tune_{kernel}_checkpoint.json")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"completed": [], "results": {}}


def _save_checkpoint(path: Path, state: dict):
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# main sweep
# ---------------------------------------------------------------------------

def sweep_coarse(
    kernel: str,
    grid: dict[str, list],
    base_url: str,
    model: str,
    checkpoint: dict,
    ckpt_path: Path,
) -> list[TrialResult]:
    """Coarse sweep: tile_size × num_warps。其他用默认值。"""
    results: list[TrialResult] = []
    tiles = grid["tile_size"]
    warps = grid["num_warps"]
    total = len(tiles) * len(warps)
    done = set(tuple(r["cfg"].items()) if isinstance(r, dict) else ("", "") for r in checkpoint["completed"])

    idx = 0
    for ts in tiles:
        for nw in warps:
            cfg = {"tile_size": ts, "num_warps": nw}
            cfg_key = json.dumps(cfg, sort_keys=True)
            if cfg_key in done:
                idx += 1
                prev = checkpoint["results"].get(cfg_key, {})
                results.append(TrialResult(cfg=cfg, ttft_p50=prev.get("ttft_p50", 0)))
                continue

            tag = f"tune_{kernel}_coarse_{ts}_{nw}"
            print(f"[coarse {idx+1}/{total}] tile={ts} warps={nw}  ", end="", flush=True)
            trial = _run_benchmark(tag, base_url, model)
            print(f"ttft={trial.ttft_p50:.3f}s e2e={trial.e2e_p50:.3f}s" if trial.is_valid() else f"FAIL: {trial.error[:80]}")
            trial.cfg = cfg
            results.append(trial)

            checkpoint["completed"].append({"cfg": cfg})
            checkpoint["results"][cfg_key] = {
                "ttft_p50": trial.ttft_p50,
                "e2e_p50": trial.e2e_p50,
                "decode_tok_s": trial.decode_tok_s,
            }
            _save_checkpoint(ckpt_path, checkpoint)
            idx += 1

    return results


def sweep_fine(
    kernel: str,
    best_coarse: dict,
    grid: dict[str, list],
    base_url: str,
    model: str,
    checkpoint: dict,
    ckpt_path: Path,
) -> list[TrialResult]:
    """Fine sweep: stages × waves_per_eu × segments（以 coarse 最优为基础）。

    num_softmax_segments 仅 decode kernel 使用，prefill 跳过。
    """
    results: list[TrialResult] = []
    stages_list = grid.get("num_stages", [1])
    waves_list = grid.get("waves_per_eu", [None])
    # segments 仅 decode 有效
    segs_list = grid.get("num_softmax_segments", [16]) if kernel == "decode" else [16]

    total = len(stages_list) * len(waves_list) * len(segs_list)
    idx = 0

    for ns in stages_list:
        for wpe in waves_list:
            for seg in segs_list:
                cfg = {**best_coarse, "num_stages": ns, "waves_per_eu": wpe,
                       "num_softmax_segments": seg}
                tag = f"tune_{kernel}_fine_s{ns}_w{wpe}_sg{seg}"
                print(f"[fine {idx+1}/{total}] stages={ns} wpe={wpe} seg={seg}  ",
                      end="", flush=True)
                trial = _run_benchmark(tag, base_url, model)
                print(f"ttft={trial.ttft_p50:.3f}s" if trial.is_valid() else f"FAIL: {trial.error[:80]}")
                trial.cfg = cfg
                results.append(trial)
                idx += 1

    return results


def pick_best(results: list[TrialResult]) -> dict:
    valid = [r for r in results if r.is_valid()]
    if not valid:
        raise RuntimeError("无有效 trial——所有组合均失败")

    best = min(valid, key=lambda r: r.score)
    # Break ties by E2E
    tied = [r for r in valid if abs(r.score - best.score) < 0.001]
    if len(tied) > 1:
        best = min(tied, key=lambda r: r.e2e_p50)

    return {
        "tile_size": best.cfg.get("tile_size"),
        "num_warps": best.cfg.get("num_warps"),
        "num_stages": best.cfg.get("num_stages", 1),
        "waves_per_eu": best.cfg.get("waves_per_eu"),
        "num_softmax_segments": best.cfg.get("num_softmax_segments", 16),
        "_score_ttft_p50": best.ttft_p50,
        "_score_e2e_p50": best.e2e_p50,
        "_score_decode_tok_s": best.decode_tok_s,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kernel", choices=["decode", "prefill"], default="decode",
                    help="目标 kernel 类型 (default: decode)")
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--model", default="qwen2.5-7b-baseline")
    ap.add_argument("--coarse-only", action="store_true",
                    help="仅跑 coarse sweep（不跑 fine）")
    ap.add_argument("--resume", help="从 checkpoint JSON 恢复上次中断的 sweep")
    ap.add_argument("--dry-run", action="store_true",
                    help="打印搜索空间但不实际跑 benchmark")
    args = ap.parse_args()

    coarse_grid = COARSE_GRID_DECODE if args.kernel == "decode" else COARSE_GRID_PREFILL
    fine_grid = FINE_GRID

    if args.dry_run:
        print(f"=== Coarse grid ({args.kernel}) ===")
        tiles = coarse_grid["tile_size"]
        warps = coarse_grid["num_warps"]
        print(f"  tile_size × num_warps: {tiles} × {warps} = {len(tiles)*len(warps)} trials")
        if not args.coarse_only:
            stages = fine_grid["num_stages"]
            waves = fine_grid["waves_per_eu"]
            segs = fine_grid["num_softmax_segments"] if args.kernel == "decode" else ["(skip)"]
            print(f"  fine: stages={stages} waves_per_eu={waves} segments={segs} = {len(stages)*len(waves)*len(segs)} trials")
        return

    # server check
    if not _check_server(args.base_url):
        print(f"ERROR: vLLM server 未响应 ({args.base_url}/models)")
        print("请先启动: bash infer/scripts/launch_baseline.sh")
        sys.exit(1)

    ckpt_path = _checkpoint_path(args.kernel)
    checkpoint = _load_checkpoint(ckpt_path) if not args.resume else _load_checkpoint(Path(args.resume))

    # 1. Coarse sweep
    print(f"\n{'='*50}\n  Coarse sweep: {args.kernel} ({len(coarse_grid['tile_size'])*len(coarse_grid['num_warps'])} trials)\n{'='*50}")
    coarse_results = sweep_coarse(
        args.kernel, coarse_grid, args.base_url, args.model, checkpoint, ckpt_path
    )
    best_coarse = pick_best(coarse_results)
    print(f"\nBest coarse: tile={best_coarse['tile_size']} warps={best_coarse['num_warps']} "
          f"→ ttft={best_coarse['_score_ttft_p50']:.3f}s")

    if args.coarse_only:
        _write_tune_cfg(best_coarse, TUNE_CFG_OUT)
        print(f"已写入: {TUNE_CFG_OUT}")
        return

    # 2. Fine sweep
    n_fine = len(fine_grid["num_stages"]) * len(fine_grid["waves_per_eu"]) * len(fine_grid["num_softmax_segments"])
    print(f"\n{'='*50}\n  Fine sweep: ({n_fine} trials)\n{'='*50}")
    fine_results = sweep_fine(
        args.kernel, best_coarse, fine_grid, args.base_url, args.model, checkpoint, ckpt_path
    )
    all_results = coarse_results + fine_results
    best = pick_best(all_results)

    _write_tune_cfg(best, TUNE_CFG_OUT)
    print(f"\n{'='*50}")
    print(f"  ✅ 最优: tile={best['tile_size']} warps={best['num_warps']} "
          f"stages={best['num_stages']} wpe={best['waves_per_eu']} "
          f"seg={best['num_softmax_segments']}")
    print(f"  ttft_p50={best['_score_ttft_p50']:.3f}s e2e_p50={best['_score_e2e_p50']:.3f}s "
          f"decode={best['_score_decode_tok_s']:.0f} tok/s")
    print(f"  已写入: {TUNE_CFG_OUT}")
    print(f"  重启 vLLM serve 生效（或 export AMDK_TUNE_CONFIG={TUNE_CFG_OUT}）")


if __name__ == "__main__":
    main()
