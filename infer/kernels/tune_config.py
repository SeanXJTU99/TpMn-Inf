# SPDX-License-Identifier: Apache-2.0
"""RDNA3 kernel 调参配置。

所有 tile / launch 参数集中于此，禁止在 kernel 内写死——实测调参
（infer/amdk/tune_attention.py，待写）产出 JSON 覆盖默认值。

默认值是 gfx1100 (wave32, 96 CU, LDS 64KB/CU) 的**未实测起点**：
- prefill TILE 64：上游 CUDA 版为 32；RDNA3 无 async-copy，加大 tile
  以摊薄地址计算与 launch 开销，换更高 HBM 利用率（960GB/s 带宽瓶颈）
- num_stages=1：RDNA3 软件流水收益低，多 stage 徒增 LDS 压力
- waves_per_eu：AMD 专属 occupancy 提示，CUDA 上不可传（launch 时守卫）
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

_ENV_KEY = "AMDK_TUNE_CONFIG"  # 指向 JSON 覆盖文件


@dataclass(frozen=True)
class KernelLaunchCfg:
    tile_size: int
    num_warps: int
    num_stages: int
    waves_per_eu: int | None  # None = 不传（非 ROCm 环境）


@dataclass(frozen=True)
class AttnTuneCfg:
    prefill: KernelLaunchCfg = field(
        default_factory=lambda: KernelLaunchCfg(
            tile_size=64, num_warps=4, num_stages=1, waves_per_eu=2
        )
    )
    decode: KernelLaunchCfg = field(
        default_factory=lambda: KernelLaunchCfg(
            tile_size=32, num_warps=2, num_stages=1, waves_per_eu=4
        )
    )
    # decode 3D kernel 的并行 softmax 段数（上游写死 16）
    num_softmax_segments: int = 16


def _load_kernel_cfg(d: dict, default: KernelLaunchCfg) -> KernelLaunchCfg:
    return KernelLaunchCfg(
        tile_size=d.get("tile_size", default.tile_size),
        num_warps=d.get("num_warps", default.num_warps),
        num_stages=d.get("num_stages", default.num_stages),
        waves_per_eu=d.get("waves_per_eu", default.waves_per_eu),
    )


def load_tune_cfg() -> AttnTuneCfg:
    """优先级: $AMDK_TUNE_CONFIG 指定的 JSON > 包内 amd_tune.json > 代码默认。"""
    default = AttnTuneCfg()
    path = os.environ.get(_ENV_KEY)
    if path is None:
        candidate = Path(__file__).parent / "amd_tune.json"
        path = str(candidate) if candidate.exists() else None
    if path is None:
        return default

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return AttnTuneCfg(
        prefill=_load_kernel_cfg(data.get("prefill", {}), default.prefill),
        decode=_load_kernel_cfg(data.get("decode", {}), default.decode),
        num_softmax_segments=data.get(
            "num_softmax_segments", default.num_softmax_segments
        ),
    )


TUNE = load_tune_cfg()
