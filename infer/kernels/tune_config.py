# SPDX-License-Identifier: Apache-2.0
"""多平台 kernel 调参配置。

所有 tile / launch 参数集中于此，禁止在 kernel 内写死。按平台自动加载对应 JSON。

平台默认值：
  AMD RDNA3 (gfx1100, wave32, 96 CU, LDS 64KB/CU):
    prefill TILE=64, num_stages=1（RDNA3 软件流水收益低）
    decode  TILE=32, num_stages=1

  Ascend 910B (达芬奇, UB 192KB, L1 512KB):
    prefill TILE=64, num_stages=2（双缓冲隐藏 DMA）
    decode  TILE=64, num_stages=1（GEMV 路径，非 Cube）
    MTE 32B 对齐，Cube 512B 对齐，tile 16 整数倍
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

_ENV_KEY = "AMDK_TUNE_CONFIG"  # 指向 JSON 覆盖文件

Platform = Literal["amd", "ascend", "nvidia", "unknown"]


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

    # Ascend 专用：P1/P3 的 GEMV BLOCK 参数
    ascend_gemv_block_k: int = 128
    ascend_gemv_block_n: int = 128


def detect_platform() -> Platform:
    """检测当前硬件平台.

    检查顺序: ROCm → Ascend NPU → CUDA → unknown.
    """
    import torch  # 延迟导入，config 模块可能在 torch 安装前被导入

    try:
        if torch.version.hip is not None:
            return "amd"
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "ascend"
        if torch.cuda.is_available():
            return "nvidia"
    except Exception:
        pass
    return "unknown"


def _load_kernel_cfg(d: dict, default: KernelLaunchCfg) -> KernelLaunchCfg:
    return KernelLaunchCfg(
        tile_size=d.get("tile_size", default.tile_size),
        num_warps=d.get("num_warps", default.num_warps),
        num_stages=d.get("num_stages", default.num_stages),
        waves_per_eu=d.get("waves_per_eu", default.waves_per_eu),
    )


def _platform_defaults(platform: Platform) -> AttnTuneCfg:
    """返回各平台的代码默认值（未实测的起点）。"""
    if platform == "ascend":
        return AttnTuneCfg(
            prefill=KernelLaunchCfg(
                tile_size=64, num_warps=2, num_stages=2, waves_per_eu=None
            ),
            decode=KernelLaunchCfg(
                tile_size=64, num_warps=1, num_stages=1, waves_per_eu=None
            ),
            # Ascend: 3D softmax 分段对 GEMV 无意义，用 2D 路径，段数置 NA
            num_softmax_segments=-1,
            ascend_gemv_block_k=128,
            ascend_gemv_block_n=128,
        )
    return AttnTuneCfg()  # AMD / nvidia / unknown — 用原默认值


def _tune_json_name(platform: Platform) -> str:
    return f"{platform}_tune.json"


def load_tune_cfg(platform: Platform | None = None) -> AttnTuneCfg:
    """优先级: $AMDK_TUNE_CONFIG > 包内 <platform>_tune.json > 平台代码默认。

    platform: 自动检测或手动覆盖。
    """
    if platform is None:
        platform = detect_platform()
    default = _platform_defaults(platform)

    path = os.environ.get(_ENV_KEY)
    if path is None:
        candidate = Path(__file__).parent / _tune_json_name(platform)
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
        ascend_gemv_block_k=data.get(
            "ascend_gemv_block_k", default.ascend_gemv_block_k
        ),
        ascend_gemv_block_n=data.get(
            "ascend_gemv_block_n", default.ascend_gemv_block_n
        ),
    )


TUNE = load_tune_cfg()
