# SPDX-License-Identifier: Apache-2.0
"""infer.kernels — 框架无关 Triton 算子（AMD/Ascend 共享核心）。

仅依赖 torch + triton，不 import vllm 或 sglang。
"""

import torch

__version__ = "0.1.0"


def detect_platform() -> str:
    if torch.version.hip is not None:
        return "amd"
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "ascend"
    if torch.cuda.is_available():
        return "cuda"
    return "unknown"
