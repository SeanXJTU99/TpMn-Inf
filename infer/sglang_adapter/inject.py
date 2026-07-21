# SPDX-License-Identifier: Apache-2.0
"""模型层注入 — 将 kernels/ RDNA3 融合 kernel 注入 SGLang 模型层。

用法（SGLang 模型加载后）:
  from sglang_adapter.inject import patch_geglu_ffn
  patch_geglu_ffn(model)

对比 vLLM adapter:
  vLLM: CUSTOM backend 自动接管 attention 路径
  SGLang: monkey-patch Qwen2MLP.forward 替换 FFN 为融合 kernel
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from kernels.fused_geglu_ffn import fused_geglu_ffn_decode
from kernels.tune_config import TUNE

logger = logging.getLogger(__name__)

# SGLang MergedColumnParallelLinear 存为 [out_features, in_features]
# 我们的 kernel 期望 [in_features, out_features]（vLLM row-major）
# 通过 stride 适配避免内存拷贝。
#
# SGLang gate_up_proj.weight:  [2*intermediate, hidden]，stride_0=hidden, stride_1=1
#   等价于 vLLM layout 的 .t()，只是 strides 互换
#
# SGLang down_proj.weight:     [hidden, intermediate]，stride_0=intermediate, stride_1=1
#   我们的 kernel 期望 [intermediate, hidden]，stride_0=hidden, stride_1=1
#   等价于 .t()


def _patch_mlp_forward(mlp_module, original_forward) -> None:
    """替换单个 Qwen2MLP 的 forward 为融合版本。"""

    def fused_forward(
        self, x: torch.Tensor, forward_batch: Any = None
    ) -> torch.Tensor:
        # 仅 decode（1 token）走融合路径；prefill 回退
        if x.shape[0] != 1:
            return original_forward(self, x, forward_batch)

        # SGLang 的 gate_up_proj 是 MergedColumnParallelLinear
        hidden_size = self.gate_up_proj.input_size  # SGLang 属性名
        w_gate_up = self.gate_up_proj.weight        # [2*intermediate, hidden]
        w_down = self.down_proj.weight               # [hidden, intermediate]

        result = fused_geglu_ffn_decode(
            x,
            w_gate_up.t().contiguous(),  # → [hidden, 2*intermediate]
            w_down.t().contiguous(),      # → [intermediate, hidden]
        )
        # SGLang RowParallelLinear 返回 (output, bias)，保持接口一致
        return result

    # 注入
    mlp_module.forward = fused_forward.__get__(mlp_module, type(mlp_module))


def patch_geglu_ffn(model: torch.nn.Module) -> int:
    """遍历模型所有 Qwen2DecoderLayer，替换 MLP forward 为融合版本。

    Returns: 替换的层数。
    """
    from sglang.srt.models.qwen2 import Qwen2DecoderLayer, Qwen2MLP

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, Qwen2MLP):
            original = Qwen2MLP.forward
            _patch_mlp_forward(module, original)
            count += 1

    logger.info("amdk: patched %d Qwen2MLP layers with fused GEGLU+FFN", count)
    return count


def patch_qkv_rope(model: torch.nn.Module) -> int:
    """P1 注入占位符。SGLang 上游已有 fused_qk_norm_rope_store.py，
    覆盖面：QK norm + RoPE + KV cache store。
    我们的 P1 (RMSNorm+QKV+RoPE) 在上述之外还覆盖 QKV 投影——
    待 kernel 适配 SGLang 的 QKVParallelLinear 权重布局后实现。
    """
    logger.warning("amdk: P1 injection not yet implemented for SGLang")
    return 0
