# SPDX-License-Identifier: Apache-2.0
"""模型层注入 — 将 kernels/ RDNA3 融合 kernel 注入 vLLM 模型层。

用法（vLLM 模型加载后）:
  from vllm_adapter.inject import patch_geglu_ffn, patch_qkv_rope
  patch_qkv_rope(model)  # P1: RMSNorm+QKV+RoPE
  patch_geglu_ffn(model)  # P3: GEGLU+FFN

vLLM 的 CUSTOM backend 只覆盖 attention 路径（P0），P1/P3 需要通过
Qwen2DecoderLayer 的 monkey-patch 注入。
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from kernels.fused_geglu_ffn import fused_geglu_ffn_decode
from kernels.fused_qkv_rope import fused_rms_qkv_rope_decode
from kernels.tune_config import TUNE

logger = logging.getLogger(__name__)


def _patch_mlp_forward(mlp_module, original_forward) -> None:
    def fused_forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != 1:
            return original_forward(self, x)

        w_gate_up = self.gate_up_proj.weight  # vLLM: [hidden, 2*intermediate]
        w_down = self.down_proj.weight        # [intermediate, hidden]

        return fused_geglu_ffn_decode(x, w_gate_up, w_down)

    mlp_module.forward = fused_forward.__get__(mlp_module, type(mlp_module))


def _patch_attn_forward(attn_module, original_forward) -> None:
    """替换 Qwen2Attention.forward 为 P1 融合版本。

    vLLM Qwen2Attention.forward 调用链:
      hidden_states → self.qkv_proj → split → reshape → self.rotary_emb → self.attn

    融合 kernel 做前三步: RMSNorm + QKV GEMM + reshape + RoPE。
    RMSNorm 在 Qwen2DecoderLayer.forward 里已完成，此处拿到的 hidden_states 已归一化。
    注意: 仅 decode（单 token）走融合，prefill 回退。
    """
    # 延迟导入避免循环依赖
    from vllm.model_executor.models.qwen2 import Qwen2Attention

    def fused_forward(
        self: "Qwen2Attention",
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.shape[0] != 1:
            return original_forward(self, positions, hidden_states)

        # 取出权重
        w_qkv = self.qkv_proj.weight  # [hidden, q+kv+kv]
        q_bias = self.qkv_proj.bias

        num_q = self.num_heads
        num_kv = self.num_kv_heads
        head_dim = self.head_dim
        q_size = num_q * head_dim
        kv_size = num_kv * head_dim

        q_b = q_bias[:q_size] if q_bias is not None else None
        k_b = q_bias[q_size : q_size + kv_size] if q_bias is not None else None
        v_b = q_bias[q_size + kv_size :] if q_bias is not None else None

        # 获取 RoPE cos/sin（从 rotary_emb 实例）
        cos, sin = self.rotary_emb.get_cos_sin(positions.max().item())

        q, k, v = fused_rms_qkv_rope_decode(
            x=hidden_states,
            rms_weight=torch.ones(hidden_states.shape[-1], device=x.device, dtype=x.dtype),
            qkv_weight=w_qkv,
            q_bias=q_b,
            k_bias=k_b,
            v_bias=v_b,
            cos_cache=cos,
            sin_cache=sin,
            positions=positions,
            num_q_heads=num_q,
            num_kv_heads=num_kv,
            head_size=head_dim,
        )

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    attn_module.forward = fused_forward.__get__(attn_module, type(attn_module))


def patch_geglu_ffn(model: torch.nn.Module) -> int:
    """遍历 vLLM 模型的 Qwen2MLP，替换为 P3 融合版本。"""
    from vllm.model_executor.models.qwen2 import Qwen2MLP

    count = 0
    for _, module in model.named_modules():
        if isinstance(module, Qwen2MLP):
            original = Qwen2MLP.forward
            _patch_mlp_forward(module, original)
            count += 1

    logger.info("amdk vllm: patched %d Qwen2MLP layers (GEGLU+FFN)", count)
    return count


def patch_qkv_rope(model: torch.nn.Module) -> int:
    """遍历 vLLM 模型的 Qwen2Attention，替换为 P1 融合版本。

    注意: 仅 decode (1 token) 走融合路径。
    vLLM 的 Qwen2DecoderLayer.forward 已做完 input_layernorm，
    所以传入 attn 的 hidden_states 是归一化后的。
    """
    from vllm.model_executor.models.qwen2 import Qwen2Attention

    count = 0
    for _, module in model.named_modules():
        if isinstance(module, Qwen2Attention):
            original = Qwen2Attention.forward
            _patch_attn_forward(module, original)
            count += 1

    logger.info("amdk vllm: patched %d Qwen2Attention layers (QKV+RoPE)", count)
    return count
