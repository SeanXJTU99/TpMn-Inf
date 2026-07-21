# SPDX-License-Identifier: Apache-2.0
"""P1: Fused RMSNorm + QKV Projection + RoPE (Qwen2.5-7B decode 专优化)。

融合链: hidden_states → RMSNorm → Linear(QKV) → reshape to heads → RoPE
上游分 3 次 kernel launch（norm / linear / rope），此处融合为单 kernel。

Qwen2.5-7B 特化:
  - GQA: num_q_heads=28, num_kv_heads=4, head_size=128, hidden_size=3584
  - neox-style RoPE: 相邻 dim pair 旋转，Qwen2 base_theta=1e6
  - 仅 bf16/fp16，无 FP8

GEMM 策略:
  AMD (RDNA3):    tl.sum(x * w, axis=0) Vector 内积 — GEMV M=1 场景 MFMA 不 gain
  Ascend (910B):  同上 Vector 内积 — Cube 对 M=1 GEMV 利用率 <5%，Vector > Cube

参考:
  - 上游 RMSNorm: vllm/model_executor/layers/layernorm.py
  - 上游 QKV: vllm/model_executor/layers/linear.py QKVParallelLinear
  - 上游 RoPE: vllm/model_executor/layers/rotary_embedding/
"""

import torch

from vllm.triton_utils import tl, triton

from .tune_config import TUNE

_IS_HIP = torch.version.hip is not None
_IS_ASCEND = hasattr(torch, "npu") and torch.npu.is_available()


def _launch_extra(cfg) -> dict:
    """AMD 专属 launch kwargs。"""
    extra = {"num_warps": cfg.num_warps, "num_stages": cfg.num_stages}
    if _IS_HIP and cfg.waves_per_eu is not None:
        extra["waves_per_eu"] = cfg.waves_per_eu
    return extra


# ---- 辅助: Triton 内 RMSNorm -------------------------------------------------


@triton.jit
def _rms_norm_factor(
    x_ptr,  # [num_tokens, hidden_size]
    stride_x_0: tl.int64,
    hidden_size: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """逐 token 计算 rms_norm 缩放因子：1 / sqrt(mean(x^2) + eps)。
    grid 为 (num_tokens,)，每 block 处理一个 token 的全部 hidden dims。
    """
    token_idx = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    x_offs = token_idx * stride_x_0 + offs_k
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for k in range(0, hidden_size, BLOCK_K):
        mask = (k + offs_k) < hidden_size
        x_val = tl.load(x_ptr + x_offs + k, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x_val * x_val)
    return tl.math.rsqrt(sum_sq / hidden_size + 1e-6)


# ---- decode kernel -----------------------------------------------------------


@triton.jit
def _kernel_fused_rms_qkv_rope_decode(
    # 输入
    x_ptr,  # [num_tokens, hidden_size]
    rms_weight_ptr,  # [hidden_size]  — RMSNorm weight
    qkv_weight_ptr,  # [hidden_size, total_qkv_size]
    q_bias_ptr,  # [q_size]  or None → 0
    k_bias_ptr,  # [kv_size]
    v_bias_ptr,  # [kv_size]
    # RoPE
    cos_ptr,  # [max_seq_len, head_size // 2]
    sin_ptr,  # [max_seq_len, head_size // 2]
    positions_ptr,  # [num_tokens]  — RoPE position id per token
    # 输出
    q_out_ptr,  # [num_tokens, q_size]
    k_out_ptr,  # [num_tokens, kv_size]
    v_out_ptr,  # [num_tokens, kv_size]
    # 维度
    stride_x_0: tl.int64,
    stride_qkv_0: tl.int64,
    stride_qkv_1: tl.int64,  # = 1
    stride_q_out_0: tl.int64,
    stride_k_out_0: tl.int64,
    stride_v_out_0: tl.int64,
    hidden_size: tl.constexpr,  # 3584
    q_size: tl.constexpr,  # num_q_heads * head_size = 28*128 = 3584
    kv_size: tl.constexpr,  # num_kv_heads * head_size = 4*128 = 512
    head_size: tl.constexpr,  # 128
    num_q_heads: tl.constexpr,  # 28
    num_kv_heads: tl.constexpr,  # 4
    # constexpr
    BLOCK_K: tl.constexpr,  # hidden dim tile（accumulate axis）
    HEADS_PER_BLOCK: tl.constexpr,  # 每个 block 处理的 head 数
    USE_Q_BIAS: tl.constexpr,
    USE_KV_BIAS: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,  # cos/sin table 最大位置
    HAS_V: tl.constexpr = True,
):
    pid = tl.program_id(0)
    token_idx = 0  # decode: 单 token
    pos = tl.load(positions_ptr + token_idx)

    # ---- 确定本 block 处理的 head 范围 ----
    total_q_blocks = (num_q_heads + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
    total_kv_blocks = (num_kv_heads + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK

    if pid < total_q_blocks:
        # Q head block
        is_q = True
        head_start = pid * HEADS_PER_BLOCK
        head_end = min(head_start + HEADS_PER_BLOCK, num_q_heads)
        num_local_heads = head_end - head_start
        out_size = num_local_heads * head_size
        out_base = head_start * head_size  # Q weight offset
        out_ptr = q_out_ptr + token_idx * stride_q_out_0 + out_base
        bias_ptr = q_bias_ptr + out_base if USE_Q_BIAS else None
    elif pid < total_q_blocks + total_kv_blocks:
        # K head block
        is_q = False
        is_v = False
        kv_pid = pid - total_q_blocks
        head_start = kv_pid * HEADS_PER_BLOCK
        head_end = min(head_start + HEADS_PER_BLOCK, num_kv_heads)
        num_local_heads = head_end - head_start
        out_size = num_local_heads * head_size
        out_base = head_start * head_size
        weight_offset = q_size + out_base  # K weight starts after Q
        out_ptr = k_out_ptr + token_idx * stride_k_out_0 + out_base
        bias_ptr = k_bias_ptr + out_base if USE_KV_BIAS else None
    else:
        # V head block
        is_q = False
        is_v = True
        kv_pid = pid - total_q_blocks - total_kv_blocks
        head_start = kv_pid * HEADS_PER_BLOCK
        head_end = min(head_start + HEADS_PER_BLOCK, num_kv_heads)
        num_local_heads = head_end - head_start
        out_size = num_local_heads * head_size
        out_base = head_start * head_size
        weight_offset = q_size + kv_size + out_base  # V weight after Q+K
        out_ptr = v_out_ptr + token_idx * stride_v_out_0 + out_base
        bias_ptr = v_bias_ptr + out_base if USE_KV_BIAS else None

    # ---- 1. RMSNorm: 全 hidden_size 归一化（decode 单 token 7KB，每 block 独立算） ----
    offs_k = tl.arange(0, BLOCK_K)
    rms_factor = _rms_norm_factor(
        x_ptr, stride_x_0, hidden_size, BLOCK_K,
    )

    # ---- 2. GEMM: x_hat @ W^T → Q/K/V 输出（reduce over hidden_size） ----
    # W layout: [hidden_size, total_qkv], stride_qkv_0 = total_qkv (= 1 维连续 HBM)
    # 对 Q block: 取列 [out_base, out_base+out_size); K/V block: [weight_offset, weight_offset+out_size)
    col_start = weight_offset if pid >= total_q_blocks else out_base
    acc = tl.zeros([out_size], dtype=tl.float32)
    offs_out = tl.arange(0, out_size)
    offs_w_col = col_start + offs_out  # [out_size] — W 的列索引

    for k in range(0, hidden_size, BLOCK_K):
        k_mask = (k + offs_k) < hidden_size
        # RMSNorm: load + normalize
        x_offs = token_idx * stride_x_0 + k + offs_k
        x_val = tl.load(x_ptr + x_offs, mask=k_mask, other=0.0).to(tl.float32)
        rms_w = tl.load(rms_weight_ptr + k + offs_k, mask=k_mask, other=0.0).to(
            tl.float32
        )
        x_norm = x_val * rms_factor * rms_w  # [BLOCK_K]

        # Load weight tile: W[k:k+BLOCK_K, col_start:col_start+out_size]
        # stride_qkv_0 = total_qkv, stride_qkv_1 = 1
        w_ptrs = (
            qkv_weight_ptr
            + (k + offs_k[:, None]) * stride_qkv_0  # row index [BLOCK_K, 1]
            + offs_w_col[None, :] * stride_qkv_1      # col index [1, out_size]
        )
        w_val = tl.load(w_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)
        # w_val: [BLOCK_K, out_size]

        # acc[out_size] += sum_k(x_norm[k] * w[k, out_size])
        acc += tl.sum(x_norm[:, None] * w_val, axis=0)

    # ---- Bias ----
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_out, mask=offs_out < out_size, other=0.0)
        acc = acc + bias.to(tl.float32)

    # ---- 3. RoPE (仅 Q / K) ----
    if not is_v:
        half = head_size // 2
        offs_h = tl.arange(0, half)
        cos_offs = pos * half + offs_h
        cos = tl.load(cos_ptr + cos_offs, mask=offs_h < half, other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr + cos_offs, mask=offs_h < half, other=0.0).to(tl.float32)

        # 逐 head 写回 RoPE 结果（neox-style：相邻 pair 旋转）
        for h in range(num_local_heads):
            h_off = h * head_size
            offs_even = h_off + offs_h * 2       # [0, 2, 4, ..., 126]
            offs_odd = h_off + offs_h * 2 + 1     # [1, 3, 5, ..., 127]
            x_even = tl.load_if_exists(acc, offs_even, 0.0)
            x_odd = tl.load_if_exists(acc, offs_odd, 0.0)
            rotated_even = x_even * cos - x_odd * sin
            rotated_odd = x_odd * cos + x_even * sin
            tl.store(out_ptr + offs_even, rotated_even, mask=offs_h < half)
            tl.store(out_ptr + offs_odd, rotated_odd, mask=offs_h < half)

        return  # RoPE 已逐 head 直接写回，跳过 acc 存储

    # ---- write (V, no RoPE) ----
    tl.store(out_ptr + offs_out, acc, mask=offs_out < out_size)


# ---- prefill kernel (placeholder) --------------------------------------------
# 预填充 path 先复用上游 TritonAttentionBackend 的拆分 kernel（正确性优先）。
# 后续定向重写一个 unified prefill kernel。


# ---- Python entry point -------------------------------------------------------


def fused_rms_qkv_rope_decode(
    x: torch.Tensor,  # [num_tokens, hidden_size]
    rms_weight: torch.Tensor,  # [hidden_size]
    qkv_weight: torch.Tensor,  # [hidden_size, total_qkv_size]
    q_bias: torch.Tensor | None,  # [q_size]
    k_bias: torch.Tensor | None,  # [kv_size]
    v_bias: torch.Tensor | None,  # [kv_size]
    cos_cache: torch.Tensor,  # [max_seq_len, head_size // 2]
    sin_cache: torch.Tensor,  # [max_seq_len, head_size // 2]
    positions: torch.Tensor,  # [num_tokens] int32
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """单 token decode 的融合 RMSNorm+QKV+RoPE。

    Returns: (q, k, v) — 每个 shape [num_tokens, heads * head_size]，Q/K 已 apply RoPE。
    """
    num_tokens, hidden_size = x.shape
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    total_qkv = q_size + 2 * kv_size
    assert qkv_weight.shape == (hidden_size, total_qkv), (
        f"weight shape {qkv_weight.shape} != ({hidden_size}, {total_qkv})"
    )
    assert num_tokens == 1, "decode path only"
    assert head_size % 2 == 0

    q_out = torch.empty(num_tokens, q_size, dtype=x.dtype, device=x.device)
    k_out = torch.empty(num_tokens, kv_size, dtype=x.dtype, device=x.device)
    v_out = torch.empty(num_tokens, kv_size, dtype=x.dtype, device=x.device)

    # 调参：decode 用专用的 decode tile
    cfg = TUNE.decode
    BLOCK_K = TUNE.ascend_gemv_block_k if _IS_ASCEND else 128
    # 每 block 处理的 head 数=1（Q:28 block, K:4 block, V:4 block = 36 block 总计）
    HEADS_PER_BLOCK = 1
    total_blocks = (
        (num_q_heads + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
        + (num_kv_heads + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK * 2
    )

    _kernel_fused_rms_qkv_rope_decode[(total_blocks,)](
        x_ptr=x,
        rms_weight_ptr=rms_weight,
        qkv_weight_ptr=qkv_weight,
        q_bias_ptr=q_bias if q_bias is not None else torch.empty(0, device=x.device),
        k_bias_ptr=k_bias if k_bias is not None else torch.empty(0, device=x.device),
        v_bias_ptr=v_bias if v_bias is not None else torch.empty(0, device=x.device),
        cos_ptr=cos_cache,
        sin_ptr=sin_cache,
        positions_ptr=positions,
        q_out_ptr=q_out,
        k_out_ptr=k_out,
        v_out_ptr=v_out,
        stride_x_0=x.stride(0),
        stride_qkv_0=qkv_weight.stride(0),
        stride_qkv_1=qkv_weight.stride(1),
        stride_q_out_0=q_out.stride(0),
        stride_k_out_0=k_out.stride(0),
        stride_v_out_0=v_out.stride(0),
        hidden_size=hidden_size,
        q_size=q_size,
        kv_size=kv_size,
        head_size=head_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        BLOCK_K=BLOCK_K,
        HEADS_PER_BLOCK=HEADS_PER_BLOCK,
        USE_Q_BIAS=q_bias is not None,
        USE_KV_BIAS=k_bias is not None,
        MAX_SEQ_LEN=cos_cache.shape[0],
    )

    return q_out, k_out, v_out
