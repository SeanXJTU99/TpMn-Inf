# SPDX-License-Identifier: Apache-2.0
"""P3: Fused GEGLU + FFN (Qwen2.5-7B decode 专优化)。

融合链: hidden → gate_up_proj → SiLU(gate)*up → down_proj
上游: 3 kernel launch (gate_up GEMM + SiLU_and_Mul element-wise + down GEMM)
本 kernel: 全部融合为单 kernel，中间激活 [1, 18944] 不写回 HBM。

参考:
  - 上游 Qwen2MLP: vllm/model_executor/models/qwen2.py
  - 上游 SiLU: vllm/model_executor/layers/activation.py _swiglu_step_and_mul_kernel
  - 上游 MergedColumnParallelLinear / RowParallelLinear: linear.py

特化: Qwen2.5-7B — hidden=3584, intermediate=18944, SiLU activation
仅 decode 路径 (M=1)。prefill 走上游。
"""

import torch

from vllm.triton_utils import tl, triton

from amdk.tune_config import TUNE

_IS_HIP = torch.version.hip is not None


def _launch_extra(cfg) -> dict:
    extra = {"num_warps": cfg.num_warps, "num_stages": cfg.num_stages}
    if _IS_HIP and cfg.waves_per_eu is not None:
        extra["waves_per_eu"] = cfg.waves_per_eu
    return extra


@triton.jit
def _kernel_fused_geglu_ffn_decode(
    # 输入
    x_ptr,                 # [num_tokens, hidden_size]
    w_gate_up_ptr,         # [hidden_size, 2 * intermediate_size]
    w_down_ptr,            # [intermediate_size, hidden_size]
    # 输出
    out_ptr,               # [num_tokens, hidden_size]
    # 维度
    stride_x_0: tl.int64,
    stride_w_gate_0: tl.int64,
    stride_w_gate_1: tl.int64,
    stride_w_down_0: tl.int64,
    stride_w_down_1: tl.int64,
    stride_out_0: tl.int64,
    hidden_size: tl.constexpr,        # 3584
    intermediate_size: tl.constexpr,  # 18944
    # tiling
    BLOCK_K: tl.constexpr,   # hidden dim tile (reduce over this)
    BLOCK_N: tl.constexpr,   # intermediate tile (blocks over intermediate)
):
    """单 token decode kernel: tile intermediate, accumulate output in registers。

    grid = (cdiv(intermediate_size, BLOCK_N),)
    每 block 处理 BLOCK_N 个 intermediate 列，产出 partial output[hidden_size]。
    """
    pid = tl.program_id(0)
    token_idx = 0  # decode: 单 token

    i_start = pid * BLOCK_N
    offs_n = i_start + tl.arange(0, BLOCK_N)  # intermediate col indices
    n_mask = offs_n < intermediate_size

    # ---- Phase 1: gate/up 累加器 (bf16 → fp32) ----
    gate_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    up_acc   = tl.zeros([BLOCK_N], dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    for k in range(0, hidden_size, BLOCK_K):
        k_mask = (k + offs_k) < hidden_size

        # Load x[k:k+BLOCK_K]  [BLOCK_K]
        x_offs = token_idx * stride_x_0 + k + offs_k
        x_val = tl.load(x_ptr + x_offs, mask=k_mask, other=0.0).to(tl.float32)

        # Load W_gate[k:k+BLOCK_K, offs_n]  — gate 列 [BLOCK_K, BLOCK_N]
        w_gate_ptrs = (
            w_gate_up_ptr
            + (k + offs_k[:, None]) * stride_w_gate_0
            + offs_n[None, :] * stride_w_gate_1
        )
        w_gate = tl.load(w_gate_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)

        # Load W_up[k:k+BLOCK_K, offs_n+intermediate]  — up 列
        w_up_ptrs = (
            w_gate_up_ptr
            + (k + offs_k[:, None]) * stride_w_gate_0
            + (offs_n[None, :] + intermediate_size) * stride_w_gate_1
        )
        w_up = tl.load(w_up_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)

        # gate_acc[n] += sum_k(x[k] * w_gate[k, n])
        gate_acc += tl.sum(x_val[:, None] * w_gate, axis=0)
        up_acc   += tl.sum(x_val[:, None] * w_up,   axis=0)

    # ---- Phase 2: SiLU(gate) * up ----
    gate_silu = tl.sigmoid(gate_acc) * gate_acc  # SiLU
    activated = gate_silu * up_acc  # [BLOCK_N]

    # ---- Phase 3: down projection accumulate ----
    # out[h] += activated[n] * w_down[n, h] → reduce over n within this tile
    out_partial = tl.zeros([hidden_size], dtype=tl.float32)
    offs_h = tl.arange(0, hidden_size)

    # For each active n in this tile: out += activated[n] * w_down[offs_n[n], :]
    for n_local in range(BLOCK_N):
        actual_n = i_start + n_local
        if actual_n >= intermediate_size:
            break
        a_val = activated[n_local]  # scalar
        w_down_ptrs = (
            w_down_ptr
            + actual_n * stride_w_down_0
            + offs_h * stride_w_down_1
        )
        w_down_row = tl.load(w_down_ptrs, mask=offs_h < hidden_size, other=0.0).to(tl.float32)
        out_partial += a_val * w_down_row

    # ---- write (atomic add over intermediate tiles) ----
    out_offs = token_idx * stride_out_0 + offs_h
    tl.atomic_add(out_ptr + out_offs, out_partial, mask=offs_h < hidden_size)


# ---- Python entry point -------------------------------------------------------


def fused_geglu_ffn_decode(
    x: torch.Tensor,                # [1, hidden_size]
    w_gate_up: torch.Tensor,        # [hidden_size, 2 * intermediate_size]
    w_down: torch.Tensor,           # [intermediate_size, hidden_size]
) -> torch.Tensor:
    """单 token decode: RMSNorm(已在上层做完) → gate_up → SiLU(gate)*up → down。

    Returns: output [1, hidden_size]
    """
    num_tokens, hidden_size = x.shape
    assert num_tokens == 1, "decode path only"
    assert w_gate_up.shape[0] == hidden_size
    intermediate_size = w_down.shape[0]
    assert w_gate_up.shape[1] == 2 * intermediate_size
    assert w_down.shape[1] == hidden_size

    out = torch.zeros(num_tokens, hidden_size, dtype=x.dtype, device=x.device)

    cfg = TUNE.decode
    BLOCK_K = 128
    BLOCK_N = 512  # intermediate tile — 每 block 产出 partial output[hidden=3584] ≈ 7KB fp32

    _kernel_fused_geglu_ffn_decode[
        ((intermediate_size + BLOCK_N - 1) // BLOCK_N,)
    ](
        x_ptr=x,
        w_gate_up_ptr=w_gate_up,
        w_down_ptr=w_down,
        out_ptr=out,
        stride_x_0=x.stride(0),
        stride_w_gate_0=w_gate_up.stride(0),
        stride_w_gate_1=w_gate_up.stride(1),
        stride_w_down_0=w_down.stride(0),
        stride_w_down_1=w_down.stride(1),
        stride_out_0=out.stride(0),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
        **_launch_extra(cfg),
    )

    return out
