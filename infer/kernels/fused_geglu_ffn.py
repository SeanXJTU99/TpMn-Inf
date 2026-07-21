# SPDX-License-Identifier: Apache-2.0
"""P3: Fused GEGLU + FFN (Qwen2.5-7B decode 专优化)。

融合链: hidden → gate_up_proj → SiLU(gate)*up → down_proj
上游: 3 kernel launch (gate_up GEMM + SiLU_and_Mul element-wise + down GEMM)

AMD (RDNA3):     单 kernel 融合，tl.atomic_add 写回 partial output
                 3 launch → 1 launch

Ascend (910B):   两次 launch（gate_up GEMV → activation → down GEMV）
                 禁止 tl.atomic_add: HBM round-trip ~100-500ns/次，
                 P3 输出 4096 维 → 409μs/token（远超融合节省的 ~40μs）

参考:
  - 上游 Qwen2MLP: vllm/model_executor/models/qwen2.py
  - 上游 SiLU: vllm/model_executor/layers/activation.py _swiglu_step_and_mul_kernel
  - 上游 MergedColumnParallelLinear / RowParallelLinear: linear.py

特化: Qwen2.5-7B — hidden=3584, intermediate=18944, SiLU activation
仅 decode 路径 (M=1)。prefill 走上游。
"""

import torch

from vllm.triton_utils import tl, triton

from .tune_config import TUNE

_IS_HIP = torch.version.hip is not None
_IS_ASCEND = hasattr(torch, "npu") and torch.npu.is_available()


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


# ---- Ascend GEMV kernel (no tl.dot, no atomic_add) ---------------------------


@triton.jit
def _kernel_gemv_ascend(
    # 输入
    x_ptr,           # [1, in_features]
    w_ptr,           # [in_features, out_features]
    # 输出
    out_ptr,         # [out_features]
    # 维度
    stride_w_0: tl.int64,
    stride_w_1: tl.int64,  # = 1
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    # tiling
    BLOCK_K: tl.constexpr,  # in_features tile (reduce dim)
    BLOCK_N: tl.constexpr,  # out_features tile (output dim)
):
    """Vector GEMV: output[n_block] = sum_k x[k] * W[k, n_block].

    每 block 计算 BLOCK_N 个输出列，沿 K 维分块累加。
    使用 Vector 内积（tl.sum），不触发 Cube tl.dot。
    原因: M=1 时 Cube 利用率 <5%，Vector 路径反而更快。
    无 atomic_add — 直接写 HBM，每个 block 负责不交叠的输出区间。
    """
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)
    n_mask = offs_n < out_features

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_K)

    for k in range(0, in_features, BLOCK_K):
        k_mask = (k + offs_k) < in_features

        # Load x[k:k+BLOCK_K]  [BLOCK_K]
        x_val = tl.load(x_ptr + k + offs_k, mask=k_mask, other=0.0).to(tl.float32)

        # Load W[k:k+BLOCK_K, n_start:n_start+BLOCK_N]  [BLOCK_K, BLOCK_N]
        w_offs = (
            w_ptr
            + (k + offs_k[:, None]) * stride_w_0
            + offs_n[None, :] * stride_w_1
        )
        w_val = tl.load(w_offs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)

        # Vector inner-product: acc[n] += sum_k x[k] * W[k, n]
        acc += tl.sum(x_val[:, None] * w_val, axis=0)

    # Direct write — no atomic_add (each block owns non-overlapping output range)
    out_offs = n_start + offs_n
    tl.store(out_ptr + out_offs, acc, mask=n_mask)


# ---- Decode fusion kernel (AMD) ------------------------------------------------


def fused_geglu_ffn_decode(
    x: torch.Tensor,                # [1, hidden_size]
    w_gate_up: torch.Tensor,        # [hidden_size, 2 * intermediate_size]
    w_down: torch.Tensor,           # [intermediate_size, hidden_size]
) -> torch.Tensor:
    """单 token decode: RMSNorm(已在上层做完) → gate_up → SiLU(gate)*up → down。

    Returns: output [1, hidden_size]

    AMD: 单 kernel 融合（gate_up GEMM → SiLU → down GEMM → atomic_add）
    Ascend: 两次 launch（GEMV → activation → GEMV），无 atomic_add
    """
    num_tokens, hidden_size = x.shape
    assert num_tokens == 1, "decode path only"
    assert w_gate_up.shape[0] == hidden_size
    intermediate_size = w_down.shape[0]
    assert w_gate_up.shape[1] == 2 * intermediate_size
    assert w_down.shape[1] == hidden_size

    cfg = TUNE.decode

    if _IS_ASCEND:
        # ---- Ascend 路径: 两次 GEMV launch，无 atomic_add ----
        # 原因: tl.atomic_add = HBM round-trip（~100-500ns/次）
        #       单 token 4096 维 → ~409μs，远超融合节省的 launch overhead（~40μs）
        #       → 牺牲融合（2 launch）换确定性 + 低延迟
        BLOCK_K = TUNE.ascend_gemv_block_k  # 128
        BLOCK_N = TUNE.ascend_gemv_block_n  # 128

        # Launch 1: gate_up GEMV [1, hidden] @ [hidden, 2*intermediate] → [1, 2*intermediate]
        gate_up_out = torch.empty(1, 2 * intermediate_size, dtype=x.dtype, device=x.device)
        gate_up_2d = gate_up_out.view(-1)  # flatten to [2*intermediate]
        grid_1 = ((2 * intermediate_size + BLOCK_N - 1) // BLOCK_N,)
        _kernel_gemv_ascend[grid_1](
            x_ptr=x,
            w_ptr=w_gate_up,
            out_ptr=gate_up_2d,
            stride_w_0=w_gate_up.stride(0),
            stride_w_1=w_gate_up.stride(1),
            in_features=hidden_size,
            out_features=2 * intermediate_size,
            BLOCK_K=BLOCK_K,
            BLOCK_N=BLOCK_N,
            **_launch_extra(cfg),
        )

        # SiLU(gate) * up → activation [1, intermediate]
        gate = gate_up_out[:, :intermediate_size]
        up = gate_up_out[:, intermediate_size:]
        activated = torch.sigmoid(gate) * gate * up  # SiLU(gate) * up

        # Launch 2: down GEMV [1, intermediate] @ [intermediate, hidden] → [1, hidden]
        out = torch.empty(num_tokens, hidden_size, dtype=x.dtype, device=x.device)
        activated_1d = activated.view(-1)  # [intermediate]
        grid_2 = ((hidden_size + BLOCK_N - 1) // BLOCK_N,)
        _kernel_gemv_ascend[grid_2](
            x_ptr=activated_1d,
            w_ptr=w_down,
            out_ptr=out.view(-1),
            stride_w_0=w_down.stride(0),
            stride_w_1=w_down.stride(1),
            in_features=intermediate_size,
            out_features=hidden_size,
            BLOCK_K=BLOCK_K,
            BLOCK_N=BLOCK_N,
            **_launch_extra(cfg),
        )
        return out

    # ---- AMD 路径: 单 kernel 融合 ----
    out = torch.zeros(num_tokens, hidden_size, dtype=x.dtype, device=x.device)
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
