# SPDX-License-Identifier: Apache-2.0
"""P1 fused RMSNorm+QKV+RoPE 正确性测试（vs torch 参考实现）。

运行: pytest infer/amdk/tests/test_fused_qkv_rope.py -v
"""

import math

import pytest
import torch

pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("requires GPU (ROCm/CUDA)", allow_module_level=True)

from amdk.fused_rms_qkv_rope import fused_rms_qkv_rope_decode


# Qwen2.5-7B
HIDDEN = 3584
NUM_Q_HEADS = 28
NUM_KV_HEADS = 4
HEAD_SIZE = 128
Q_SIZE = NUM_Q_HEADS * HEAD_SIZE  # 3584
KV_SIZE = NUM_KV_HEADS * HEAD_SIZE  # 512
TOTAL_QKV = Q_SIZE + 2 * KV_SIZE  # 4608
EPS = 1e-6
DEFAULT_POS = 42


def ref_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rms * weight


def ref_rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """neox-style RoPE: 相邻 dim pair 旋转。x: [..., head_size]"""
    half = x.shape[-1] // 2
    x_even = x[..., 0::2]  # [..., half]
    x_odd = x[..., 1::2]
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_odd * cos + x_even * sin
    out = torch.empty_like(x)
    out[..., 0::2] = rotated_even
    out[..., 1::2] = rotated_odd
    return out


def make_fixtures(device, dtype, pos=DEFAULT_POS):
    """构造解码场景下的随机张量（含合理值范围的权重）。"""
    torch.manual_seed(0)
    x = torch.randn(1, HIDDEN, dtype=dtype, device=device) * 0.5
    rms_w = torch.ones(HIDDEN, dtype=dtype, device=device)
    # QKV 权重用 Xavier-ish 幅度
    qkv_w = torch.randn(HIDDEN, TOTAL_QKV, dtype=dtype, device=device) * 0.03
    q_b = torch.randn(Q_SIZE, dtype=dtype, device=device) * 0.02
    k_b = torch.randn(KV_SIZE, dtype=dtype, device=device) * 0.02
    v_b = torch.randn(KV_SIZE, dtype=dtype, device=device) * 0.02

    max_seq_len = 256
    inv_freq = 1.0 / (1e6 ** (torch.arange(0, HEAD_SIZE, 2, device=device) / HEAD_SIZE))
    freqs = pos * inv_freq  # [half]
    cos = freqs.cos().to(dtype)  # [half]
    sin = freqs.sin().to(dtype)
    cos_pad = torch.zeros(max_seq_len, HEAD_SIZE // 2, dtype=dtype, device=device)
    sin_pad = torch.zeros(max_seq_len, HEAD_SIZE // 2, dtype=dtype, device=device)
    cos_pad[pos] = cos
    sin_pad[pos] = sin

    positions = torch.tensor([pos], dtype=torch.int32, device=device)
    return dict(
        x=x, rms_w=rms_w, qkv_w=qkv_w, q_b=q_b, k_b=k_b, v_b=v_b,
        cos_cache=cos_pad, sin_cache=sin_pad, positions=positions,
    )


def ref_fused(fx):
    """参考实现：RMSNorm → Linear → split → reshape → RoPE"""
    x = fx["x"]
    dtype = x.dtype
    x_norm = ref_rms_norm(x, fx["rms_w"], EPS)
    qkv = x_norm @ fx["qkv_w"]
    qkv = qkv + torch.cat([fx["q_b"], fx["k_b"], fx["v_b"]], dim=-1)
    q, k, v = qkv.split([Q_SIZE, KV_SIZE, KV_SIZE], dim=-1)
    # RoPE
    pos = int(fx["positions"][0])
    cos = fx["cos_cache"][pos].to(x.device)
    sin = fx["sin_cache"][pos].to(x.device)
    q_r = ref_rope_neox(q.view(1, NUM_Q_HEADS, HEAD_SIZE), cos, sin)
    k_r = ref_rope_neox(k.view(1, NUM_KV_HEADS, HEAD_SIZE), cos, sin)
    return q_r.reshape(1, Q_SIZE), k_r.reshape(1, KV_SIZE), v


TOL = {torch.bfloat16: (1e-1, 2e-1), torch.float16: (5e-2, 1e-1)}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_decode_matches_reference(dtype):
    device = "cuda"
    fx = make_fixtures(device, dtype)
    ref_q, ref_k, ref_v = ref_fused(fx)

    got_q, got_k, got_v = fused_rms_qkv_rope_decode(
        x=fx["x"],
        rms_weight=fx["rms_w"],
        qkv_weight=fx["qkv_w"],
        q_bias=fx["q_b"],
        k_bias=fx["k_b"],
        v_bias=fx["v_b"],
        cos_cache=fx["cos_cache"],
        sin_cache=fx["sin_cache"],
        positions=fx["positions"],
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
    )

    atol, rtol = TOL[dtype]
    torch.testing.assert_close(got_q.float(), ref_q.float(), atol=atol, rtol=rtol,
                               msg="Q mismatch")
    torch.testing.assert_close(got_k.float(), ref_k.float(), atol=atol, rtol=rtol,
                               msg="K mismatch")
    torch.testing.assert_close(got_v.float(), ref_v.float(), atol=atol, rtol=rtol,
                               msg="V mismatch")


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_no_bias(dtype):
    """Qwen2.5 无 bias（vLLM 里通常 skip_bias_add=True）。"""
    device = "cuda"
    fx = make_fixtures(device, dtype)
    got_q, got_k, got_v = fused_rms_qkv_rope_decode(
        x=fx["x"],
        rms_weight=fx["rms_w"],
        qkv_weight=fx["qkv_w"],
        q_bias=None,
        k_bias=None,
        v_bias=None,
        cos_cache=fx["cos_cache"],
        sin_cache=fx["sin_cache"],
        positions=fx["positions"],
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
    )

    # Reference: no bias
    x_norm = ref_rms_norm(fx["x"], fx["rms_w"], EPS)
    qkv = x_norm @ fx["qkv_w"]
    q, k, v = qkv.split([Q_SIZE, KV_SIZE, KV_SIZE], dim=-1)
    pos = int(fx["positions"][0])
    cos = fx["cos_cache"][pos].to(device)
    sin = fx["sin_cache"][pos].to(device)
    ref_q = ref_rope_neox(q.float().view(1, NUM_Q_HEADS, HEAD_SIZE), cos, sin).reshape(1, Q_SIZE)
    ref_k = ref_rope_neox(k.float().view(1, NUM_KV_HEADS, HEAD_SIZE), cos, sin).reshape(1, KV_SIZE)
    ref_v = v.float()

    atol, rtol = TOL[dtype]
    torch.testing.assert_close(got_q.float(), ref_q, atol=atol, rtol=rtol, msg="Q (no bias)")
    torch.testing.assert_close(got_k.float(), ref_k, atol=atol, rtol=rtol, msg="K (no bias)")
    torch.testing.assert_close(got_v.float(), ref_v, atol=atol, rtol=rtol, msg="V (no bias)")
