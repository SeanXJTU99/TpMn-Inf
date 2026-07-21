# SPDX-License-Identifier: Apache-2.0
"""P3 fused GEGLU+FFN 正确性测试（vs torch 参考实现）。

运行: pytest infer/amdk/tests/test_fused_geglu.py -v
"""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("requires GPU (ROCm/CUDA)", allow_module_level=True)

from ..fused_geglu_ffn import fused_geglu_ffn_decode

# Qwen2.5-7B
HIDDEN = 3584
INTERMEDIATE = 18944


def ref_geglu_ffn(x, w_gate_up, w_down):
    """参考实现：gate_up_proj → SiLU(gate)*up → down_proj"""
    gate_up = x @ w_gate_up  # [1, 2*intermediate]
    gate = gate_up[:, :INTERMEDIATE]
    up = gate_up[:, INTERMEDIATE:]
    activated = F.silu(gate) * up
    return activated @ w_down  # [1, hidden]


TOL = {torch.bfloat16: (1e-1, 2e-1), torch.float16: (5e-2, 1e-1)}


def _make_weights(device, dtype):
    torch.manual_seed(1)
    x = torch.randn(1, HIDDEN, dtype=dtype, device=device) * 0.3
    # Xavier-ish for large intermediate
    w_gate_up = torch.randn(HIDDEN, 2 * INTERMEDIATE, dtype=dtype, device=device) * 0.015
    w_down = torch.randn(INTERMEDIATE, HIDDEN, dtype=dtype, device=device) * 0.015
    return x, w_gate_up, w_down


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_decode_matches_reference(dtype):
    device = "cuda"
    x, w_gate_up, w_down = _make_weights(device, dtype)

    got = fused_geglu_ffn_decode(x, w_gate_up, w_down)
    ref = ref_geglu_ffn(x.float(), w_gate_up.float(), w_down.float())

    atol, rtol = TOL[dtype]
    torch.testing.assert_close(
        got.float(), ref, atol=atol, rtol=rtol, msg="GEGLU+FFN mismatch"
    )
    print(f"[{dtype}] OK — max diff: {(got.float() - ref).abs().max():.4f}")


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_identity_like(dtype):
    """特殊情形：随机权重但小规模（hidden=16, intermediate=64），便于调试。"""
    device = "cuda"
    torch.manual_seed(2)
    hid, imd = 16, 64
    x = torch.randn(1, hid, dtype=dtype, device=device)
    w_gate_up = torch.randn(hid, 2 * imd, dtype=dtype, device=device) * 0.1
    w_down = torch.randn(imd, hid, dtype=dtype, device=device) * 0.1

    got = fused_geglu_ffn_decode(x, w_gate_up, w_down)
    # ref: manual
    gate_up = x.float() @ w_gate_up.float()
    gate, up = gate_up[:, :imd], gate_up[:, imd:]
    activated = F.silu(gate) * up
    ref = activated @ w_down.float()

    atol, rtol = TOL[dtype]
    torch.testing.assert_close(got.float(), ref, atol=atol, rtol=rtol)
