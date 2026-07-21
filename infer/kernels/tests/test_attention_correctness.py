# SPDX-License-Identifier: Apache-2.0
"""amdk RDNA3 unified attention 正确性测试（需 GPU + Triton，WSL2/ROCm 上跑）。

对照 fp32 朴素实现验证 2D（prefill/混合）与 3D+reduce（decode）两条路径。

  pytest infer/amdk/tests/test_attention_correctness.py -v
"""

import pytest
import torch

pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("requires GPU (ROCm/CUDA)", allow_module_level=True)

from ..tune_config import TUNE
from ..attention import unified_attention_rdna3

BLOCK_SIZE = 16
DEVICE = "cuda"

# (num_q_heads, num_kv_heads, head_size)
QWEN25_7B = (28, 4, 128)
SMALL_MHA = (8, 8, 64)


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def make_inputs(seq_params, heads, dtype, seed=0):
    """seq_params: list of (context_len, query_len)。返回 kernel 全部入参与参考用数据。"""
    torch.manual_seed(seed)
    num_q_heads, num_kv_heads, head_size = heads

    q_lens = [q for _, q in seq_params]
    seq_lens_list = [c + q for c, q in seq_params]
    total_q = sum(q_lens)

    q = torch.randn(total_q, num_q_heads, head_size, dtype=dtype, device=DEVICE)

    blocks_per_seq = [cdiv(s, BLOCK_SIZE) for s in seq_lens_list]
    num_blocks = sum(blocks_per_seq) + 8
    kv_cache = (
        torch.randn(
            num_blocks, 2, BLOCK_SIZE, num_kv_heads, head_size,
            dtype=dtype, device=DEVICE,
        )
        / 2
    )
    key_cache, value_cache = kv_cache.unbind(1)

    # 每个 seq 分配互不重叠的随机物理块
    perm = torch.randperm(num_blocks)
    max_blocks = max(blocks_per_seq)
    block_table = torch.zeros(
        len(seq_params), max_blocks, dtype=torch.int32, device=DEVICE
    )
    offset = 0
    for i, nb in enumerate(blocks_per_seq):
        block_table[i, :nb] = perm[offset : offset + nb].to(torch.int32)
        offset += nb

    cu_seqlens_q = torch.tensor(
        [0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32, device=DEVICE
    )
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=DEVICE)
    out = torch.empty_like(q)
    scale = head_size**-0.5
    return {
        "q": q,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_table": block_table,
        "cu_seqlens_q": cu_seqlens_q,
        "seq_lens": seq_lens,
        "out": out,
        "scale": scale,
        "seq_params": seq_params,
        "heads": heads,
    }


def ref_attention(inp):
    """fp32 朴素实现：逐 seq 从 paged cache 收集 K/V，带 context 偏移的 causal。"""
    num_q_heads, num_kv_heads, head_size = inp["heads"]
    num_queries_per_kv = num_q_heads // num_kv_heads
    outs = []
    q_start = 0
    for i, (ctx_len, q_len) in enumerate(inp["seq_params"]):
        seq_len = ctx_len + q_len
        nb = cdiv(seq_len, BLOCK_SIZE)
        block_ids = inp["block_table"][i, :nb].long()
        # [nb*BLOCK_SIZE, kv_heads, head] -> 截断到 seq_len
        k_seq = inp["key_cache"][block_ids].reshape(-1, num_kv_heads, head_size)
        v_seq = inp["value_cache"][block_ids].reshape(-1, num_kv_heads, head_size)
        k_seq = k_seq[:seq_len].float()
        v_seq = v_seq[:seq_len].float()
        # GQA 展开
        k_seq = k_seq.repeat_interleave(num_queries_per_kv, dim=1)
        v_seq = v_seq.repeat_interleave(num_queries_per_kv, dim=1)

        q_seq = inp["q"][q_start : q_start + q_len].float()  # [q_len, H, D]
        # [H, q_len, seq_len]
        scores = torch.einsum("qhd,khd->hqk", q_seq, k_seq) * inp["scale"]
        q_pos = torch.arange(q_len, device=DEVICE)[:, None] + ctx_len
        k_pos = torch.arange(seq_len, device=DEVICE)[None, :]
        causal = k_pos <= q_pos  # [q_len, seq_len]
        scores.masked_fill_(~causal[None], float("-inf"))
        p = scores.softmax(dim=-1)
        o = torch.einsum("hqk,khd->qhd", p, v_seq)
        outs.append(o)
        q_start += q_len
    return torch.cat(outs, dim=0)


def alloc_segm_buffers(num_seqs, num_q_heads, head_size):
    segments = TUNE.num_softmax_segments
    hp = 1 << (head_size - 1).bit_length()
    return dict(
        seq_threshold_3D=num_seqs,
        num_par_softmax_segments=segments,
        softmax_segm_output=torch.empty(
            num_seqs, num_q_heads, segments, hp, dtype=torch.float32, device=DEVICE
        ),
        softmax_segm_max=torch.empty(
            num_seqs, num_q_heads, segments, dtype=torch.float32, device=DEVICE
        ),
        softmax_segm_expsum=torch.empty(
            num_seqs, num_q_heads, segments, dtype=torch.float32, device=DEVICE
        ),
    )


def run_kernel(inp, use_3d: bool):
    q_lens = [q for _, q in inp["seq_params"]]
    extra = (
        alloc_segm_buffers(len(q_lens), inp["heads"][0], inp["heads"][2])
        if use_3d
        else {}
    )
    unified_attention_rdna3(
        q=inp["q"],
        k=inp["key_cache"],
        v=inp["value_cache"],
        out=inp["out"],
        cu_seqlens_q=inp["cu_seqlens_q"],
        max_seqlen_q=max(q_lens),
        seqused_k=inp["seq_lens"],
        max_seqlen_k=int(inp["seq_lens"].max()),
        softmax_scale=inp["scale"],
        causal=True,
        block_table=inp["block_table"],
        **extra,
    )
    return inp["out"]


TOLERANCE = {torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
             torch.float16: dict(atol=1e-2, rtol=1e-2)}


@pytest.mark.parametrize("heads", [QWEN25_7B, SMALL_MHA], ids=["qwen25-gqa", "mha"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
class TestUnifiedAttentionRdna3:
    def test_pure_decode_3d(self, heads, dtype):
        """纯 decode 小 batch → 3D + reduce 路径。"""
        inp = make_inputs([(511, 1), (2047, 1), (63, 1), (8191, 1)], heads, dtype)
        got = run_kernel(inp, use_3d=True)
        torch.testing.assert_close(
            got.float(), ref_attention(inp), **TOLERANCE[dtype]
        )

    def test_pure_decode_2d(self, heads, dtype):
        """同负载走 2D 路径，两条路径都必须对。"""
        inp = make_inputs([(511, 1), (2047, 1), (63, 1), (8191, 1)], heads, dtype)
        got = run_kernel(inp, use_3d=False)
        torch.testing.assert_close(
            got.float(), ref_attention(inp), **TOLERANCE[dtype]
        )

    def test_mixed_prefill_decode(self, heads, dtype):
        """混合批（prefill + decode + chunked 续算）→ 2D。"""
        inp = make_inputs([(0, 256), (100, 17), (2048, 1), (37, 3)], heads, dtype)
        got = run_kernel(inp, use_3d=False)
        torch.testing.assert_close(
            got.float(), ref_attention(inp), **TOLERANCE[dtype]
        )

    def test_long_context_decode(self, heads, dtype):
        """16k 深上下文 decode（游戏长会话形态）。"""
        inp = make_inputs([(16383, 1)], heads, dtype)
        got = run_kernel(inp, use_3d=True)
        torch.testing.assert_close(
            got.float(), ref_attention(inp), **TOLERANCE[dtype]
        )

    def test_single_token_seq(self, heads, dtype):
        """边界：ctx=0 的首 token。"""
        inp = make_inputs([(0, 1)], heads, dtype)
        got = run_kernel(inp, use_3d=False)
        torch.testing.assert_close(
            got.float(), ref_attention(inp), **TOLERANCE[dtype]
        )
