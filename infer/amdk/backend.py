# SPDX-License-Identifier: Apache-2.0
"""RDNA3 attention backend — 挂到 vLLM AttentionBackendEnum.CUSTOM。

复用上游 TritonAttentionBackend 的全部框架逻辑（metadata builder /
KV cache 布局 / reshape_and_cache），只把 attention 计算换成
amdk 的 RDNA3 特化 kernel。

限制（Qwen2.5 特化，越界即报错回退上游 backend）：
- 仅 DECODER attention
- 无 sliding window / alibi / sinks / softcap / FP8
"""

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from amdk.tune_config import TUNE
from amdk.unified_attention_rdna3 import unified_attention_rdna3


class Rdna3AttentionMetadataBuilder(TritonAttentionMetadataBuilder):
    """仅当调参配置改变 softmax 段数时重分配 decode 分段缓冲。"""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        segments = TUNE.num_softmax_segments
        if segments != self.num_par_softmax_segments:
            self.num_par_softmax_segments = segments
            headdim_padded = self.softmax_segm_output.shape[-1]
            self.softmax_segm_output = torch.empty(
                (
                    self.seq_threshold_3D,
                    self.num_heads_q,
                    segments,
                    headdim_padded,
                ),
                dtype=torch.float32,
                device=device,
            )
            self.softmax_segm_max = torch.empty(
                (self.seq_threshold_3D, self.num_heads_q, segments),
                dtype=torch.float32,
                device=device,
            )
            self.softmax_segm_expsum = torch.empty(
                (self.seq_threshold_3D, self.num_heads_q, segments),
                dtype=torch.float32,
                device=device,
            )


class Rdna3AttentionImpl(TritonAttentionImpl):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 特化范围守卫：碰到不支持的特性宁可显式失败，
        # 用户可退回 --attention-backend TRITON_ATTN
        assert self.attn_type == AttentionType.DECODER, (
            "amdk: only decoder attention is supported"
        )
        assert self.sliding_window == (-1, 0), "amdk: sliding window not supported"
        assert self.alibi_slopes is None, "amdk: alibi not supported"
        assert self.logits_soft_cap == 0, "amdk: softcap not supported"
        assert self.sinks is None, "amdk: sinks not supported"
        assert not self.kv_cache_dtype.startswith("fp8"), (
            "amdk: fp8 KV cache not supported"
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """结构对齐上游 TritonAttentionImpl.forward，计算换成 RDNA3 kernel。

        NOTE: piece-wise CUDA graph 下本方法以 eager 执行，
        保持 PyTorch op 最少（见上游注释）。
        """
        assert output is not None, "Output tensor must be provided."
        assert output_scale is None and output_block_scale is None, (
            "amdk: fused output quant not supported"
        )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        num_actual_tokens = attn_metadata.num_actual_tokens

        key_cache, value_cache = kv_cache.unbind(1)

        unified_attention_rdna3(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=True,
            block_table=attn_metadata.block_table,
            seq_threshold_3D=attn_metadata.seq_threshold_3D,
            num_par_softmax_segments=attn_metadata.num_par_softmax_segments,
            softmax_segm_output=attn_metadata.softmax_segm_output,
            softmax_segm_max=attn_metadata.softmax_segm_max,
            softmax_segm_expsum=attn_metadata.softmax_segm_expsum,
        )

        return output


class Rdna3AttentionBackend(TritonAttentionBackend):
    supported_dtypes = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes = ["auto", "float16", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        # 必须与注册的 enum 成员名一致（vLLM 会反查 AttentionBackendEnum[name]）
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["Rdna3AttentionImpl"]:
        return Rdna3AttentionImpl

    @staticmethod
    def get_builder_cls() -> type["Rdna3AttentionMetadataBuilder"]:
        return Rdna3AttentionMetadataBuilder

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False
