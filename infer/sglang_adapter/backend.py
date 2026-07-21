# SPDX-License-Identifier: Apache-2.0
"""SGLang AmdkAttnBackend — 通过 @register_attention_backend("amdk") 注册，
--attention-backend amdk 启用。继承上游 TritonAttnBackend 全部功能，
注入 kernels.tune_config 调参。

上游参考: sglang/python/sglang/srt/layers/attention/triton_backend.py
注册参考: sglang/python/sglang/srt/layers/attention/attention_registry.py

与 vLLM adapter 的差异:
  - SGLang 用 flat kv_indices（PAGE_SIZE 可配），非 block table
  - 上游已有 fused_qk_norm_rope_store.py（P1 部分重叠）
  - P3 (GEGLU+FFN) 上游仍无融合，需模型层注入
  - 注册只需装饰器 + 工厂函数，无 enum/override 体系
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from kernels.tune_config import TUNE

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def create_amdk_backend(
    model_runner: "ModelRunner",
    skip_prefill: bool = False,
    kv_indptr_buf: Optional[torch.Tensor] = None,
):
    """注册为 @register_attention_backend("amdk") 的工厂函数。

    实例化上游 TritonAttnBackend 并注入 RDNA3 调参。
    """
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

    instance = TritonAttnBackend(
        model_runner,
        skip_prefill=skip_prefill,
        kv_indptr_buf=kv_indptr_buf,
    )

    # 按 CU 数调大 KV splits（RDNA3 96 CU，需更多 split 提升 occupancy）
    default_splits = getattr(instance, "max_kv_splits", 8)
    instance.max_kv_splits = max(default_splits, 16)
    instance._amdk_enabled = True
    instance._amdk_tune = TUNE

    logger.info(
        "amdk: TritonAttnBackend ready | "
        "max_kv_splits=%d | decode_tile=%d warps=%d | prefill_tile=%d warps=%d",
        instance.max_kv_splits,
        TUNE.decode.tile_size,
        TUNE.decode.num_warps,
        TUNE.prefill.tile_size,
        TUNE.prefill.num_warps,
    )

    return instance
