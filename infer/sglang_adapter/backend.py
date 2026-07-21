# SPDX-License-Identifier: Apache-2.0
"""SGLang attention backend — 将 infer.kernels RDNA3 kernel 注入 SGLang。

TODO（需拉 SGLang 源码后确认）:
  1. SGLang attention backend 的 base class 路径和注册方式
     - 预期位置: sglang.srt.layers.attention.triton_backend
     - SGLang >=0.4.x 有 --attention-backend 标志，可注册自定义实现
  2. SGLang 内部 API: attention forward 的调用签名（q/k/v shape, paged KV cache layout）
     - 预期与 vLLM 类似但 RadixAttention 有 extra fields
  3. 插件注册: 是 entry_points (setup.cfg/pyproject) 还是 monkey-patch import paths
     - 参考: SGLang 源码里的 FlashInfer backend 注册方式

SGLang 关键特性（需要适配的）:
  - RadixAttention: prefix-aware KV cache 共享（影响 block table 查询逻辑）
  - 连续 batching: 与 vLLM 的 block manager 不同
  - chunked prefill: 可能影响我们的 P0 2D kernel dispatch

暂时作为占位文件，集成代码在拉 SGLang 源码后补充。
"""

raise NotImplementedError(
    "SGLang backend is a stub — "
    "clone SGLang source and study its attention backend API first"
)
