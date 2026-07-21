# SGLang adapter

将 `kernels` RDNA3 Triton kernel 注入 SGLang，通过 `--attention-backend amdk` 启用。

## 架构

```
sglang serve ... --attention-backend amdk
    → @register_attention_backend("amdk")
    → AmdkAttnBackend (继承 TritonAttnBackend)
    → kernels.tune_config (RDNA3 tile/warp/stage 参数)
```

## 与 vLLM adapter 对比

| | vLLM | SGLang |
|---|---|---|
| 注册方式 | `register_backend(Enum.CUSTOM)` | `@register_attention_backend("amdk")` |
| 入口 | `vllm.general_plugins` entry_point | `plugin.register()` 手动调用 |
| KV 组织 | paged block table (BLOCK_SIZE=16) | flat kv_indices (PAGE_SIZE 可配) |
| P1 融合 | 自研 `fused_qkv_rope.py` | 上游已有 `fused_qk_norm_rope_store.py` |
| P3 融合 | 自研 `fused_geglu_ffn.py` | **仍需要**（上游无） |

## 已实现

- `backend.py`：`create_amdk_backend()` 工厂函数，实例化 `TritonAttnBackend` + 注入 tune_config
- `plugin.py`：`register()` 将 amdk 注入 `ATTENTION_BACKENDS` 字典

## 待完成

- [ ] `launch_sglang.sh` 启动脚本
- [ ] baseline_bench.py 支持 SGLang endpoint
- [ ] P0 深度适配：SGLang flat kv_indices 版 attention fork（当前复用上游 kernel，仅调参）
- [ ] P3 模型层注入：在 SGLang Qwen2 MLP 中替换 GEGLU+FFN 为 `kernels.fused_geglu_ffn`
- [ ] 端到端测试：pytest + benchmark 对照
