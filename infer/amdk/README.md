# amdk — RDNA3 特化 Triton kernel（Phase 2）

vLLM 源码零修改：经 `vllm.general_plugins` 入口把 `Rdna3AttentionBackend`
注册为 `AttentionBackendEnum.CUSTOM`。

## 安装与启用（WSL2，与 vLLM 同一 venv）

```bash
pip install -e infer/                 # 注册插件入口
vllm serve ... --attention-backend CUSTOM
```

回退：去掉 `--attention-backend CUSTOM` 即回到上游自动选择（TRITON_ATTN）。

## 文件

| 文件 | 内容 |
|---|---|
| `unified_attention_rdna3.py` | P0: 上游 unified attention 的 gfx1100 特化 fork（2D + 3D/reduce） |
| `fused_rms_qkv_rope.py` | P1: Fused RMSNorm + QKV Projection + RoPE（decode 单 kernel） |
| `backend.py` | CUSTOM backend / Impl / MetadataBuilder |
| `tune_config.py` | 全部调参参数（tile / warps / stages / waves_per_eu / 段数） |
| `plugin.py` | vLLM 插件注册入口 |
| `tests/` | P0 5 组 + P1 2 组正确性测试（vs torch 参考实现） |

## 相对上游的特化 / 新增

- P0 fork: 删除 alibi/qq_bias/softcap/sinks/mm_prefix/sliding window/FP8（编译更快）
- P1: 上游无此融合——RMSNorm + QKV 线性 + RoPE 三步合为单 kernel，消除 3 次 launch
- 已知限制: P1 仅 decode 路径（prefill 走上游拆分 kernel）

## 调参流程

1. 默认值（未实测起点）见 `tune_config.py` 注释
2. 实测：`tune_attention.py`（待写）扫 tile×warps×stages×waves_per_eu×segments
3. 最优组合写入 `amd_tune.json`（或 `AMDK_TUNE_CONFIG=path` 指定），重启生效
4. 每次调参后跑 `infer/bench/baseline_bench.py --tag <名字>` 留档对比

## 验证

```bash
pytest infer/amdk/tests/ -v                    # 正确性（需 GPU）
python3 infer/bench/baseline_bench.py --tag amdk_pa   # 端到端对比基线
```
