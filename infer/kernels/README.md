# kernels — 框架无关 Triton 算子

仅依赖 `torch` + `triton`。由 `vllm_adapter` 和 `sglang_adapter` 调用。

| 文件 | 内容 |
|---|---|
| `attention.py` | P0: unified paged attention (2D prefill + 3D decode) |
| `fused_qkv_rope.py` | P1: Fused RMSNorm + QKV projection + RoPE (decode) |
| `fused_geglu_ffn.py` | P3: Fused GEGLU + FFN (decode) |
| `tune_config.py` | 硬件调参 + 平台检测 |
| `tune_attention.py` | 自动参数网格搜索 |

## 硬件调参

```python
from kernels import detect_platform  # → "amd" | "ascend" | "cuda"
from kernels.tune_config import TUNE  # 自动加载对应 JSON
```

Triton `tl.constexpr` 等价于 CUDA 模板特化——每次改 tile size 只是 JIT 缓存 key 不同，
不会重复编译已涵盖的组合。
