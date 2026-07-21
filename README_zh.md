# TpMn-Inf — 非 NVIDIA 生态 LLM 推理优化

[English](README.md) | 中文

为实时剧情生成游戏构建的非 NVIDIA 生态本地推理方案。用单卡 AMD GPU（或 Ascend NPU）运行开源 LLM + LoRA adapter，替换云端 API，目标零推理成本 + 叙事质量不劣于云端。

支持模型：Qwen2.5/3、GLM、Llama 等 decoder-only 架构（7B-14B 级），基座冻结后通过 LoRA 后训练定制化。

## 架构

```
game_server (FastAPI :8000)
    → AsyncOpenAI client
    → vLLM / SGLang OpenAI-compatible endpoint (:8080)
    → Triton kernels (硬件无关 DSL → 目标硬件 ISA)
        ├── AMD ROCm backend  (RDNA3 wave32, LDS 64KB/CU)
        └── Ascend CANN backend (910B, UB 192KB, Cube 16-align)
```

**设计原则**：
- **vLLM/SGLang 框架零修改** — 算子通过框架插件机制注入
- **Triton 一次编写** — 同组 kernel 跨 AMD/Ascend，平台差异通过 `tl.constexpr` 分支 + 调参 JSON 覆盖
- **模型无关** — 支持 Qwen、GLM、Llama 等 decoder-only 模型，适配层最小化

## 项目结构

```
├── infer/                    # 推理引擎（核心交付）
│   ├── kernels/               #   框架无关 Triton 算子
│   │   ├── attention.py       #     P0: PagedAttention (2D/3D)
│   │   ├── fused_qkv_rope.py  #     P1: RMSNorm+QKV+RoPE
│   │   ├── fused_geglu_ffn.py #     P3: GEGLU+FFN
│   │   ├── tune_config.py     #     多平台调参分发
│   │   └── ascend_tune.json   #     Ascend 910B tile 默认值
│   ├── vllm_adapter/          #   vLLM CUSTOM backend 插件
│   ├── sglang_adapter/        #   SGLang amdk backend
│   ├── bench/                 #   Benchmark 工具
│   └── scripts/               #   启动脚本
├── training/eval/            # 评估体系（Phase 3.5）
│   ├── checks/               #   程序化硬校验（persona/schema/leak/slop）
│   ├── judges/               #   LLM judge（pairwise + rubric）
│   ├── human/                #   盲测工具
│   └── runners/              #   回放器 + 主入口
└── game_server/              # 游戏服务（公开 API 契约 + 测试）
    ├── models.py / config.py #   数据模型
    ├── tests/                #   104 个测试
    └── static/               #   Web UI
```

## 快速开始

### 当前可在 Windows 本机跑

```bash
# eval 硬校验（零 GPU 依赖）
python -c "from training.eval.checks.persona import run_all_persona; ..."
```

### 需要 AMD GPU + WSL2/Linux

详见 `TESTING.md`（8 节完整测试步骤）。

## 当前状态

| 模块 | 状态 |
|------|:---:|
| P0 PagedAttention (AMD + Ascend) | 代码完成，待 GPU 实测 |
| P1 Fused QKV+RoPE (AMD + Ascend) | 代码完成，待 GPU 实测 |
| P3 Fused GEGLU+FFN (AMD + Ascend) | 代码完成，待 GPU 实测 |
| eval 硬校验 (16 checks) | Windows 本机可跑 |
| 自动调参 (tune_attention.py) | 代码完成，待 GPU sweep |
| vLLM adapter (CUSTOM backend + inject) | 代码完成 |
| SGLang adapter (amdk backend + inject) | 代码完成 |
| Ascend 910B 适配 (kernel IS_ASCEND 分支) | 代码完成，待 910B 硬件 |
| Phase 1 基线 | 待 GPU 硬件 |

## 硬件

- 目标硬件：AMD (RDNA3/CDNA3) / Ascend 910B+ / NVIDIA (回退兼容)
- 目标模型：decoder-only 架构，7B-14B 参数量，LoRA 后训练
- 开发环境：Linux + ROCm 7.x / CANN，Triton → AMDGPU / Ascend backend
