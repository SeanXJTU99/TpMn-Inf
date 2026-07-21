# TpMn-Inf — Non-NVIDIA LLM Inference Optimization

为实时剧情生成游戏构建的非 NVIDIA 生态本地推理方案。用单卡 AMD GPU（或 Ascend NPU）+ LoRA Qwen2.5-7B 替换云端 API，目标零推理成本 + 叙事质量不劣于云端。

## 架构

```
game_server (FastAPI :8000)
    → AsyncOpenAI client
    → vLLM / SGLang OpenAI-compatible endpoint (:8080)
    → Triton kernels (硬件无关 DSL → 目标硬件 ISA)
        ├── AMD ROCm backend  (gfx1100 wave32)
        └── Ascend CANN backend (规划中)
```

**设计原则**：
- **vLLM/SGLang 框架零修改** — 算子通过框架插件机制注入
- **Triton 一次编写** — 同组 kernel 跨 AMD/Ascend，仅换调参 JSON
- **代码卡关，AI 润色** — 游戏硬规则在 Python 层，LLM 只管叙事

## 项目结构

```
├── infer/                    # 推理引擎（核心交付）
│   ├── amdk/                 #   RDNA3 特化 Triton kernel + vLLM 插件
│   │   ├── unified_attention_rdna3.py   # P0: PagedAttention
│   │   ├── fused_rms_qkv_rope.py        # P1: RMSNorm+QKV+RoPE 融合
│   │   ├── fused_geglu_ffn.py           # P3: GEGLU+FFN 融合
│   │   ├── tune_config.py / tune_attention.py   # 自动调参
│   │   └── tests/                       # 正确性测试
│   ├── bench/                #   Benchmark 工具
│   └── scripts/              #   启动脚本
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
| P0 PagedAttention (Triton fork) | 代码完成，待 GPU 实测 |
| P1 Fused QKV+RoPE | 代码完成，待 GPU 实测 |
| P3 Fused GEGLU+FFN | 代码完成，待 GPU 实测 |
| eval 硬校验 (16 checks) | Windows 本机可跑 |
| 自动调参 (tune_attention.py) | 代码完成，待 GPU sweep |
| Phase 1 基线 | 待 WSL2 环境 |
| SGLang adapter | 规划中（见 refactor_plan） |
| Ascend 适配 | 待确认 CANN 环境 |

## 硬件

- 目标：AMD RX 7900 XTX (gfx1100, RDNA3, 24GB) 或 Ascend 910B
- 开发：WSL2 + ROCm 7.2 / 云端 Linux
- Triton → AMDGPU backend / CANN fork
