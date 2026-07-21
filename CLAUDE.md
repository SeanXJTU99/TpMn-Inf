# CLAUDE.md — gameAMDenging

AMD 架构 vLLM 推理引擎深度优化项目。

## 目标

为型月圣杯战争 AI 叙事游戏构建本地 LLM 推理替代方案，用单卡 AMD GPU + LoRA Qwen2.5-7B 替换 DeepSeek V4 API，降低推理成本至零，同时达到匹配甚至超越云端 API 的叙事质量。

## 目录结构

```
E:\gameAMDenging\
├── CLAUDE.md               # 本文件
├── lucky-baking-coral.md   # 详细技术计划（权威）
├── PROGRESS.md             # 实施进度记录（每完成一项更新）
├── eval_plan.md            # 评估体系设计（三轴：长上下文/指令遵循/创意写作）
├── TESTING.md              # 测试操作步骤（eval 硬校验 + kernel + benchmark + 调参）
├── .gitignore
├── vllm/                   # vLLM v0.18.0 sparse clone（只读参考，git 忽略）
├── game_server/            # 原 typemoon 游戏服务代码（独立子项目）
│   ├── CLAUDE.md           #   游戏服务的 CLAUDE.md
│   ├── game_server.py      #   FastAPI 主引擎
│   ├── config.py / models.py / atomic_rules.py / ai_client.py / system_prompts.py
│   ├── servant_db.json     #   35 骑英灵库
│   ├── client.py           #   终端 CLI
│   ├── static/             #   手机端 Web UI
│   └── tests/              #   104 个测试
├── infer/                  # 推理引擎
│   ├── PHASE1.md           #   Phase 1 环境搭建 + 基线步骤
│   ├── pyproject.toml      #   amdk 包（pip install -e infer/ 注册 vLLM 插件）
│   ├── amdk/               #   Triton 核函数（RDNA3 特化，CUSTOM backend）
│   ├── bench/              #   基线/回归 benchmark
│   ├── scripts/            #   启动脚本
│   ├── tvm_graph/          #   TVM 子图编译（待开发）
│   └── server.py           #   OpenAI-compatible API（待开发）
└── training/               # 后训练管线
    ├── data/               #   SFT/DPO 数据集（待开发）
    ├── scripts/            #   Gemini 评判 + 数据构造（待开发）
    ├── adapters/           #   LoRA weights 产出（待开发）
    └── eval/               #   评估体系（见 eval_plan.md）：checks/runners/judges/human/report —— 代码完成
```

## 核心决策

1. **vLLM 框架不动**：只重写算子层，scheduler/block manager/API server 保留
2. **Triton + TVM 互补**：Triton 做 kernel 级融合（PagedAttention / Fused QKV+RoPE / Fused GEGLU），TVM 做图级子图编译 + MetaSchedule auto-tune
3. **不做 TVM GEMM auto-tune**：已确认删除
4. **LoRA 全流程**：基座 Qwen2.5-7B 冻结，两份 adapter（Arbiter rank=64 / Narrator rank=32），三阶段后训练（SFT→RLAIF→DPO）
5. **Gemini 2.5 Pro 作 judge + 正例生成**：免费 tier，四维评分（文风/一致性/战术暗示/第二人称）
6. **AMD 专属调参**：RDNA3 wave32 / CDNA wave64 分别调 tile size 和 occupancy

## 实施顺序（6 Phase）

详细内容见 `lucky-baking-coral.md`，摘要：

| Phase | 内容 | 周期 |
|-------|------|:---:|
| 1 | vLLM AMD 基线跑通 | 1 周 |
| 2 | Triton 算子重构（PagedAttention → Fused QKV → GEGLU → LoRA） | 4 周 |
| 3 | TVM 图级编译 + auto-tune | 1 周 |
| 3.5 | 评估基建（回放器 + 硬校验 + pairwise judge，见 `eval_plan.md`） | 2-3 天 |
| 4 | LoRA 后训练（SFT → Gemini RLAIF → DPO），每轮 DPO 过评估门禁 | 3 周 |
| 5 | 压测 + 全量评估 + vault 开封 + 人工盲测 | 1 周 |
| 6 | 接入 game_server.py（vLLM endpoint） | 3 天 |

## 确认硬件（2026-07）

RX 7900 XTX：gfx1100 / RDNA3 wave32 / 96 CU / 24GB VRAM / 960GB/s / FP32 60TFLOPS。
ROCm 7.2 已支持 RDNA3 + vLLM + FA2 + 4bit 量化；**不支持 FP8 / FA3 / TensorRT-LLM** → 全程 FP16/BF16 kernel。
24GB 显存：Qwen2.5-7B FP16 (~15GB) 可整卡放下，无需强制 4bit；AWQ 4bit 作为扩 KV cache 余量的可选项。

**开发环境**：WSL2 + 云端 Linux。所有代码、脚本、文档一律按 Linux 环境编写（Windows 仅作宿主）。
**数据现状**：v3.3 对局日志暂无；评估/训练数据集由 Gemini 窗口提供（见 `eval_plan.md` §1 偏差警示）。

## AMD 架构参数

| 参数 | RDNA3 (RX 7900) | CDNA2 (MI250X) | CDNA3 (MI300X) |
|------|:---:|:---:|:---:|
| wave size | 32 | 64 | 64 |
| LDS/CU | 64KB | 64KB | 64KB |
| FA tile Br×Bc | 128×64 | 128×128 | 256×128 |

陷阱：AMD wave ≠ NVIDIA warp，HIPify 直接翻译会导致 divergence；LDS bank conflict 需手动 padding。

## 与 typemoon 的关系

- `D:\Pycharm\pyProj\typemoon`：v3.3 稳定版，DeepSeek API，继续可玩
- `E:\gameAMDenging\game_server\`：上述代码的副本，后续适配 vLLM endpoint
- 开发完成后切换 `ai_client.py` 的 endpoint 即可迁移
