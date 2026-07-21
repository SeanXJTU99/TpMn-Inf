# TpMn-Inf — Non-NVIDIA LLM Inference Optimization

English | [中文](README_zh.md)

Local inference for real-time narrative generation games. Replaces cloud APIs with a single AMD GPU (or Ascend NPU) running open-source LLMs + LoRA adapters — zero inference cost, no quality regression.

## Architecture

```
game_server (FastAPI :8000)
    → AsyncOpenAI client
    → vLLM / SGLang (:8080)
    → Triton kernels (hardware-agnostic DSL → native GPU ISA)
        ├── AMD ROCm  (RDNA3/CDNA3)
        └── Ascend CANN (910B, UB 192KB, Cube 16-align)
```

**Principles**:
- **Zero framework modification** — kernels injected via framework plugin mechanisms
- **Write once, run anywhere** — same Triton ops across AMD/Ascend, platform-specific paths via `tl.constexpr` branches + tuning JSON
- **Model agnostic** — supports Qwen 2.5/3, GLM, Llama, and other decoder-only architectures (7B–14B)

## Project Structure

```
├── infer/                    # Inference engine
│   ├── kernels/              #   Framework-agnostic Triton ops
│   │   ├── attention.py       #     P0: PagedAttention (2D/3D)
│   │   ├── fused_qkv_rope.py  #     P1: RMSNorm+QKV+RoPE
│   │   ├── fused_geglu_ffn.py #     P3: GEGLU+FFN
│   │   ├── tune_config.py     #     Multi-platform tuning dispatch
│   │   └── ascend_tune.json   #     Ascend 910B tile defaults
│   ├── vllm_adapter/         #   vLLM CUSTOM backend plugin
│   ├── sglang_adapter/       #   SGLang amdk backend
│   ├── bench/                #   Benchmarks
│   └── scripts/              #   Launch scripts
├── training/eval/            # Evaluation framework
│   ├── checks/               #   Programmatic hard checks (16 checks)
│   ├── judges/               #   LLM judge (pairwise + rubric)
│   ├── human/                #   Blind A/B tool
│   └── runners/              #   Replay engine + CLI
└── game_server/              # Game server (public API contract)
    ├── models.py / config.py #   Data models
    ├── tests/                #   104 tests
    └── static/               #   Web UI
```

## Quick Start

### On Windows (no GPU needed)

```bash
# Eval hard checks
python -c "from training.eval.checks.persona import run_all_persona; ..."
```

### With AMD GPU (Linux/WSL2)

See `TESTING.md` for step-by-step (8 sections, from ROCm setup to autotuning).

## Status

| Module | Status |
|--------|:---:|
| P0 PagedAttention (Triton, AMD + Ascend) | Done, awaiting GPU test |
| P1 Fused QKV+RoPE (AMD + Ascend) | Done, awaiting GPU test |
| P3 Fused GEGLU+FFN (AMD + Ascend) | Done, awaiting GPU test |
| Eval hard checks (16 checks) | Runnable on Windows |
| Autotuning (tune_attention.py) | Done, awaiting GPU sweep |
| vLLM adapter (CUSTOM backend + inject) | Done |
| SGLang adapter (amdk backend + inject) | Done |
| Ascend 910B adapter (kernel IS_ASCEND paths) | Done, awaiting 910B hardware |
| Baseline | Awaiting GPU hardware |

## Hardware & Models

- **Target**: AMD (RDNA3/CDNA3) / Ascend 910B+ / NVIDIA (compatibility fallback)
- **Models**: Decoder-only architectures, 7B–14B parameters, LoRA post-training
- **Dev**: Linux + ROCm 7.x / CANN, Triton → AMDGPU / Ascend backend
