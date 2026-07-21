# TpMn-Inf — Non-NVIDIA LLM Inference Optimization

Local inference for real-time narrative generation games. Replaces cloud APIs with a single AMD GPU (or Ascend NPU) running open-source LLMs + LoRA adapters — zero inference cost, no quality regression.

## Architecture

```
game_server (FastAPI :8000)
    → AsyncOpenAI client
    → vLLM / SGLang (:8080)
    → Triton kernels (hardware-agnostic DSL → native GPU ISA)
        ├── AMD ROCm  (RDNA3/CDNA3)
        └── Ascend CANN (planned)
```

**Principles**:
- **Zero framework modification** — kernels injected via framework plugin mechanisms
- **Write once, run anywhere** — same Triton ops across AMD/Ascend/NVIDIA, only swap tuning JSON
- **Model agnostic** — supports Qwen 2.5/3, GLM, Llama, and other decoder-only architectures (7B–14B)

## Project Structure

```
├── infer/                    # Inference engine
│   ├── kernels/              #   Framework-agnostic Triton ops
│   ├── vllm_adapter/         #   vLLM CUSTOM backend plugin
│   ├── sglang_adapter/       #   SGLang adapter (stub)
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
| P0 PagedAttention (Triton fork) | Done, awaiting GPU test |
| P1 Fused QKV+RoPE | Done, awaiting GPU test |
| P3 Fused GEGLU+FFN | Done, awaiting GPU test |
| Eval hard checks (16 checks) | Runnable on Windows |
| Autotuning (tune_attention.py) | Done, awaiting GPU sweep |
| Baseline | Awaiting WSL2 env |
| SGLang adapter | Stub complete |

## Hardware & Models

- **Target**: AMD (RDNA3/CDNA3) / Ascend 910B+ / NVIDIA (compatibility fallback)
- **Models**: Decoder-only architectures, 7B–14B parameters, LoRA post-training
- **Dev**: Linux + ROCm 7.x / CANN, Triton → AMDGPU / Ascend backend
