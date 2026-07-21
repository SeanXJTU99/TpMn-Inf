#!/usr/bin/env bash
# Phase 1 基线：vLLM serve Qwen2.5-7B-Instruct（RX 7900 XTX / gfx1100 / WSL2）
# 用法: bash launch_baseline.sh [额外 vllm serve 参数...]
set -euo pipefail

# ---- RDNA3 / WSL2 必需环境变量 ----
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100
export GPU_MAX_HW_QUEUES=1              # RDNA3 稳定性
# export VLLM_USE_TRITON_FLASH_ATTN=0   # attention 异常时的退路（见 PHASE1.md §9）

# ---- 国内 HF 镜像 ----
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

MODEL=${MODEL:-$HOME/models/Qwen2.5-7B-Instruct}
PORT=${PORT:-8080}
MAX_LEN=${MAX_LEN:-16384}               # 24GB: 7B bf16 权重 ~15GB，16k 上下文 KV 充裕

# Phase 2: 装好 amdk 后（pip install -e infer/）追加参数启用 RDNA3 kernel：
#   bash launch_baseline.sh --attention-backend CUSTOM

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization 0.90 \
  --served-model-name qwen2.5-7b-baseline \
  "$@"
