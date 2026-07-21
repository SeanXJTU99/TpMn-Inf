#!/usr/bin/env bash
# Phase 2: SGLang serve Qwen2.5-7B-Instruct（RX 7900 XTX / gfx1100 / WSL2）
# 用法: bash launch_sglang.sh [额外 sglang serve 参数...]
set -euo pipefail

# ---- RDNA3 / WSL2 必需环境变量 ----
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100
export GPU_MAX_HW_QUEUES=1

# ---- 国内 HF 镜像 ----
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

MODEL=${MODEL:-$HOME/models/Qwen2.5-7B-Instruct}
PORT=${PORT:-8080}
MAX_LEN=${MAX_LEN:-16384}

# ---- SGLang 专属: triton attention backend 超参 ----
# 环境变量传递给 sglang_adapter 的 tune_config
export AMDK_TUNE_CONFIG=${AMDK_TUNE_CONFIG:-}  # 可覆盖 JSON 路径

# Phase 3: 装好 tp-inf 后（pip install -e infer/）通过 --attention-backend amdk 启用 RDNA3 kernel：
#   bash launch_sglang.sh --attention-backend amdk

exec sglang serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype bfloat16 \
  --context-length "$MAX_LEN" \
  --mem-fraction-static 0.90 \
  --served-model-name qwen2.5-7b-baseline \
  "$@"
