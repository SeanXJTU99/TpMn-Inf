# SPDX-License-Identifier: Apache-2.0
"""vLLM 插件入口：把 amdk backend 注册到 AttentionBackendEnum.CUSTOM。

安装（vLLM 同一 venv）:  pip install -e infer/
启用:  vllm serve ... --attention-backend CUSTOM
"""

import logging

logger = logging.getLogger(__name__)


def register() -> None:
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "amdk.backend.Rdna3AttentionBackend",
    )
    logger.info("amdk: registered Rdna3AttentionBackend as CUSTOM")
