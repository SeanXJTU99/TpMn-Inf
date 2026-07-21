# SPDX-License-Identifier: Apache-2.0
"""SGLang 插件注册入口。

SGLang 无 formal plugin entry_point（不像 vLLM 的 general_plugins）。
注册方式：在 SGLang 进程启动时，在 ModelRunner.init_attention_backend()
之前调用 register()，把 "amdk" backend 注入 ATTENTION_BACKENDS 字典。

启用:
  export SGLANG_PLUGINS=amdk   # 或直接 python 侧 import sglang_adapter.plugin; register()
  sglang serve ... --attention-backend amdk

与 vLLM adapter 的对比:
  vllm:   entry_points → vllm.general_plugins → 自动 load
  sglang: 手动 import + register() → 注入 factory 到全局 dict
"""

import logging

logger = logging.getLogger(__name__)


def register() -> None:
    """将 "amdk" backend 注册到 SGLang 的 ATTENTION_BACKENDS。

    调用时机：SGLang serve 启动后、ModelRunner 初始化前。
    """
    from sglang.srt.layers.attention.attention_registry import (
        ATTENTION_BACKENDS,
        register_attention_backend,
    )

    if "amdk" in ATTENTION_BACKENDS:
        return  # 已注册（幂等）

    from sglang_adapter.backend import create_amdk_backend

    # 使用 SGLang 内置装饰器注册
    register_attention_backend("amdk")(create_amdk_backend)

    logger.info("amdk: registered 'amdk' attention backend (SGLang)")
