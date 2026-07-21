# SPDX-License-Identifier: Apache-2.0
"""SGLang 插件注册入口。

TODO: 确认 SGLang 插件机制后实现。
  预期方式: SGLang 通过在模型配置里指定 attention_backend 来选择 backend
  可能需要: 在 sglang.srt.layers.attention 里注册自定义 backend class
"""


def register() -> None:
    raise NotImplementedError(
        "SGLang plugin registration is a stub — study SGLang source first"
    )
