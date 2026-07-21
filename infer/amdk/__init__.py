# SPDX-License-Identifier: Apache-2.0
"""amdk — RDNA3 (gfx1100) 特化 Triton kernel 包。

经 vLLM general_plugins 入口注册为 CUSTOM attention backend，
vLLM 源码零修改（CLAUDE.md 核心决策 1）。
"""

__version__ = "0.1.0"
