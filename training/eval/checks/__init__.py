# SPDX-License-Identifier: Apache-2.0
"""eval checks — 程序化硬校验（零 GPU / 零 API 依赖，本机直接跑）。

每个 check 是签名为 (text: str, context: dict) -> CheckResult 的可调用对象。
"""

from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# 公共类型
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """单项 check 的输出。"""
    name: str                           # e.g. "persona_first_person"
    passed: bool
    violations: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "violations": self.violations,
            "metrics": self.metrics,
            "detail": self.detail,
        }


CheckFn = Callable[[str, dict[str, Any]], CheckResult]
