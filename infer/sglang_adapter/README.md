# SGLang adapter

将 `infer.kernels` 中的 RDNA3 特化 Triton kernel 注入 SGLang。

## 待办

1. 拉 SGLang 源码 sparse clone（与 vLLM 同样方式）
2. 阅读 `sglang/srt/layers/attention/` 确认 backend 接口
3. 实现 `backend.py` — 继承/替换 SGLang 默认 Triton backend
4. 实现 `plugin.py` — 确认注册方式后补全
5. 写 `launch_sglang.sh` — 启动脚本
6. 在 `baseline_bench.py` 增加 SGLang server 模式
7. 正确性测试: `pytest kernels/tests/`（kernel 层与框架无关，直接复用）
