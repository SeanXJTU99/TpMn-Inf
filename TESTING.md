# TESTING.md — 测试操作步骤

## 分类

| 类别 | 环境 | 说明 |
|------|------|------|
| **eval 硬校验** | Windows 本机 | 纯 Python，regex/JSON 无 GPU 依赖，**立即可跑** |
| **kernel 正确性** | WSL2 + ROCm | 需要 GPU + Triton |
| **端到端 benchmark** | WSL2 + ROCm | 需要 vLLM server 运行中 |
| **自动调参** | WSL2 + ROCm | 需要 vLLM server + 可重启 |

---

## 一、eval 硬校验（Windows 本机，立即可跑）

### 1.1 单项冒烟

```cmd
cd E:\gameAMDenging
python -c "from training.eval.checks.persona import run_all_persona; r=run_all_persona('冬木市的夜晚，你走在桥上。我认为这很危险。根据GM报告，你应该撤退。',{}); [print(f'{x.name}: {\"PASS\" if x.passed else \"FAIL\"} — {x.detail}') for x in r]"
```

预期输出：
```
persona_first_person: FAIL — 发现 1 处第一人称叙事：['我认为']
persona_third_person: PASS — 无违规
persona_meta_language: FAIL — 发现 1 处元语言泄出：['根据GM报告']
persona_markdown_leak: PASS — 无违规
persona_english_intrusion: PASS — 英文比例 0.00%
```

### 1.2 全量 hard checks

```cmd
cd E:\gameAMDenging
python -c "
from training.eval.checks.persona import run_all_persona
from training.eval.checks.schema  import run_all_schema
from training.eval.checks.leak    import run_all_leak
from training.eval.checks.slop    import run_all_slop

# 模拟一段 Narrator 输出（含故意违规）
text = '''冬木市的夜晚，你走在未远川大桥上。我认为这很危险。根据GM报告，
Berserker 正朝你冲来。你的 Servant 灵体化后撤，嘴角勾起一抹弧度。
魔力回路的灼痛提醒你今夜已消耗过半。'''

print('=== persona (5 checks) ===')
for r in run_all_persona(text, {}):
    print(f'  {r.name:30s} {\"PASS\" if r.passed else \"FAIL\"}')

print()
print('=== schema (4 checks) ===')
arb_json = '{\"judgment_report\":{\"result\":\"ok\",\"damage\":30},\"updated_memory_system\":{}}'
for r in run_all_schema(arb_json, {}):
    print(f'  {r.name:30s} {\"PASS\" if r.passed else \"FAIL\"}')

print()
print('=== leak (3 checks) ===')
for r in run_all_leak(text, {}):
    print(f'  {r.name:30s} {\"PASS\" if r.passed else \"FAIL\"}')

print()
print('=== slop (4 checks) ===')
for r in run_all_slop(text, {'samples': [text, text + '。' * 50]}):
    print(f'  {r.name:30s} {\"PASS\" if r.passed else \"FAIL\"}')
"
```

### 1.3 盲测工具

```cmd
cd E:\gameAMDenging
# 准备两段叙事文本（story_a.txt / story_b.txt），行数相等
python training\eval\human\blind_ab.py --tag demo --interleaved story_a.txt story_b.txt
```

### 1.4 评估报告入口

```cmd
cd E:\gameAMDenging
python training\eval\runners\run_eval.py --input fake_results.jsonl --tag dryrun
```

---

## 二、WSL2 环境搭建（一次性）

按 `infer/PHASE1.md` §1-§5 执行：

```bash
# 1. WSL2 Ubuntu 内验证 GPU
ls /dev/dxg && /opt/rocm/bin/rocminfo | grep gfx          # 期望 gfx1100

# 2. 安装 vLLM（ROCm 预编译 wheel）
python3 -m venv ~/venv-vllm && source ~/venv-vllm/bin/activate
pip install uv
uv pip install vllm==0.18.0 \
  --extra-index-url https://wheels.vllm.ai/rocm/0.18.0/rocm700

# 3. 验证 PyTorch ROCm
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望: True / AMD Radeon RX 7900 XTX

# 4. 拉取模型（国内走镜像）
export HF_ENDPOINT=https://hf-mirror.com
pip install "huggingface_hub[cli]"
hf download Qwen/Qwen2.5-7B-Instruct --local-dir ~/models/Qwen2.5-7B-Instruct

# 5. 安装 amdk 插件
pip install -e /mnt/e/gameAMDenging/infer/
# 验证: python3 -c "import amdk; print(amdk.__version__)"
# 期望: 0.1.0
```

---

## 三、冒烟测试（确认 vLLM 可跑）

```bash
source ~/venv-vllm/bin/activate

# 启动（默认 bf16 / 16k 上下文 / :8080）
bash /mnt/e/gameAMDenging/infer/scripts/launch_baseline.sh

# 另开终端
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-7b-baseline","messages":[{"role":"user","content":"用一句话描述冬木市的夜晚"}],"max_tokens":64}'
# 期望: 有中文 JSON 返回，无 NaN/乱码
```

---

## 四、基线 benchmark

```bash
source ~/venv-vllm/bin/activate

# 基线（上游 Triton backend，不启用 amdk）
python3 /mnt/e/gameAMDenging/infer/bench/baseline_bench.py --runs 5 --tag baseline

# 查看结果
cat /mnt/e/gameAMDenging/infer/bench/results/baseline_*.json | python3 -m json.tool | head -30
```

---

## 五、P0 attention 正确性测试

```bash
source ~/venv-vllm/bin/activate

# 单元测试（不需要 vLLM server，直接用 torch + triton 跑 kernel）
pytest /mnt/e/gameAMDenging/infer/amdk/tests/test_attention_correctness.py -v
```

期望输出：
```
test_pure_decode_3d[qwen25-gqa-bfloat16] PASSED
test_pure_decode_2d[qwen25-gqa-bfloat16] PASSED
test_mixed_prefill_decode[qwen25-gqa-bfloat16] PASSED
test_long_context_decode[qwen25-gqa-bfloat16] PASSED
test_single_token_seq[qwen25-gqa-bfloat16] PASSED
...
10 passed
```

若 FAIL：错误信息含具体 max_diff 值 → 贴到会话里，分析是 tile size 还是 mask 问题。

---

## 六、启用 amdk backend 后 benchmark 对照

```bash
source ~/venv-vllm/bin/activate

# 停止之前的 vLLM server (Ctrl+C)，重新启动带 CUSTOM backend
bash /mnt/e/gameAMDenging/infer/scripts/launch_baseline.sh --attention-backend CUSTOM

# 对照 benchmark
python3 /mnt/e/gameAMDenging/infer/bench/baseline_bench.py --runs 5 --tag amdk_p0

# 对比 baseline vs amdk_p0 的 TTFT/E2E/decode tok/s
ls /mnt/e/gameAMDenging/infer/bench/results/
```

---

## 七、P1 / P3 正确性测试

```bash
source ~/venv-vllm/bin/activate

# P1: Fused RMSNorm + QKV + RoPE
pytest /mnt/e/gameAMDenging/infer/amdk/tests/test_fused_qkv_rope.py -v
# 期望: 2 passed (bf16 + fp16, bias + no_bias)

# P3: Fused GEGLU + FFN
pytest /mnt/e/gameAMDenging/infer/amdk/tests/test_fused_geglu.py -v
# 期望: 2 passed (bf16 + fp16)
```

---

## 八、自动调参

```bash
source ~/venv-vllm/bin/activate

# dry-run — 确认搜索空间
python3 /mnt/e/gameAMDenging/infer/amdk/tune_attention.py --dry-run

# P0 decode sweep（需 vLLM server 运行在 :8080）
python3 /mnt/e/gameAMDenging/infer/amdk/tune_attention.py --kernel decode

# 产出 infer/amdk/amd_tune.json → 重启 vLLM 后自动生效
cat /mnt/e/gameAMDenging/infer/amdk/amd_tune.json
```

中断后可恢复：
```bash
python3 /mnt/e/gameAMDenging/infer/amdk/tune_attention.py --kernel decode \
  --resume /tmp/amdk_tune_decode_checkpoint.json
```

---

## 测试顺序总结

```
1. Windows 本机         → eval 硬校验（§一，可立即跑）
2. WSL2 环境搭建         → PHASE1.md §1-§5（§二）
3. 冒烟测试              → curl 确认 vLLM 可跑（§三）
4. 基线 benchmark        → 记录 baseline 数据（§四）
5. P0 正确性             → pytest 单元测试（§五）
6. P0 端到端对照         → --attention-backend CUSTOM + benchmark（§六）
7. P1/P3 正确性          → pytest 单元测试（§七）
8. 自动调参              → tune_attention.py → amd_tune.json（§八）
9. 调参后重跑 benchmark  → 确认性能提升幅度
```
