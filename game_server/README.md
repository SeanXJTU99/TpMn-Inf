# 《万能愿望机：残响协议》AI 动态推演引擎 v3.3

专为 Steam / 移动端文字策略游戏设计的独立后端。FastAPI 框架，融合 **"GM 驱动世界推演"**、**"第二人称战术叙事"**、**"插卡式英灵库"**、**"双轨制记忆系统"** 以及 **"硬原子规则代码卡关"**。

---

## 🪐 引擎运作全流程

1. **游戏初始化** (`/api/game/init`)：从本地 `servant_db.json` 随机抽取 7 张英灵卡片，创建 session，返回初始快照。
2. **玩家提交指令** (`/api/game/execute_turn`)：前端发送 `session_id` + 玩家指令。
3. **🔒 硬原子规则校验**（代码层）：令咒剩余、HP、角色存活状态在 Python 中硬卡关。不满足条件 → 直接拦截返回 422。
4. **🌍 世界脉搏**（代码层）：随机抽取 1-2 组存活敌对阵营作为本回合活跃方，生成事件种子。
5. **🧠 分流路由打分**：Ollama 本地 Qwen2.5-3B 优先；不可用则自动降级到 DeepSeek-V4-Flash。
6. **🎲 GM 推演**：全程 DeepSeek-V4-Pro。GM 驱动世界独立运转——空白回合禁令，即使玩家蛰伏，外界冲突也会主动渗透。
7. **🎭 第二人称战术叙事**：DeepSeek-V4-Flash 将 GM 报告转化为「你」视角的沉浸式战地纪实，附状态行和战术暗示。

---

## 🛠️ 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn openai pydantic
```

### 2. 配置 API 密钥

**DeepSeek API**（必需 — 云端 AI 主力）:
- Linux/Mac: `export DEEPSEEK_API_KEY="sk-xxxxxxxx"`
- Windows (CMD): `set DEEPSEEK_API_KEY="sk-xxxxxxxx"`

**Ollama**（可选 — 本地路由加速）:
```bash
# 安装 Ollama 后拉取模型
ollama pull qwen2.5:3b
```

### 3. 启动引擎

```bash
python game_server.py
```

服务器默认运行在 `http://127.0.0.1:8000`。

**⚠️ 重启前先检查旧进程** — 旧进程未退出会导致新代码不生效：

```bash
# Windows — 查看端口占用
netstat -ano | findstr ":8000"

# 如果看到 LISTENING，记下最后一列的 PID，强制结束：
taskkill //PID <PID号> //F

# 确认端口已释放（无输出 = 已释放）
netstat -ano | findstr ":8000"
```

然后重新 `python game_server.py` 启动新服务器。

**判断新代码是否生效**：调用 `/api/game/init`，若返回的 JSON 包含 `player_servant_name` 字段（而非 `???`），说明新代码已加载。

### 4. 启动终端客户端

```bash
# 新开一个终端
python client.py
```

纯标准库实现，零额外依赖。交互界面：

```
  🌙  第 1 天 · 夜  |  回合 0  |  Session: a1b2c3d4

  > 你的指令: 观察周边环境，确认召唤的英灵

  ⏳ 因果律推演中...

  [AI 返回的暗黑叙事...]

  > 你的指令: /new    # 放弃当前战局，重新开局
  > 你的指令: /quit   # 退出游戏
  > 你的指令: /help   # 查看所有命令
```

---

## 📡 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 + 各组件可用性探测 |
| `/api/game/init` | POST | 初始化新游戏，随机抽7骑英灵，返回 session_id |
| `/api/game/execute_turn` | POST | 执行一回合推演 |
| `/api/game/execute_turn/stream` | POST | SSE 流式推演 — 打字机效果，5 种事件 |
| `/api/game/session/{id}` | GET | 查看 session 状态（调试用） |
| `/api/game/sessions` | GET | 列出所有活跃 session（调试用） |
| `/` `/mobile` | GET | 手机端 Web 聊天界面 |

### `/api/game/init` 请求示例

```json
{
  "preferred_servants": ["Archer_Gilgamesh", "Saber_Artoria"]
}
```

`preferred_servants` 可选 — 指定偏好的英灵（用于测试），不足 7 张则自动补抽。

### `/api/game/execute_turn` 请求示例

```json
{
  "session_id": "a1b2c3d4",
  "player_input": "命令Saber_Artoria解放誓约胜利之剑，从正面轰击敌方城堡。"
}
```

不需要手动传 `memory_system` — 服务端自动管理。

### 响应示例

```json
{
  "narrative": "金色的光粒子在Saber高举的剑尖凝聚——那不是魔力，而是星球本身的意志...(后略)",
  "memory_system": {
    "active_servant_keys": ["Saber_Artoria", "Archer_Gilgamesh", ...],
    "chronicle_history": ["...", "第3回合：Saber解放Excalibur摧毁敌方阵地"],
    "current_snapshot": {
      "Saber_Artoria": {
        "hp": 100, "status": "宝具解放后轻微疲劳", "location": "冬木市·废墟前",
        "command_spells": 0, "is_alive": true, "mana_remaining": 50,
        "max_hp": 100
      }
    }
  },
  "turn_summary": {
    "complexity_score": 7,
    "router_source": "ollama",
    "arbiter_model": "deepseek-v4-pro",
    "total_tokens": 2847,
    "total_latency_sec": 3.21
  }
}
```

---

## 📁 项目结构

| 文件 | 说明 |
|------|------|
| `game_server.py` | 主引擎 — FastAPI 服务器 + 多级路由推演链路 |
| `config.py` | 集中配置中心 — 模型名、API Key、阈值一键切换 |
| `models.py` | Pydantic 强类型数据模型 — `CharacterState` 杜绝 key 拼错崩溃 |
| `atomic_rules.py` | 硬原子规则引擎 — 令咒/HP/死亡/魔力在代码层硬卡关 |
| `ai_client.py` | 统一 AI 客户端 — 封装 DeepSeek + Ollama 异步调用 |
| `system_prompts.py` | AI 系统提示词 — Router/GM/Narrator 三大角色设定 |
| `servant_db.json` | 英灵卡片库 — 35 骑英灵的硬核设定 |
| `static/mobile.html` | 手机端 Web 聊天界面 — 零依赖，SSE 流式打字机 |
| `docs/optimization_review.md` | 优化建议文档 — 完整的代码审查与演进方向 |
| `docs/bug_log.md` | Bug 日志 — 15 个已修复缺陷的完整记录 |
| `client.py` | 终端 CLI 客户端 — 纯标准库，零依赖 |

---

## 🎮 AI 模型架构

```
                              ┌─ 世界脉搏 (Python, 零成本) ─┐
                              │   随机活跃阵营 + 事件种子     │
                              └──────────────┬──────────────┘
                                             ↓
Router (复杂度打分)
  ├── Ollama Qwen2.5-3B (本地优先, 超时5s自动降级)
  └── DeepSeek-V4-Flash (降级后备)
                                             ↓
GM (游戏掌控者)   ─── DeepSeek-V4-Pro        (全程最强推理, 驱动世界运转)
                                             ↓
Narrator ────────── DeepSeek-V4-Flash        (第二人称战术叙事 + 状态行 + 战术暗示)
```

所有模型名称可在 `config.py` 中一键切换。

---

## 🔒 硬原子规则（代码卡关，AI 润色）

以下规则在 Python 层硬编码校验，AI 无权绕过：

- **令咒铁律**：`command_spells <= 0` → 所有令咒相关指令直接拦截
- **死亡铁律**：`is_alive=false` 或 `hp <= 0` → 该角色不可执行任何操作
- **魔力铁律**：`mana_remaining < 30` → 不可发动宝具
- **战争时限**：第 7 天结束后若未决出唯一胜者 → 平局（圣杯未显现）
- **Key 稳定性**：`CharacterState` 强类型模型在 Pydantic 校验层拦截 AI 的 key 拼写错误

---

## 🔧 配置说明 (`config.py`)

```python
# API 密钥
DEEPSEEK_API_KEY = "sk-xxxx"     # 必需
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# Ollama 本地模型
OLLAMA_ROUTER_MODEL = "qwen2.5:3b"
OLLAMA_TIMEOUT_SEC = 5.0         # 超时自动降级

# 模型分配（按需切换）
ROUTER_FALLBACK_MODEL = "deepseek-v4-flash"
ARBITER_LOW_TIER_MODEL = "deepseek-v4-pro"
ARBITER_HIGH_TIER_MODEL = "deepseek-v4-pro"  # 主裁判全程使用最强模型
NARRATOR_MODEL = "deepseek-v4-flash"

# 游戏参数
SERVANTS_PER_GAME = 7
MAX_DAYS = 7                 # 圣杯战争最大昼夜数（7天后无赢家=平局）
TIER_THRESHOLD = 9               # >= 9 分触发深度推理升舱
```

---

## 🧪 本地试玩流程

```bash
# 1. 确保 Ollama 运行中（可选）
ollama serve

# 2. 启动服务器
python game_server.py

# 3. 初始化一局游戏
curl -X POST http://127.0.0.1:8000/api/game/init \
  -H "Content-Type: application/json" \
  -d '{}'

# 4. 记录返回的 session_id，开始游戏
curl -X POST http://127.0.0.1:8000/api/game/execute_turn \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "你的session_id",
    "player_input": "观察周边环境，确认自身状态。"
  }'

# 5. 查看 session 状态
curl http://127.0.0.1:8000/api/game/session/你的session_id
```

---

## 📝 后续规划

- [x] 前端打字机特效（SSE 流式 + 手机端 Web 聊天界面）
- [ ] 后端持久化（Redis / SQLite）替代内存存储
- [ ] Context Caching（DeepSeek 上下文缓存命中后输入 1/10 价格）
- [ ] 双向快照反作弊（服务端完整状态 vs 前端剧情迷雾）
- [ ] 英灵库扩充至 50+ 张卡
- [ ] 支持玩家自定义御主名称与背景
