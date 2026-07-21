# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

《万能愿望机：残响协议》— A FastAPI backend for a Type-Moon Holy Grail War real-time AI narrative game. The server receives player commands, routes them through a multi-tier AI pipeline (Router → Arbiter → Narrator), and returns dark literary narrative in Gen Urobuchi's style.

- **Language**: Python 3.11+
- **Server**: FastAPI + uvicorn, in-memory session storage
- **AI providers**: DeepSeek V4 (`deepseek-v4-flash` / `deepseek-v4-pro`) via OpenAI-compatible SDK; optional local Ollama (`qwen2.5:3b`) for routing
- **Client**: `client.py` — pure stdlib terminal CLI, zero dependencies

## Commands

```bash
# Install server dependencies
pip install fastapi uvicorn openai pydantic

# Run all 104 tests
python -m pytest tests/ -q

# Run a single test file
python -m pytest tests/test_atomic_rules.py -v

# Run a single test
python -m pytest tests/test_atomic_rules.py::TestCommandSpells::test_zero_spells_raises -v

# Start server (requires DEEPSEEK_API_KEY env var)
set DEEPSEEK_API_KEY=sk-xxx
python game_server.py

# Start CLI client (separate terminal)
python client.py

# Check server health
curl http://127.0.0.1:8000/health

# Kill stale server on port 8000 (Windows)
netstat -ano | findstr ":8000"          # find PID in last column
taskkill //PID <PID> //F                # force kill

# Verify new code is loaded (should contain "player_servant_name", not just 4 keys)
curl -s http://127.0.0.1:8000/api/game/init -X POST -d "{}" -H "Content-Type: application/json"
```

## Architecture: turn execution pipeline

Every player turn flows through this exact chain (in `game_server.py:execute_game_turn`):

```
1. Session validation (in-memory dict lookup)
2. 🔒 Hard atomic rules check (atomic_rules.py) — returns 422 BEFORE any AI call if violated
3. Load servant profiles from servant_db.json
4. Router: Ollama local → fallback deepseek-v4-flash (complexity score 1-10)
5. Arbiter: deepseek-v4-pro (always — 主裁判全程最强推理, no tier downgrade)
   - Receives full snapshot + servant profiles + player input
   - Outputs JSON: judgment_report + updated_memory_system
   - Pydantic model_validate intercepts key typos, type errors
6. 🏁 Code-layer victory check (determine_game_result) — not AI-decided
7. Narrator: deepseek-v4-flash — converts arbiter report into literary text
8. Return narrative + updated snapshot + optional game_over info
```

**Key design principle**: "代码卡关，AI 润色" — hard game rules (command spells, HP, death, turn limit) are enforced in Python `if/else` in `atomic_rules.py`. The AI never touches life-or-death decisions.

## Core data models (models.py)

- **`CharacterState`** — per-character runtime state: `hp`, `max_hp`, `status`, `location`, `command_spells`, `is_alive`, `mana_remaining`. Uses `ConfigDict(extra="forbid")` to reject AI-created fields.
- **`GameMemorySystem`** — dual-track memory: `chronicle_history` (append-only historical facts), `current_snapshot` (dict of CharacterState), `current_day` (1-N), `current_phase` ("day"|"night")
- **`GameOverInfo`** — `is_over`, `result` ("victory"|"draw"|"defeat"), `winner_name`, `epilogue`
- **`EngineFinalResponse`** — `narrative` + `memory_system` + `turn_summary` + optional `game_over`

## Hard atomic rules (atomic_rules.py)

All rules checked before AI calls. Violations return HTTP 422 with machine-readable codes:

| Rule | Code | Triggers on |
|------|------|-------------|
| Input safety | `INPUT_EMPTY`, `INPUT_TOO_LONG` | Empty/whitespace-only or >2000 chars |
| Character alive | `CHARACTER_DEAD` | Player targets dead character (`is_alive=False` or `hp<=0`) |
| Command spells | `NO_COMMAND_SPELLS` | Zero command spells + input contains spell keywords (CN/EN/JP) |
| Mana for NP | `INSUFFICIENT_MANA` | Mana < 30% + input contains Noble Phantasm keywords |
| Day limit | `WAR_ENDED` | `current_day > max_days` (7) |
| Game over | `GAME_ALREADY_OVER` | Session already ended |

`RuleViolation.__str__` returns `[CODE] message` format — used in HTTP 422 detail for machine parsing.

**`determine_game_result`** runs AFTER arbiter judgment (not before):
- 1 master alive → that side wins
- 0 masters alive → draw
- `current_day > 7` → draw (timeout)
- Otherwise → continue

## AI client (ai_client.py)

- Uses `AsyncOpenAI` (OpenAI-compatible SDK) pointed at DeepSeek's endpoint
- `call_deepseek(model, system_prompt, user_prompt, temperature, response_schema)` — async, returns `(text, usage_dict)`
- `call_ollama(...)` — same signature, with `asyncio.wait_for(timeout=5s)`. Caller catches timeout and falls back to DeepSeek
- Clients are lazily initialized module-level singletons

## Game initialization

`POST /api/game/init` creates a session with:
- 7 randomly drawn servants from `servant_db.json` (35 cards currently)
- 7 masters: `Protagonist_Master` (player, CS=3) + 6 enemy masters (言峰绮礼, 远坂时臣, 爱因兹贝伦的御主, 间桐脏砚, 肯尼斯, 流浪魔术师), each with 1-3 command spells
- `current_day=1`, `current_phase="night"`

## Servant database (servant_db.json)

35 cards, keyed by `Class_HistoricalName` (e.g. `Saber_Artoria`). Each card has `true_name`, `title`, `description`, `attributes`, `class_abilities`, `skills` (list), `noble_phantasm` (name/rank/effect dict).

When a servant key is missing from the DB, the system passes a warning string to the arbiter instead of crashing.

## Session model

In-memory `Dict[str, GameSession]`. Each session tracks: `session_id` (8-char UUID), `active_servant_keys`, `turn_count`, `is_game_over`, `game_result`, `memory_system`. Sessions expire after `SESSION_TTL_MINUTES` (120 min) of inactivity.

## Test structure

- `tests/conftest.py` — shared fixtures: `alive_master`, `alive_servant`, `dead_character`, `low_mana_servant`, `zero_spells_master`, `sample_memory`, `empty_memory`
- `tests/test_models.py` — 34 tests: Pydantic validation, coercion, bounds, `extra="forbid"`
- `tests/test_atomic_rules.py` — 33 tests: every rule individually + `run_all_atomic_checks` integration + `RuleViolation`
- `tests/test_config.py` — 17 tests: defaults, validation, env vars
- `tests/test_game_server.py` — 18 tests: all API endpoints with mocked AI calls (`AsyncMock`). Mock data must include `Enemy_Master` to prevent immediate victory detection.

Mock patterns: use `unittest.mock.patch("game_server.call_deepseek", AsyncMock(...))`. For multi-turn tests, use `side_effect=callable` (function form) instead of a finite list to avoid exhaustion.

## Important constraints

- **Never use old model names** `deepseek-chat` or `deepseek-reasoner` — they stop working 2026-07-24. Always use `deepseek-v4-flash` and `deepseek-v4-pro`.
- **Pydantic v2 coercion is intentional** — strings like `"100"` auto-convert to `int 100`. This is beneficial: AI sometimes outputs numeric strings, and the system remains robust.
- **`GameMemorySystem.current_day` has no `le=7` Field constraint** — this is deliberate. Terminal states (day 8+) must be constructable so `atomic_rules.check_day_limit` can intercept them. The Field only enforces `ge=1`.
- **All `current_snapshot` keys must be preserved** — the arbiter must not add, remove, or rename character keys. Only field values within each `CharacterState` may change.
- **Initial snapshot must have 14 characters** (7 servants + 7 masters). Missing enemy masters cause immediate false victory.
