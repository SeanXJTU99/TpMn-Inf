"""
《万能愿望机：残响协议》— 终端 CLI 客户端
=============================================
纯标准库实现，零额外依赖。
连接 localhost:8000，循环交互直到游戏结束。
"""

import json
import urllib.request
import urllib.error
import sys
import os
import textwrap

# ==========================================
# 配置
# ==========================================
CLIENT_VERSION = "3.2.0"
SERVER_URL = os.environ.get("TYPEMOON_SERVER", "http://127.0.0.1:8000")
TERMINAL_WIDTH = 72


# ==========================================
# HTTP 辅助
# ==========================================
def api_post(path: str, body: dict) -> dict:
    """发送 POST 请求到服务器。"""
    url = f"{SERVER_URL}{path}"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(error_body).get("detail", error_body)
        except json.JSONDecodeError:
            detail = error_body
        raise RuntimeError(f"服务器返回 {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"无法连接服务器 ({SERVER_URL}): {e.reason}")


def api_get(path: str) -> dict:
    """发送 GET 请求。"""
    url = f"{SERVER_URL}{path}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"无法连接服务器: {e.reason}")


# ==========================================
# 终端渲染
# ==========================================
def clear_screen():
    os.system("cls" if sys.platform == "win32" else "clear")


def divider(char: str = "─") -> str:
    return char * TERMINAL_WIDTH


def box(text: str, title: str = "") -> str:
    """给文本加框。"""
    lines = []
    lines.append(f"┌{divider()}┐")
    if title:
        padding = (TERMINAL_WIDTH - len(title) - 2) // 2
        lines.append(f"│ {' ' * padding}{title}{' ' * (TERMINAL_WIDTH - padding - len(title) - 2)} │")
        lines.append(f"├{divider()}┤")
    for line in text.split("\n"):
        # 按终端宽度自动折行
        wrapped = textwrap.wrap(line, width=TERMINAL_WIDTH - 2)
        if not wrapped:
            lines.append(f"│ {' ' * (TERMINAL_WIDTH - 2)} │")
        for w in wrapped:
            pad = TERMINAL_WIDTH - 2 - _display_width(w)
            lines.append(f"│ {w}{' ' * max(0, pad)} │")
    lines.append(f"└{divider()}┘")
    return "\n".join(lines)


def _display_width(s: str) -> int:
    """估算字符串在终端中的显示宽度（中文=2，英文=1）。"""
    width = 0
    for ch in s:
        if "一" <= ch <= "鿿" or "　" <= ch <= "〿" or "＀" <= ch <= "￯":
            width += 2
        else:
            width += 1
    return width


def print_narrative(text: str):
    """打印 AI 叙事——不做 box，给纯文字沉浸感。"""
    print()
    for line in text.split("\n"):
        if not line.strip():
            print()
        else:
            wrapped = textwrap.wrap(line, width=TERMINAL_WIDTH)
            for w in wrapped:
                print(f"  {w}")
    print()


def print_game_over(info: dict):
    """打印结局画面。"""
    result = info["result"]
    if result == "victory":
        emoji = "🏆"
        title = "圣 杯 战 争 · 胜 利"
    elif result == "defeat":
        emoji = "💀"
        title = "圣 杯 战 争 · 败 北"
    else:
        emoji = "⚖️"
        title = "圣 杯 战 争 · 平 局"

    print()
    print(divider("═"))
    print(f"  {emoji}  {title}")
    if info.get("winner_name"):
        print(f"     获胜者: {info['winner_name']}")
    print(divider("═"))
    if info.get("epilogue"):
        print()
        for line in textwrap.wrap(info["epilogue"], width=TERMINAL_WIDTH):
            print(f"  {line}")
    print()
    print(divider("═"))


def print_status_bar(day: int, phase: str, session_id: str, turn: int):
    """打印顶部状态栏。"""
    phase_icon = "🌙" if phase == "night" else "☀️"
    phase_cn = "夜" if phase == "night" else "昼"
    print(divider("═"))
    print(f"  {phase_icon}  第 {day} 天 · {phase_cn}  |  "
          f"回合 {turn}  |  Session: {session_id}")
    print(divider("═"))


# ==========================================
# 主交互循环
# ==========================================
def run_game():
    """完整的游戏主循环。"""
    clear_screen()
    print(divider("═"))
    print(f"  万能愿望机：残响协议 — 终端客户端 v{CLIENT_VERSION}")
    print("  输入 /help 查看可用命令")
    print(divider("═"))

    # 检查服务器
    try:
        health = api_get("/health")
    except RuntimeError as e:
        print(f"\n  ❌ {e}")
        print("  请先启动服务器: python game_server.py\n")
        return

    server_ver = health.get("version", "?")
    print(f"\n  服务器版本: v{server_ver}  |  英灵库: {health.get('servant_count', '?')} 张卡片")
    print(f"  Ollama: {'✓' if health.get('ollama_available') else '✗'}  |  DeepSeek: {'✓' if health.get('deepseek_configured') else '✗'}")

    if not health.get("deepseek_configured"):
        print("\n  ⚠️  DEEPSEEK_API_KEY 未设置，AI 调用将失败。")
        print("  请设置环境变量后重启服务器。\n")

    while True:
        # ── 初始化新游戏 ──
        print("\n  正在初始化游戏...")
        try:
            data = api_post("/api/game/init", {})
        except RuntimeError as e:
            print(f"\n  ❌ 初始化失败: {e}\n")
            return

        session_id = data["session_id"]
        memory = data["memory_system"]
        narrative = data.get("message", "")
        player_servant_name = data.get("player_servant_name", "???")
        player_servant_key = data.get("player_servant_key", "")
        total_servants = len(memory.get("active_servant_keys", []))

        clear_screen()
        print(f"\n  ✦ 圣杯选中了 {total_servants} 骑英灵参战 ✦")
        print(f"  你的契约英灵: {player_servant_name}")
        print()

        if narrative:
            print_narrative(narrative)

        # ── 回合循环 ──
        game_over = False
        while not game_over:
            current_day = memory.get("current_day", 1)
            current_phase = memory.get("current_phase", "night")
            turn_count = data.get("turn_summary", {}).get("turn_count", 0)

            print_status_bar(current_day, current_phase, session_id, turn_count)

            # 获取玩家输入
            try:
                user_input = input("\n  > 你的指令: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n  游戏中断。再见。\n")
                return

            if not user_input:
                continue

            # 特殊命令
            if user_input.startswith("/"):
                result = handle_command(user_input, session_id)
                if result == "new_game":
                    break  # 跳出回合循环，回到外层 while 重新 init
                elif result == "quit":
                    return
                else:
                    continue  # 无效命令，继续当前回合

            # 发送回合
            print("\n  ⏳ 因果律推演中...")
            try:
                data = api_post("/api/game/execute_turn", {
                    "session_id": session_id,
                    "player_input": user_input,
                })
            except RuntimeError as e:
                print(f"\n  ❌ {e}")
                # 如果是游戏已结束，询问新游戏
                if "GAME_ALREADY_OVER" in str(e) or "WAR_ENDED" in str(e):
                    choice = _ask_new_game_or_quit()
                    if choice == "new_game":
                        break
                    else:
                        return
                continue

            # 显示叙事
            narrative = data.get("narrative", "")
            memory = data.get("memory_system", memory)
            print_narrative(narrative)

            # 结算摘要（可选，调试用）
            ts = data.get("turn_summary", {})
            if ts:
                score = ts.get("complexity_score", "?")
                router = ts.get("router_source", "?")
                arbiter = ts.get("arbiter_model", "?")[:20]
                tokens = ts.get("total_tokens", 0)
                latency = ts.get("total_latency_sec", 0)
                print(f"  [评分={score} 路由={router} 裁判={arbiter} | "
                      f"{tokens} tokens | {latency}s]")

            # 检查游戏结束
            go = data.get("game_over")
            if go and go.get("is_over"):
                print_game_over(go)
                memory = data.get("memory_system", memory)
                # 显示最终状态
                snap = memory.get("current_snapshot", {})
                if snap:
                    print("\n  最终存活状态:")
                    for name, state in snap.items():
                        mark = "✓" if state.get("is_alive") and state.get("hp", 0) > 0 else "✗"
                        print(f"    {mark} {name}: HP={state.get('hp')} "
                              f"令咒={state.get('command_spells')} "
                              f"位置={state.get('location')}")

                game_over = True
                choice = _ask_new_game_or_quit()
                if choice == "new_game":
                    break  # 跳出内层，回到外层重新 init
                else:
                    return

        # 如果 break 了内层循环且是 new_game，继续外层循环
        continue


def handle_command(cmd: str, session_id: str) -> str:
    """处理特殊命令。返回 "new_game" / "quit" / "continue"."""
    cmd_lower = cmd.lower()

    if cmd_lower in ("/quit", "/q", "/exit"):
        print("\n  游戏中断。愿你下次获得更好的角色。\n")
        return "quit"

    elif cmd_lower in ("/new", "/newgame", "/restart"):
        print("\n  放弃当前战争，重新召唤英灵...\n")
        return "new_game"

    elif cmd_lower in ("/help", "/h", "/?"):
        print(f"""
  {divider('─')}
  可用命令:
    /new      放弃当前战局，开启新的游戏
    /quit     完全退出游戏
    /help     显示此帮助
    /status   查看当前存活状态
    /history  查看编年史
  {divider('─')}
""")
        return "continue"

    elif cmd_lower in ("/status", "/s"):
        # 查询 session 状态
        try:
            info = api_get(f"/api/game/session/{session_id}")
            print(f"\n  Session: {info['session_id']}")
            print(f"  回合数: {info['turn_count']}")
            print(f"  参战英灵: {', '.join(info['active_servant_keys'])}")
            print(f"  创建时间: {info['created_at']}")
            print(f"  最后活跃: {info['last_turn_at']}")
        except RuntimeError as e:
            print(f"\n  ❌ {e}")
        return "continue"

    elif cmd_lower == "/history":
        # 历史需要从 memory_system 中取 —— 这里用一个简单的变通
        print("\n  编年史记录暂不支持在命令中直接查看，")
        print("  它会在每回合执行后自动更新。\n")
        return "continue"

    else:
        print(f"\n  未知命令: {cmd}。输入 /help 查看可用命令。")
        return "continue"


def _ask_new_game_or_quit() -> str:
    """游戏结束后询问玩家意图。"""
    print()
    while True:
        try:
            choice = input("  [N] 开启新战局  [Q] 退出  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见。\n")
            return "quit"
        if choice in ("n", "new", "1"):
            print("\n  新的游戏即将开幕...\n")
            return "new_game"
        elif choice in ("q", "quit", "exit", "2"):
            print("\n  愿圣杯与你的记忆同在。再见。\n")
            return "quit"
        else:
            print("  请输入 N (新战局) 或 Q (退出)")


# ==========================================
# 入口
# ==========================================
if __name__ == "__main__":
    run_game()
