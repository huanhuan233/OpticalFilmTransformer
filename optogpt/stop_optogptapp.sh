#!/bin/bash
# =========================================================
# OptoGPT 一键停止脚本（强化版）
# 前端: 8173 (Vite/Node)  后端: 8174 (Django)
# 先按名称，再按端口；先 TERM 再 KILL
# =========================================================
set -euo pipefail

echo "🛑 停止 OptoGPT 前后端 ..."

kill_by_name() {
  local pattern="$1"
  local desc="$2"
  if pgrep -f "$pattern" >/dev/null 2>&1; then
    echo "→ 尝试按名称停止：$desc ($pattern) [TERM]"
    pkill -f "$pattern" || true
    sleep 1
  fi
}

kill_by_port() {
  local port="$1"
  local desc="$2"
  # 找到监听/占用该端口的 PID 列表
  local pids
  pids=$(lsof -ti tcp:"$port" || true)
  if [ -n "$pids" ]; then
    echo "→ 发现占用端口 $port 的进程：$pids  [TERM]"
    kill $pids || true
    sleep 1
  fi
  pids=$(lsof -ti tcp:"$port" || true)
  if [ -n "$pids" ]; then
    echo "→ 进程仍在，强制杀掉：$pids  [KILL]"
    kill -9 $pids || true
  fi
}

# 1) 按名称尝试（后端 Django）
kill_by_name "manage.py runserver 0.0.0.0:8174" "Django runserver(8174)"
kill_by_name "python .*manage.py runserver .*8174" "Django runserver(8174)"

# 2) 按名称尝试（前端 Vite / npm / node）
kill_by_name "vite.*8173"              "Vite dev server(8173)"
kill_by_name "node .*vite"             "Node+Vite(8173)"
kill_by_name "npm run dev"             "npm run dev(8173)"
kill_by_name "pnpm vite"               "pnpm vite(8173)"
kill_by_name "yarn vite"               "yarn vite(8173)"

# 3) 端口级别兜底（最有效）
kill_by_port 8174 "后端(8174)"
kill_by_port 8173 "前端(8173)"

# 4) 验证
sleep 1
front_left=$(ss -ltnp 2>/dev/null | grep ":8173 " || true)
back_left=$(ss -ltnp 2>/dev/null | grep ":8174 " || true)

if [ -z "$front_left" ] && [ -z "$back_left" ]; then
  echo "✅ 已完全停止（8173/8174 均无监听）"
else
  echo "⚠️ 仍有监听："
  [ -n "$front_left" ] && echo "  8173: $front_left"
  [ -n "$back_left" ] && echo "  8174: $back_left"
  echo "可手动查看： ss -ltnp | grep -E ':8173|:8174'"
fi
