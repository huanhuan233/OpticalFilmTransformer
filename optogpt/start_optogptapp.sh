#!/bin/bash
# =========================================================
# OptoGPT 一键启动（后端 Django + 前端 Vite）
# 放置路径：/home/sysadmin/WorkSpace/PXY/optogpt/optogpt/start_optogptapp.sh
# 端口：后端 8174 / 前端 8173
# 日志：../logs/
# Python venv：/home/sysadmin/WorkSpace/PXY/optogpt_env
# =========================================================
set -euo pipefail

# -------- 1) 路径设置 --------
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"          # .../optogpt/optogpt
FRONTEND_DIR="$BASE_DIR/frontend"
BACKEND_DIR="$BASE_DIR/optogpt_backend"

LOG_DIR="$(dirname "$BASE_DIR")/logs"
mkdir -p "$LOG_DIR"
FRONT_LOG="$LOG_DIR/frontend.log"
BACK_LOG="$LOG_DIR/backend.log"

# -------- 2) Python 环境 --------
VENV_ACT="/home/sysadmin/WorkSpace/PXY/optogpt_env/bin/activate"
if [ -f "$VENV_ACT" ]; then
  source "$VENV_ACT"
  echo "✅ 已激活 Python venv: $VIRTUAL_ENV"
else
  echo "⚠️ 未找到 venv ($VENV_ACT)，使用系统 Python"
fi

# -------- 3) 清理旧进程 --------
pkill -f "manage.py runserver 0.0.0.0:8174" 2>/dev/null || true
pkill -f "vite.*8173" 2>/dev/null || true
pkill -f "npm run dev" 2>/dev/null || true

# -------- 4) 启动 Django 后端 --------
echo "🧠 启动后端 Django (端口 8174)..."
cd "$BACKEND_DIR"
nohup python manage.py runserver 0.0.0.0:8174 >"$BACK_LOG" 2>&1 &
BACK_PID=$!
echo "后端 PID: $BACK_PID | 日志: $BACK_LOG"

# 检查端口启动（30 秒内）
echo -n "⏳ 等待后端 8174 启动中"
for i in {1..60}; do
  if (echo >/dev/tcp/127.0.0.1/8174) >/dev/null 2>&1; then
    echo " ✓"
    break
  fi
  echo -n "."
  sleep 0.5
  if ! ps -p "$BACK_PID" >/dev/null 2>&1; then
    echo -e "\n❌ 后端进程退出，最近日志如下："
    tail -n 40 "$BACK_LOG" || true
    exit 1
  fi
done

# -------- 5) 启动前端 Vite --------
echo "🌐 启动前端 Vite (端口 8173)..."
cd "$FRONTEND_DIR"
nohup npm run dev >"$FRONT_LOG" 2>&1 &
FRONT_PID=$!
echo "前端 PID: $FRONT_PID | 日志: $FRONT_LOG"

# -------- 6) 启动成功提示 --------
echo ""
echo "✅ OptoGPT 全部启动完成"
echo "---------------------------------------"
echo "前端访问:  http://<服务器IP>:8173"
echo "后端接口:  http://<服务器IP>:8174"
echo "日志文件:"
echo "  前端: $FRONT_LOG"
echo "  后端: $BACK_LOG"
echo ""
echo "停止服务命令:"
echo "  kill $FRONT_PID $BACK_PID"
echo "或"
echo "  pkill -f 'manage.py runserver 0.0.0.0:8174|vite|npm run dev'"
echo "---------------------------------------"
