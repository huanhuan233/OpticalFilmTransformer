#!/usr/bin/env bash
# 一键停止 run_optogpt.py 的后台进程

# 找到所有 run_optogpt.py 相关的进程
PIDS=$(ps -ef | grep "[p]ython .*run_optogpt.py" | awk '{print $2}')

if [ -z "$PIDS" ]; then
  echo "❌ 没有找到正在运行的 run_optogpt.py 进程"
  exit 0
fi

echo "🔎 找到进程: $PIDS"
# 杀掉
kill -9 $PIDS
echo "✅ 已终止 run_optogpt.py 进程"
