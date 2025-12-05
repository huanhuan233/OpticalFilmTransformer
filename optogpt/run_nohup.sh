#!/usr/bin/env bash

# 工作目录
cd /home/sysadmin/WorkSpace/PXY/optogpt/optogpt || exit

# 激活 conda 环境
source /home/sysadmin/anaconda3/etc/profile.d/conda.sh
conda activate /home/sysadmin/WorkSpace/PXY/optogpt_env

# 日志文件
LOG="train_$(date +%F_%H%M).log"

# 后台执行
nohup python run_optogpt.py \
  --epochs 200 \
  --batch_size 512 \
  --dropout 0.1 \
  --max_lr 2e-4 \
  --warm_steps 3000 \
  --layers 6 \
  --head_num 8 \
  --d_model 512 \
  --d_ff 2048 \
  --max_len 22 \
  > "$LOG" 2>&1 &
echo "训练已启动，日志保存在 $LOG"