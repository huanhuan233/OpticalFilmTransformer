#!/usr/bin/env bash

# 工作目录
cd /data/PXY/optogpt/optogpt || exit

# 激活 conda 环境
source /home/sysadmin/anaconda3/etc/profile.d/conda.sh
conda activate /home/sysadmin/WorkSpace/PXY/optogpt_env

# 日志目录和文件
LOG_DIR="$(dirname "$(pwd)")/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/train_$(date +%F_%H%M).log"

# 后台执行（使用 -u 参数禁用输出缓冲，实时写入日志）
# 设置使用 GPU 3
CUDA_VISIBLE_DEVICES=3 nohup python -u run_optogpt.py \
  --seeds 42 \
  --epochs 200 \
  --ratios 2 \
  --batch_size 1024 \
  --dropout 0.1 \
  --max_lr 4e-4 \
  --warm_steps 2000 \
  --smoothing 0.05 \
  \
  --struc_dim 91 \
  --spec_dim 150 \
  --layers 6 \
  --head_num 8 \
  --d_model 512 \
  --d_ff 2048 \
  --max_len 22 \
  \
  --save_name model_inverse \
  --save_folder test \
  > "$LOG" 2>&1 &
echo "训练已启动，日志保存在 $LOG"