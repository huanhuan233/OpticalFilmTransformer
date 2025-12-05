#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OptoGPT 逆设计训练脚本
自动打印所有训练参数到日志，并保存 JSON 备份
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.autograd import Variable
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
import datetime
import json
import os
import sys
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CUDA_VISIBLE_DEVICES=3

from core.datasets.datasets import *
from core.models.transformer import *
from core.trains.train import *


# ===========================
# 参数定义
# ===========================
def get_args():
    parser = argparse.ArgumentParser(description="OptoGPT Training Arguments")

    parser.add_argument('--seeds', default=42, type=int, help='random seeds')
    parser.add_argument('--epochs', default=2, type=int, help='Num of training epoches')
    parser.add_argument('--ratios', default=2, type=int, help='Ratio of training dataset')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--max_lr', default=1.0, type=float, help='maximum learning rate')
    parser.add_argument('--warm_steps', default=100, type=int, help='learning rate warmup steps')

    parser.add_argument('--smoothing', default=0.1, type=float, help='Smoothing for KL divergence')

    parser.add_argument('--struc_dim', default=104, type=int, help='Num of struc tokens')
    parser.add_argument('--spec_dim', default=150, type=int, help='Spec dimension')

    parser.add_argument('--layers', default=1, type=int, help='Encoder layers')
    parser.add_argument('--head_num', default=8, type=int, help='Attention head numbers')
    parser.add_argument('--d_model', default=128, type=int, help='Total attention dim = head_num * head_dim')
    parser.add_argument('--d_ff', default=128, type=int, help='Feed forward layer dim')
    parser.add_argument('--max_len', default=22, type=int, help='Transformer horizons')

    parser.add_argument('--save_folder', default='test', type=str, help='Output folder for saving model and logs')
    parser.add_argument('--save_name', default='model_inverse', type=str, help='Model save name')
    parser.add_argument('--spec_type', default='R_T', type=str, help='If predict R/T/R+T')

    parser.add_argument('--TRAIN_FILE', default='TRAIN_FILE', type=str, help='TRAIN_FILE')
    parser.add_argument('--TRAIN_SPEC_FILE', default='TRAIN_SPEC_FILE', type=str, help='TRAIN_SPEC_FILE')
    parser.add_argument('--DEV_FILE', default='DEV_FILE', type=str, help='DEV_FILE')
    parser.add_argument('--DEV_SPEC_FILE', default='DEV_SPEC_FILE', type=str, help='DEV_SPEC_FILE')
    parser.add_argument('--struc_index_dict', default={2: 'BOS'}, type=dict, help='struc_index_dict')
    parser.add_argument('--struc_word_dict', default={'BOS': 2}, type=dict, help='struc_word_dict')

    args = parser.parse_args()
    return args


# ===========================
# 参数打印与保存
# ===========================
def log_args(args):
    """打印所有参数到日志，并保存 JSON 文件"""
    arg_dict = vars(args)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("=" * 90)
    print(f"[{timestamp}] 启动 OptoGPT 训练任务 (设备: {DEVICE})")
    print("-" * 90)
    for k, v in arg_dict.items():
        print(f"{k:<20}: {v}")
    print("-" * 90)

    save_path = os.path.join(args.save_folder, args.save_name)
    print(f"保存目录: {save_path}")
    print("=" * 90)
    print("\n")

    # 保存 JSON 配置
    os.makedirs(args.save_folder, exist_ok=True)
    param_json = os.path.join(
        args.save_folder,
        f"train_config_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(param_json, "w", encoding="utf-8") as f:
        json.dump(arg_dict, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 参数配置已保存到: {param_json}\n")


# ===========================
# 主训练逻辑
# ===========================
if __name__ == '__main__':
    args = get_args()
    log_args(args)

    # 固定随机种子
    torch.manual_seed(args.seeds)
    np.random.seed(args.seeds)

    # 动态构造 save_name（方便区分不同实验）
    temp = [args.ratios, args.smoothing, args.batch_size, args.max_lr,
            args.warm_steps, args.layers, args.head_num, args.d_model, args.d_ff]
    args.save_name += '_' + args.spec_type
    args.save_name += '_S_R_B_LR_WU_L_H_D_F_' + str(temp)

    # 数据路径
    TRAIN_FILE = './dataset/Structure_train.pkl'
    TRAIN_SPEC_FILE = './dataset/Spectrum_train.pkl'
    DEV_FILE = './dataset/Structure_dev.pkl'
    DEV_SPEC_FILE = './dataset/Spectrum_dev.pkl'
    args.TRAIN_FILE, args.TRAIN_SPEC_FILE, args.DEV_FILE, args.DEV_SPEC_FILE = (
        TRAIN_FILE, TRAIN_SPEC_FILE, DEV_FILE, DEV_SPEC_FILE
    )

    # Step 1: 加载数据
    data = PrepareData(
        TRAIN_FILE, TRAIN_SPEC_FILE, args.ratios,
        DEV_FILE, DEV_SPEC_FILE, args.batch_size,
        args.spec_type, 'Inverse'
    )

    tgt_vocab = len(data.struc_word_dict)
    src_vocab = len(data.dev_spec[0])
    args.struc_dim = tgt_vocab
    args.spec_dim = src_vocab
    args.struc_index_dict = data.struc_index_dict
    args.struc_word_dict = data.struc_word_dict

    print(f"struc_vocab (结构词表大小): {tgt_vocab}")
    print(f"spec_vocab  (光谱维度): {src_vocab}")

    # Step 2: 构建模型
    model = make_model_I(
        args.spec_dim,
        args.struc_dim,
        args.layers,
        args.d_model,
        args.d_ff,
        args.head_num,
        args.dropout
    ).to(DEVICE)

    print('Model Transformer, Number of parameters {}'.format(count_params(model)))

    # Step 3: 训练模型
    print(">>>>>>> start train")
    train_start = time.time()

    criterion = LabelSmoothing(tgt_vocab, padding_idx=0, smoothing=args.smoothing)
    optimizer = NoamOpt(
        args.d_model, args.max_lr, args.warm_steps,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    )

    train_I(data, model, criterion, optimizer, args, DEVICE)

    print(f"<<<<<<< finished train, cost {time.time() - train_start:.4f} seconds")
