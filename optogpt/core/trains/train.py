import os
import math
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk import word_tokenize
from collections import Counter
from torch.autograd import Variable
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


# 正则化作用，提高一下输出膜系的泛化能力，避免过于死板（有可能有其他膜系搭配）
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum') # 2020 update
        self.padding_idx = padding_idx # PAD 的类别 id（PAD 不参与损失）
        self.confidence = 1.0 - smoothing # 给真实类别的概率质量
        self.smoothing = smoothing # 平滑强度 ε
        self.size = size # 类别数（= 目标词表大小 vocab_size）
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size # x 形状通常是 [batch*tokens, vocab]
        true_dist = x.data.clone() # 克隆一份张量当模板，用均匀值填充。
        true_dist.fill_(self.smoothing / (self.size - 2)) # 分母是 size-2 的原因：后面要单独给真类分配 confidence，并且把 PAD 类概率置零，所以“可均匀分配的其它类数目 = 全部 − 真类 − PAD = size−2”
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) # 用 scatter_ 在每行（每个样本）把“真实类别”的位置改成 confidence（1−ε）
        true_dist[:, self.padding_idx] = 0 # 把 PAD 类的概率整体置 0（PAD 从不作为学习目标被奖励/惩罚）
        # 如果当前样本本身就是 PAD（比如序列 padding 出来的那些位置），那么整行目标分布都置 0：
        # 含义：这些位置不计入损失（因为 KL(p‖q) 中 p 全 0，贡献为 0）。
        # 这一步与上面的把 PAD 类列清零是两层语义：
        # 行层面：样本位点是 PAD → 整行 0，彻底跳过该 token；
        # 列层面：非 PAD 样本的目标分布里，PAD 类概率恒为 0。
        mask = torch.nonzero(target.data == self.padding_idx) 
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        # 存下来方便可视化/调试    
        self.true_dist = true_dist
        # 计算 KLDivLoss：x 是 log-prob，true_dist 是概率。
        # requires_grad=False：目标分布不参与反传。
        # reduction='sum'，所以外层训练循环一般会用 ntokens 去做归一（会在 SimpleLossCompute(...)/run_epoch_I 看到 loss / norm 的写法），再把分块的 sum 累起来统计。
        return self.criterion(x, Variable(true_dist, requires_grad=False))
# 训练loss function的工厂函数
class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator # tranformer.py中的generator类
        self.criterion = criterion # 计算 KLDivLoss，前面LabelSmoothing
        self.opt = opt # 封装好的优化器
        
    def __call__(self, x, y, norm):
        x = self.generator(x) # 预测：调用 Generator 得到 log-prob
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm # 算损失：用 LabelSmoothing + KLDivLoss；
        loss.backward() # 反向传播，算梯度。
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad() #更新：梯度下降；
        return loss.data.item() * norm.float() # 返回标量 loss：方便日志和统计

# We used factor=2, warmup-step = 4000
# 优化器，用于优化全体可训练参数
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer # 这里是 Adam
        self._step = 0  # 迭代步数
        self.warmup = warmup # warmup 步数
        self.factor = factor # 缩放因子
        self.model_size = model_size # d_model
        self._rate = 0 # 当前学习率
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate # 动态修改 Adam 的 lr
        self._rate = rate
        self.optimizer.step() # 调用 Adam 更新参数
        
    def rate(self, step = None): # transformer 学习率
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

# 训练 / 验证单个 epoch 的循环
def run_epoch(data, model, criterion, optimizer, epoch, DEVICE):
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.
    for i , batch in enumerate(data):
        out = model(batch.src.to(DEVICE),  batch.src_mask.to(DEVICE))
        # print(out.size(), batch.trg.size())
        loss = criterion(out, batch.trg.to(DEVICE))

        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.optimizer.zero_grad()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch {:d} Batch: {:d} Loss: {:.4f} Tokens per Sec: {:.2f}s".format(epoch, i - 1, loss, (tokens.float() / elapsed)))
            start = time.time()
            tokens = 0
        del out, loss
    print(total_loss, i)
    return total_loss/i
# 统计模型参数量
def count_params(model):

    return sum([np.prod(layer.size()) for layer in model.parameters() if layer.requires_grad])

def save_checkpoint(model, optimizer, epoch, loss_all, path, configs):
    # save the saved file 
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer,
            'loss_all':loss_all,
            'configs':configs,
            # 'seed':seed,
        }, path)

# 正向 transformer driver function
def train(data, model, criterion, optimizer, configs, DEVICE):
    """
    Train and Save the model.
    """
    # init loss as a large value
    best_dev_loss = 1e5
    loss_all = {'train_loss':[], 'dev_loss':[]}

    save_folder = configs.save_folder
    save_name = configs.save_name
    EPOCHS = configs.epochs

    for epoch in range(EPOCHS):
        # Train model 
        model.train()
        train_loss = run_epoch(data.train_data, model, criterion, optimizer, epoch, DEVICE)

        # validate model on dev dataset

        model.eval()
        print('>>>>> Evaluate')
        with torch.no_grad():
            dev_loss = run_epoch(data.dev_data, model, criterion, None, epoch, DEVICE)
        print('<<<<< Evaluate loss: {:.8f}'.format(dev_loss))
        loss_all['train_loss'].append(train_loss.detach())
        loss_all['dev_loss'].append(dev_loss.detach())

        # save the model with best-dev-loss
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            save_checkpoint(model, optimizer, epoch, loss_all, 'saved_models/ol_transformer/'+save_folder+'/'+save_name+'_best.pt',  configs)
            print('Saved')
        if epoch%2 == 1:
            best_dev_loss = dev_loss
            save_checkpoint(model, optimizer, epoch, loss_all, 'saved_models/ol_transformer/'+save_folder+'/'+save_name+'_recent.pt',  configs)
            
        print(f">>>>> current best loss: ", best_dev_loss)

# data
# 一个 batch 迭代器（训练集或验证集）。
# 每个 batch 包含 src, trg, src_mask, trg_mask, trg_y, ntokens。
# model
# 这里是 Transformer_I（inverse 模型，decoder-only）。
# 它定义了 forward(src, tgt, src_mask, tgt_mask)。
# loss_compute
# 封装的损失计算器（SimpleLossCompute）。
# 内部包含 Generator（把 decoder hidden → vocab logits）、
# criterion（LabelSmoothing+KLDivLoss）、optimizer。
# epoch
# 当前第几轮训练，用于日志打印。
# DEVICE
# 设备（cuda / cpu）。
def run_epoch_I(data, model, loss_compute, epoch, DEVICE):
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.
    for i , batch in enumerate(data):
        # batch.src
        # 输入光谱（目标 R/T 曲线），张量 [batch_size, spec_dim]。
        # 是 inverse 任务的 条件输入。
        # batch.trg
        # 结构序列的 输入部分（teacher forcing）。
        # 形状 [batch_size, seq_len]，例如 [BOS, token1, token2, ...]。
        # batch.src_mask
        # 光谱输入的掩码，通常是 [batch_size, 1, 1, spec_dim]。
        # 这里大概率全 1，因为光谱是稠密向量。
        # batch.trg_mask
        # 目标序列 mask = padding mask × subsequent mask。
        # 保证自回归：预测第 j 个 token 时，看不到 j 之后的 token。
        # batch.trg_y
        # 结构序列的 目标部分（labels）。
        # 形状 [batch_size, seq_len]，与 trg 对齐但右移一位：
        # trg = [BOS, t1, t2, t3]
        # trg_y = [t1, t2, t3, EOS]
        # batch.ntokens
        # 有效 token 数（不含 PAD）。
        # 用来做 loss 归一化，避免不同长度 batch 损失不可比。
        out = model(batch.src.to(DEVICE), batch.trg.to(DEVICE), batch.src_mask, batch.trg_mask.to(DEVICE))
        loss = loss_compute(out, batch.trg_y.to(DEVICE), batch.ntokens.to(DEVICE))
        # out
        # 模型输出的预测分布（未过 Generator 的 decoder hidden）。
        # 形状 [batch_size, seq_len, d_model]。
        # 会被 loss_compute 送进 Generator → log_softmax → KL loss。
        # loss
        # 单个 batch 的损失标量。
        # 已经做了归一化（除以 ntokens）。
        # total_loss
        # 累计所有 batch 的损失和（最终会除以总 token 得平均 loss）。
        # total_tokens
        # 累计所有 batch 的 token 数。
        # tokens
        # 统计 50 个 batch 内的 token 数，用于吞吐率计算。
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch {:d} Batch: {:d} Loss: {:.4f} Tokens per Sec: {:.2f}s".format(epoch, i - 1, loss / batch.ntokens, (tokens.float() / elapsed )))
            start = time.time()
            tokens = 0
        del out, loss

    return total_loss / total_tokens

    
def train_I(data, model, criterion, optimizer, configs, DEVICE):
    """
    Train and Save the model.
    """
    # init loss as a large value
    best_dev_loss = 1e5
    loss_all = {'train_loss':[], 'dev_loss':[]}

    save_folder = configs.save_folder
    save_name = configs.save_name
    EPOCHS = configs.epochs

    for epoch in range(EPOCHS):
        # Train model 
        model.train()
        train_loss = run_epoch_I(data.train_data, model, SimpleLossCompute(model.generator, criterion, optimizer), epoch, DEVICE)
        model.eval()

        # validate model on dev dataset
        print('>>>>> Evaluate')
        dev_loss = run_epoch_I(data.dev_data, model, SimpleLossCompute(model.generator, criterion, None), epoch, DEVICE)
        print('<<<<< Evaluate loss: {:.2f}'.format(dev_loss))
        loss_all['train_loss'].append(train_loss.detach())
        loss_all['dev_loss'].append(dev_loss.detach())
        
        # save the model with best-dev-loss

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            save_checkpoint(model, optimizer, epoch, loss_all, 'saved_models/optogpt/'+save_folder+'/'+save_name+'_best03.pt',  configs)

        save_checkpoint(model, optimizer, epoch, loss_all, 'saved_models/optogpt/'+save_folder+'/'+save_name+'_recent03.pt',  configs)
            
        print(f">>>>> current best loss: {best_dev_loss}")
        