# Build models
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

# Transformer 需要堆叠多层 Encoder/Decoder block，每一层结构相同但参数不同。
# 复制一个子模块 N 份，保证每份权重独立。
# copy.deepcopy 确保不会共享参数
def clones(module, N):
    """
    "Produce N identical layers."
    Use deepcopy the weight are indenpendent.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 把材料+厚度（结构 token）变成向量，类似自然语言变成向量
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__() 
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # return x's embedding vector（times math.sqrt(d_model)）
        return self.lut(x) * math.sqrt(self.d_model)

# 附件位置坐标向量，让模型知道是第几层
# 用 正弦/余弦函数生成位置向量。不同维度的正弦/余弦有不同频率，保证了相对位置关系能被线性表示。
class PositionalEncoding(nn.Module):
    # 定义位置编码层
    # d_model：token 向量维度
    # dropout：正则化，防止过拟合。
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 创建 [max_len, d_model] 的矩阵，存储所有可能位置的向量。                           
        pe = torch.zeros(max_len, d_model)
        # 生成位置 [0, 1, 2, ..., max_len-1]，形状 [max_len, 1]。
        position = torch.arange(0., max_len).unsqueeze(1)
        # 生成不同频率的缩放系数。
        # 前面维度变化快（高频），后面维度变化慢（低频）。
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        # 位置 × 频率 → 得到角度矩阵 [max_len, d_model/2]。
        pe_pos   = torch.mul(position, div_term)
        # 偶数维 = sin，奇数维 = cos。 保证每个位置的向量唯一
        pe[:, 0::2] = torch.sin(pe_pos)
        pe[:, 1::2] = torch.cos(pe_pos)
        # 变成 [1, max_len, d_model]，方便和输入 [B, L, d_model] 相加。
        # [B, L, d_model] 
        # 在原始 Transformer 论文和 PyTorch 实现里，经过 Embedding 之后，输入张量的形状是：
        # B = Batch size（一次训练的样本个数）
        # L = 序列长度（例如 NLP 中的句子长度；在薄膜里就是膜层数）
        # d_model = 每个 token 的 embedding 维度（比如 512）
        # register_buffer：作为常量保存，不更新参数。
        pe = pe.unsqueeze(0)                                   
        self.register_buffer('pe', pe) # pe

    #对齐后把位置编码加上
    def forward(self, x):
        #  build pe w.r.t to the max_length
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
# 缩放点积注意力实现
# mask 在光学薄膜里的作用：Padding mask（如果某个膜系只有 12 层，剩下 8 层是空的 PAD，就必须 mask） 
# Causal mask：在训练逐层生成结构时，用因果 mask，保证模型设计第 n 层时，只参考前面已经生成的层，而不会看到未来层。
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1) 
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

# 论文中头数为8，头2和头4的注意力图更聚焦于上下相邻层，而头1则更关注长期交替关系
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        # h : number of head
        assert d_model % h == 0 # check the h number
        self.d_k = d_model // h
        self.h = h
        # 4 linear layers: WQ WK WV and final linear mapping WO，用来生成QKV拼接结果
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # 一共四个独立的线性层，分别是Q，K，V最后一个是多头拼接之后，把结果再投影回dmodel维，做一次融合
        #权重存储，方便可视化
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
# 用三组线性层把输入投影成 Q/K/V，并切分成 h 个头；
# 调用前面写好的 scaled dot-product attention，各头并行算注意力；
# 把各头结果 拼接回 d_model 维，再过一个线性层得到最终输出。
    def forward(self, query, key, value, mask=None):
        # apply the multi-head using quick method
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0) # get batch size，决定一次训练多少个样本
        # mask 在“头”这个维度上扩 1 维，便于广播给每个头。
        # padding mask：原本 [B, 1, L, L] 或 [B, L, L]，unsqueeze(1) 后可广播到 [B, H, L, L]；
        # causal mask：通常 [1, 1, L, L]，同理广播到所有 batch、所有头。
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        # parttion into h sections，switch 2,3 axis for computation. 
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) 
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # contiguous: when use transpose, PyTorch does not create a new tensor and just changes the meta data
        # use contiguous to make a copy of the tensor with transpose data 

        return self.linears[-1](x) # final linear layer
# LayerNorm（层归一化）: 对每个样本的每个 token 向量在最后一维上做归一化（减均值、除方差）,再学习两个参数
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__() # 初始化父类
        self.a_2 = nn.Parameter(torch.ones(features)) # γ：可学习的缩放参数，初值全1，形状[d_model]
        self.b_2 = nn.Parameter(torch.zeros(features))# β：可学习的平移参数，初值全0，形状[d_model]
        self.eps = eps  # 数值稳定用的微小常数，防止除0

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # 在最后一维(d_model)上求均值，保持维度 [B, L, 1]
        std = x.std(-1, keepdim=True)  # 在最后一维上求标准差，形状 [B, L, 1]
        x_zscore = (x - mean)/ torch.sqrt(std ** 2 + self.eps)  # 标准化：减均值除标准差
        return self.a_2*x_zscore+self.b_2  # 线性缩放+平移：γ⊙x̂ + β，形状仍 [B, L, d_model]
# 用来链接原始输入+子层输出
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    SublayerConnection: connect Multi-Head Attention and Feed Forward Layers 
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)   # 层归一化模块，size=d_model
        self.dropout = nn.Dropout(dropout) # dropout 正则
     # 子层计算：sublayer(LN(x))
     # 对子层输出做 dropout
     # 残差连接：原输入 x 与子层输出相加
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
# fc1 → norm → fc2
# 用来转化维度
class FullyConnectedLayers(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(FullyConnectedLayers, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim) # 线性变换，保持维度不变（y=wx+b)，在保持维度不变的情况下重新分配权重，让输入的各个维度产生新的相关性。
        self.fc2 = nn.Linear(input_dim, out_dim) # 线性变换，映射到输出维度
        self.norm = LayerNorm(input_dim)# 层归一化，更稳定
    
    def forward(self, x):
        return self.fc2(self.norm(self.fc1(x)))# ：“输入 → 线性重分配 → 归一化 → 投影到输出维度” 的结果张量。

# 在每个位置上用一个小 MLP 对特征做非线性变换，扩展再压缩，从而增强模型的表达能力。
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff) # w_1：把维度从 d_model 投影到更大的 d_ff
        self.w_2 = nn.Linear(d_ff, d_model) # w_2：再把扩展后的特征压回 d_model 维度。
        self.dropout = nn.Dropout(dropout) # 训练时随机丢弃一部分神经元，防止过拟合。

    def forward(self, x):
        h1 = self.w_1(x) # 线性扩展 d_model → d_ff （标准transformer应该是h1 = F.relu(self.w_1(x))，少了激活函数，表达能力会弱)
        h2 = self.dropout(h1) # 丢弃一些神经元
        return self.w_2(h2)# 投影回 d_model

class Encoder(nn.Module):
    "Core encoder is a stack of N layers (blocks)"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N) #复制 N 个完全相同的 编码器层 (EncoderLayer)，每个层里都有 Multi-Head Attention + FFN。
        self.norm = LayerNorm(layer.size) # 归一化一下，稳定

    def forward(self, x, mask):
        """
        Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout): 
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn # 传进来的 MultiHeadedAttention 模块。
        self.feed_forward = feed_forward # 传进来的 PositionwiseFeedForward 模块。
        self.sublayer = clones(SublayerConnection(size, dropout), 2) #复制了两个 SublayerConnection（残差连接+LayerNorm），第一个用于 Attention 子层，第二个用于 FeedForward 子层
        self.size = size # d_model

    def forward(self, x, mask):
        # X-embedding to Multi-head-Attention
        # 先经过自注意力子层
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # why use lambda? 
        # X-embedding to feed-forwad nn
        # 再经过前馈子层
        return self.sublayer[1](x, self.feed_forward)
# 负责拼装头尾
class Transformer(nn.Module):
    def __init__(self, encoder, fc, src_embed):
        super(Transformer, self).__init__()
        self.encoder = encoder  # 编码器
        self.fc = fc # 最后的全连接层 (分类/预测用)
        self.src_embed = src_embed # 输入嵌入 (embedding + 位置编码)
    # src_embed(src) → 把输入 token（比如光谱点、材料 ID）变成向量表示。 encoder(...) → 把嵌入序列送入 N 层 Encoder，得到上下文编码表示。
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def forward(self, src, src_mask):
        "Take in and process masked src and target sequences."
        # encoder output will be the decoder's memory for decoding
        en = self.encode(src, src_mask) # 得到编码结果
        en = en[:, 0,:] # 只取第一个 token 的向量
        return self.fc(en)  # 投影到最终输出

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h = 8, dropout=0.1):

    # d_model: dimension of Query, Key, Value
    # d_ff: neurons for FeedForward layer
    # h: num of head attention
    # N: number of transformer stacks. 

    c = copy.deepcopy
    #  Attention 
    attn = MultiHeadedAttention(h, d_model)
    #  FeedForward 
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    #  Positional Encoding
    position = PositionalEncoding(d_model, dropout)
    # Fully connected layers
    fc = FullyConnectedLayers(d_model, tgt_vocab)
    #  Transformer 
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        fc, 
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    # Paper title: Understanding the difficulty of training deep feedforward neural networks Xavier
    #参数初始化↓，
    for p in model.parameters():# 遍历模型里所有的参数张量
        if p.dim() > 1: # 只对维度 >1 的张量做初始化（权重矩阵）
            nn.init.xavier_uniform_(p) # 用 Xavier (Glorot) 均匀分布初始化，根据信号进出通道数（fan-in、fan-out）自动调节初始权重范围，让前向传播和反向传播时，方差保持稳定 避免梯度爆炸或消失。
    return model # 返回这个组装好且初始化过的模型


class Decoder(nn.Module):
    def __init__(self, layer, N):
        "Generic N layer decoder with masking."
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)# 把 DecoderLayer（包含 masked self-attn + cross-attn + feedforward） 复制 N 份，形成一个堆叠的解码器        
        self.norm = LayerNorm(layer.size) #正则化，稳定用

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Repeat decoder N times
        Decoderlayer get a input attention mask (src) 
        and a output attention mask (tgt) + subsequent mask 
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask) # 逐层执行 DecoderLayer
        return self.norm(x) # 出来后再做一次归一化


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn  # 多头注意力，用作 Masked Self-Attn
        self.src_attn = src_attn  # 多头注意力，用作 Cross-Attn（对 memory）
        self.feed_forward = feed_forward # 逐位置前馈网络 FFN
        self.sublayer = clones(SublayerConnection(size, dropout), 3) # 复制 3 个 SublayerConnection（Pre-Norm 残差块）：分别包住 ①自注意 ②交叉注意 ③FFN

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory # encoder output embedding
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)) # 只能参考以前的，用subsequent_mask把后面的mask掉
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask)) # 把当前生成进度跟目标光谱对齐，看看光谱里哪些部分和当前层设计最相关。
        # Context-Attention：q=decoder hidden，k,v from encoder hidden 
        return self.sublayer[2](x, self.feed_forward)

#  把该挡上的挡上用
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Transformer_I(nn.Module):
    def __init__(self, fc, decoder, tgt_embed, generator):
        super(Transformer_I, self).__init__()
        self.fc = fc
        self.decoder = decoder
        # self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator 

    # def encode(self, src, src_mask):
    #     return self.encoder(self.src_embed(src), src_mask)
    # memory
    # 来自源端（src）的 embedding。
    # 就是把输入光谱（142 维左右）经过 fc 投影到 d_model 得到的向量。
    # 形状 [B, T_src, d_model]，通常 T_src=1（因为光谱整体被当作一个“单步特征”）。
    # src_mask
    # 源端的掩码。
    # 如果光谱是定长、没有 padding → 全 1。
    # 如果光谱序列化（比如分波长点输入） → 用 mask 屏蔽掉补齐的 <PAD> 波长。
    # 形状 [B, 1, 1, T_src] 或 [B, 1, T_tgt, T_src]。
    # tgt
    # 目标序列 token（材料+厚度）。
    # 训练时是“已知的真实序列”做 teacher forcing，形状 [B, T_tgt]。
    # 推理时是“已经生成的部分序列”，不断扩展。
    # tgt_mask
    # 目标掩码 = padding mask × subsequent mask。
    # 确保 只能看前面，不能偷看未来层，同时忽略 <PAD>。
    # 形状 [B, 1, T_tgt, T_tgt]。
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    # src
    # 源输入。
    # 在逆向任务里就是 光谱向量，形状 [B, T_src, src_dim]，比如 [32, 1, 142]。
    # tgt
    # 目标 token 序列。
    # [B, T_tgt]，每个元素是一个整数 id（对应词表里的材料+厚度）。
    # 例如 [<BOS>, TiO2_100, SiO2_50, …, <EOS>, <PAD>, …]。
    # src_mask
    # 对光谱的 mask。
    # 如果光谱输入是 [B, 1, src_dim]（单向量），可以直接给一个全 1 的 mask。
    # 如果是 [B, seq_len, features]，就需要把 PAD 波长遮掉。
    # tgt_mask
    # 目标掩码，保证自回归生成。
    # 例如当生成第 5 层时，只能看 <BOS>, 层1, 层2, 层3, 层4。
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        # encoder output will be the decoder's memory for decoding
        return self.decode(self.fc(src), src_mask, tgt, tgt_mask)
# 为什么decoder有这步：因为输出的是离散 token 序列，输出的是这里选择token的概率分布
class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # decode: d_model to vocab mapping
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x): 
        return F.log_softmax(self.proj(x), dim=-1)  ##???

def make_model_I(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h = 8, dropout=0.1):

    # src_vocab: dim of spectrum 
    # tgt_vocab: list of structures

    # d_model: dimension of Query, Key, Value
    # d_ff: neurons for FeedForward layer
    # h: num of head attention
    # N: number of transformer stacks. 

    c = copy.deepcopy
    #  Attention 
    attn = MultiHeadedAttention(h, d_model)
    #  FeedForward 
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    #  Positional Encoding
    position = PositionalEncoding(d_model, dropout)
    # Fully connected layers
    fc = FullyConnectedLayers(src_vocab, d_model)
    #  Transformer 
    model = Transformer_I(
        fc,
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    # Paper title: Understanding the difficulty of training deep feedforward neural networks Xavier
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
