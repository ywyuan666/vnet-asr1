# -*- coding: utf-8 -*-
"""
u2pp_conformer_min.py  —— 教学版 U2++ Conformer（从零实现，便于理解原理）
=========================================================================
⚠️ 这是为了「看懂 U2++ 内部结构」而写的精简实现，不是用来真正训练的。
   真正训练请用 WeNet（见 run.ps1 / run.sh）。本文件可直接运行，会跑一次
   前向并打印各部分张量形状与损失，帮助你在报告/答辩里讲清楚每个模块。

U2++ 关键模块（本文件都实现了精简版）：
  1) Conformer 编码器块 = 半步前馈 + 多头自注意力 + 卷积模块 + 半步前馈
  2) CTC 分支（线性层 + CTC Loss）
  3) 双向 Transformer 解码器：左→右(L2R) + 右→左(R2L)  ← U2++ 的 “++”
  4) 联合损失 = CTC损失*α + L2R注意力损失*(1-α)*(1-β) + R2L注意力损失*(1-α)*β

运行：  python model/u2pp_conformer_min.py
"""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # 让 Windows 终端也能正常显示中文/emoji
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ----------------------------------------------------------------------
# 通用组件
# ----------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """正弦绝对位置编码（教学简化；WeNet 用的是相对位置编码 rel_pos）。"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.d_k = d_model // n_head
        self.h = n_head
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.h, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.out(out)


class ConvModule(nn.Module):
    """Conformer 卷积模块：捕捉局部特征（causal=True 时支持流式）。"""

    def __init__(self, d_model, kernel_size=15, causal=True):
        super().__init__()
        self.causal = causal
        self.pad = (kernel_size - 1) if causal else (kernel_size - 1) // 2
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, 1)  # GLU 前升维
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size,
                                   padding=0, groups=d_model)
        self.norm = nn.BatchNorm1d(d_model)
        self.pointwise2 = nn.Conv1d(d_model, d_model, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        x = x.transpose(1, 2)                 # [B, C, T]
        x = F.glu(self.pointwise1(x), dim=1)  # 门控线性单元
        x = F.pad(x, (self.pad, 0 if self.causal else self.pad))
        x = self.depthwise(x)
        x = self.act(self.norm(x))
        x = self.pointwise2(x)
        return x.transpose(1, 2)              # [B, T, C]


class ConformerBlock(nn.Module):
    """Conformer 块：½FFN → MHSA → Conv → ½FFN → LayerNorm（macaron 结构）。"""

    def __init__(self, d_model, n_head, d_ff, kernel_size=15, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForward(d_model, d_ff, dropout)
        self.norm_ff1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.conv = ConvModule(d_model, kernel_size)
        self.norm_conv = nn.LayerNorm(d_model)
        self.ff2 = FeedForward(d_model, d_ff, dropout)
        self.norm_ff2 = nn.LayerNorm(d_model)
        self.norm_final = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = x + 0.5 * self.ff1(self.norm_ff1(x))
        x = x + self.attn(self.norm_attn(x), mask)
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.ff2(self.norm_ff2(x))
        return self.norm_final(x)


# ----------------------------------------------------------------------
# 编码器：Conv2d 下采样 + N 个 Conformer 块
# ----------------------------------------------------------------------
class ConformerEncoder(nn.Module):
    def __init__(self, idim=80, d_model=144, n_head=4, d_ff=1024,
                 num_blocks=6, dropout=0.1):
        super().__init__()
        # Conv2d 前端：时间维下采样 4 倍
        self.subsample = nn.Sequential(
            nn.Conv2d(1, d_model, 3, 2), nn.ReLU(),
            nn.Conv2d(d_model, d_model, 3, 2), nn.ReLU(),
        )
        sub_dim = d_model * (((idim - 1) // 2 - 1) // 2)
        self.linear = nn.Linear(sub_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList(
            [ConformerBlock(d_model, n_head, d_ff, dropout=dropout)
             for _ in range(num_blocks)])

    def forward(self, x):
        # x: [B, T, idim]
        x = x.unsqueeze(1)                 # [B,1,T,idim]
        x = self.subsample(x)              # [B,C,T',F']
        B, C, Tt, Ff = x.shape
        x = x.transpose(1, 2).contiguous().view(B, Tt, C * Ff)
        x = self.linear(x)
        x = self.pos(x)
        for blk in self.blocks:
            x = blk(x)                     # 这里省略 padding mask（教学简化）
        return x                           # [B, T', d_model]


# ----------------------------------------------------------------------
# Transformer 解码器（单向）；双向 = 两个方向各一个
# ----------------------------------------------------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.src_attn = MultiHeadAttention(d_model, n_head, dropout)  # 简化:同结构
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.n1, self.n2, self.n3 = (nn.LayerNorm(d_model) for _ in range(3))
        # 交叉注意力这里用一个线性投影近似（教学简化）
        self.cross = nn.Linear(d_model, d_model)

    def forward(self, y, memory, tgt_mask):
        y = y + self.self_attn(self.n1(y), tgt_mask)
        # 简化的 encoder-decoder 交叉注意力：取 memory 的均值作为上下文
        ctx = self.cross(memory.mean(dim=1, keepdim=True))
        y = y + ctx
        y = y + self.ff(self.n3(y))
        return y


class TransformerDecoder(nn.Module):
    def __init__(self, vocab, d_model=144, n_head=4, d_ff=1024,
                 num_blocks=3, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_head, d_ff, dropout) for _ in range(num_blocks)])
        self.out = nn.Linear(d_model, vocab)

    def forward(self, ys, memory):
        T = ys.size(1)
        # 下三角因果 mask：第 i 个位置只能看到 <= i
        mask = torch.tril(torch.ones(T, T, device=ys.device)).bool()
        y = self.pos(self.embed(ys))
        for layer in self.layers:
            y = layer(y, memory, mask)
        return self.out(y)                 # [B, T, vocab]


# ----------------------------------------------------------------------
# U2++ 总模型：编码器 + CTC + 双向解码器
# ----------------------------------------------------------------------
class U2ppConformer(nn.Module):
    def __init__(self, vocab, idim=80, d_model=144,
                 ctc_weight=0.3, reverse_weight=0.3):
        super().__init__()
        self.encoder = ConformerEncoder(idim, d_model)
        self.ctc_fc = nn.Linear(d_model, vocab)              # CTC 分支
        self.decoder_l2r = TransformerDecoder(vocab, d_model)  # 左→右
        self.decoder_r2l = TransformerDecoder(vocab, d_model)  # 右→左 (U2++)
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight
        self.vocab = vocab

    def forward(self, feats, feat_lens, ys_in, ys_out, ys_in_r, ys_out_r):
        memory = self.encoder(feats)                  # [B, T', D]

        # ---- CTC 分支 ----
        ctc_logits = self.ctc_fc(memory).log_softmax(-1)  # [B, T', V]
        T_enc = ctc_logits.size(1)
        ctc_lens = torch.full((feats.size(0),), T_enc, dtype=torch.long)
        ys_lens = (ys_out != -1).sum(-1)
        ctc_loss = F.ctc_loss(
            ctc_logits.transpose(0, 1), ys_out.clamp(min=0),
            ctc_lens, ys_lens, blank=0, zero_infinity=True)

        # ---- 注意力解码器分支（左→右）----
        logits_l2r = self.decoder_l2r(ys_in, memory)
        att_l2r = F.cross_entropy(
            logits_l2r.reshape(-1, self.vocab), ys_out.reshape(-1),
            ignore_index=-1)

        # ---- 注意力解码器分支（右→左，U2++ 的 ++）----
        logits_r2l = self.decoder_r2l(ys_in_r, memory)
        att_r2l = F.cross_entropy(
            logits_r2l.reshape(-1, self.vocab), ys_out_r.reshape(-1),
            ignore_index=-1)

        # ---- 联合损失 ----
        att_loss = (1 - self.reverse_weight) * att_l2r + self.reverse_weight * att_r2l
        loss = self.ctc_weight * ctc_loss + (1 - self.ctc_weight) * att_loss
        return {"loss": loss, "ctc": ctc_loss, "att_l2r": att_l2r, "att_r2l": att_r2l}


# ----------------------------------------------------------------------
# 跑一次前向，验证结构正确（不是真训练）
# ----------------------------------------------------------------------
def _demo():
    torch.manual_seed(0)
    B, T, idim, V = 2, 100, 80, 30      # batch, 帧数, fbank维度, 词表大小
    L = 6                                # 文字长度
    feats = torch.randn(B, T, idim)
    feat_lens = torch.tensor([T, T])

    # 构造解码器输入/目标（sos=V-1 当起止符，-1 当 padding 忽略）
    sos = V - 1
    tokens = torch.randint(2, V - 1, (B, L))
    ys_in = torch.cat([torch.full((B, 1), sos), tokens], dim=1)
    ys_out = torch.cat([tokens, torch.full((B, 1), sos)], dim=1)
    # 右→左：把文字反过来
    tokens_r = torch.flip(tokens, dims=[1])
    ys_in_r = torch.cat([torch.full((B, 1), sos), tokens_r], dim=1)
    ys_out_r = torch.cat([tokens_r, torch.full((B, 1), sos)], dim=1)

    model = U2ppConformer(vocab=V, idim=idim)
    n_params = sum(p.numel() for p in model.parameters())
    out = model(feats, feat_lens, ys_in, ys_out, ys_in_r, ys_out_r)

    print("=" * 55)
    print("教学版 U2++ Conformer 前向测试")
    print("=" * 55)
    print(f"参数量          : {n_params/1e6:.2f} M")
    print(f"编码器输入      : {tuple(feats.shape)}  (B,T,80)")
    memory = model.encoder(feats)
    print(f"编码器输出      : {tuple(memory.shape)}  (4倍下采样)")
    print(f"CTC 损失        : {out['ctc'].item():.4f}")
    print(f"L2R 注意力损失  : {out['att_l2r'].item():.4f}")
    print(f"R2L 注意力损失  : {out['att_r2l'].item():.4f}  ← U2++ 双向解码")
    print(f"联合总损失      : {out['loss'].item():.4f}")
    print("=" * 55)
    print("结构验证通过 ✅（真正训练请用 WeNet）")


if __name__ == "__main__":
    _demo()
