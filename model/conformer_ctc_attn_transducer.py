# -*- coding: utf-8 -*-
"""
conformer_ctc_attn_transducer.py
================================
核心模型：Conformer 编码器 + CTC / Attention / Transducer 三任务联合训练。

架构设计：
  Fbank → Conv2d 下采样(4x) → N×ConformerBlock → Encoder Memory
      ├── CTC Linear → CTC Logits (CTC Loss)
      ├── Transformer Decoder(cross-attn) → Attn Logits (CE Loss)
      └── Transducer Prediction Net(RNN) → Joint Net → RNNT Logits (RNN-T Loss)

总损失 = ctc_weight * CTC_loss + attn_weight * Attention_loss + trans_weight * Transducer_loss
"""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# ======================================================================
# 通用组件
# ======================================================================

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class FeedForward(nn.Module):
    """Macaron 风格的前馈网络：Linear → Swish → Dropout → Linear → Dropout"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    """自注意力，支持流式 KV 缓存"""

    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.d_k = d_model // n_head
        self.h = n_head
        self.w_qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, attn_cache=None):
        """
        x: (B, T, D)
        mask: optional attention mask (B, 1, T, T) or (T, T)
        attn_cache: optional (cached_k, cached_v) from previous chunks
        Returns: output (B, T, D) or (output, (new_k, new_v)) if attn_cache is not None
        """
        B, T, D = x.shape
        q, k, v = self.w_qkv(x).chunk(3, dim=-1)

        if attn_cache is not None:
            cached_k, cached_v = attn_cache
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)

        q = q.view(B, T, self.h, self.d_k).transpose(1, 2)
        k_all = k.view(B, -1, self.h, self.d_k).transpose(1, 2)
        v_all = v.view(B, -1, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k_all.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v_all).transpose(1, 2).contiguous().view(B, T, D)
        out = self.out(out)

        if attn_cache is not None:
            # Return updated cache (full k, v including cached + current)
            return out, (k, v)
        return out


class MultiHeadCrossAttention(nn.Module):
    """交叉注意力：query 来自解码器，key/value 来自编码器"""

    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.d_k = d_model // n_head
        self.h = n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        Bq, Tq, D = query.shape
        Bk, Tk, _ = key.shape
        q = self.w_q(query).view(Bq, Tq, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(Bk, Tk, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(Bk, Tk, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(Bq, Tq, D)
        return self.out(out)


class ConvModule(nn.Module):
    """Conformer 卷积模块：Pointwise → GLU → Depthwise Conv → BN → Swish → Pointwise"""

    def __init__(self, d_model, kernel_size=15, dropout=0.1):
        super().__init__()
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size,
                                   padding=(kernel_size - 1) // 2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.pointwise2 = nn.Conv1d(d_model, d_model, 1)
        self.act = Swish()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.glu(self.pointwise1(x), dim=1)
        x = self.depthwise(x)
        x = self.act(self.bn(x))
        x = self.pointwise2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class ConformerBlock(nn.Module):
    """
    Conformer 块：½ FFN → MHSA → Conv → ½ FFN → LayerNorm
    三明治（Macaron）结构
    """

    def __init__(self, d_model, n_head, d_ff, kernel_size=15, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForward(d_model, d_ff, dropout)
        self.norm_ff1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_head, dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.conv = ConvModule(d_model, kernel_size, dropout)
        self.norm_conv = nn.LayerNorm(d_model)
        self.ff2 = FeedForward(d_model, d_ff, dropout)
        self.norm_ff2 = nn.LayerNorm(d_model)
        self.norm_final = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, attn_mask=None, attn_cache=None):
        """
        x: (B, T, D)
        mask: optional padding mask
        attn_mask: optional chunk/causal attention mask (overrides mask for self_attn)
        attn_cache: optional (cached_k, cached_v) for streaming inference
        Returns: (B, T, D) or (output, new_attn_cache) if attn_cache is not None
        """
        x = x + 0.5 * self.ff1(self.norm_ff1(x))

        attn_input = self.norm_attn(x)
        effective_mask = attn_mask if attn_mask is not None else mask
        # Convert 3D (B, 1, T) padding mask → 4D (B, 1, T, T) attention mask
        if effective_mask is not None and effective_mask.dim() == 3:
            # effective_mask has True = padding; for attention we want True = keep
            keep = (effective_mask == 0).float()  # (B, 1, T), 1 where valid
            effective_mask = keep.unsqueeze(-1) * keep.unsqueeze(-2)  # (B, 1, T, T)
        if attn_cache is not None:
            attn_out, new_attn_cache = self.attn(attn_input, effective_mask, attn_cache)
        else:
            attn_out = self.attn(attn_input, effective_mask)
            new_attn_cache = None
        x = x + attn_out

        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.ff2(self.norm_ff2(x))
        out = self.norm_final(x)

        if attn_cache is not None:
            return out, new_attn_cache
        return out


# ======================================================================
# 编码器：Conv2d 下采样 + N×ConformerBlock
# ======================================================================

class ConformerEncoder(nn.Module):
    """
    Conformer 编码器
    - 前端：2 层 Conv2d 时间维下采样 4 倍
    - 主体：N 个 ConformerBlock
    - 支持流式：chunk_size > 0 时启用
    """

    def __init__(self, idim=80, d_model=144, n_head=4, d_ff=1024,
                 num_blocks=6, dropout=0.1, kernel_size=15,
                 chunk_size=0, right_context=0, streaming_prob=0.5):
        super().__init__()
        # Conv2d 前端：下采样 4x
        self.subsample = nn.Sequential(
            nn.Conv2d(1, d_model, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, 3, 2, padding=1),
            nn.ReLU(),
        )
        sub_dim = d_model * (idim // 4)
        self.linear = nn.Linear(sub_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, n_head, d_ff, kernel_size, dropout)
            for _ in range(num_blocks)
        ])

        # Streaming 参数
        self.chunk_size = chunk_size       # 0 = non-streaming
        self.right_context = right_context
        self.streaming_prob = streaming_prob

    def forward(self, x, mask=None, is_streaming=None, chunk_size=None, right_context=None):
        """
        x: [B, T, idim]  Fbank 特征
        mask: optional padding mask
        is_streaming: if True, use chunk mask for streaming training
        chunk_size: override default chunk size
        right_context: override default right context
        返回 [B, T', d_model]  编码器输出
        """
        x = x.unsqueeze(1)                     # [B, 1, T, idim]
        x = self.subsample(x)                  # [B, C, T/4, idim/4]
        B, C, Tt, Ff = x.shape
        x = x.transpose(1, 2).contiguous().view(B, Tt, C * Ff)
        x = self.linear(x)                     # [B, T/4, d_model]
        x = self.pos_enc(x)

        # 流式训练：使用 chunk mask 限制注意力范围
        if is_streaming:
            cs = chunk_size if chunk_size is not None else self.chunk_size
            rc = right_context if right_context is not None else self.right_context
            if cs > 0:
                # 从 streaming_helper 导入
                from .streaming_helper import make_chunk_mask
                attn_mask = make_chunk_mask(
                    Tt, cs, rc, dtype=x.dtype, device=x.device
                )
                for blk in self.blocks:
                    x = blk(x, mask, attn_mask=attn_mask)
                return x

        # 非流式（标准 forward）
        for blk in self.blocks:
            x = blk(x, mask)
        return x  # [B, T', d_model]

    @torch.no_grad()
    def forward_chunk(
        self,
        x: torch.Tensor,
        offset: int = 0,
        required_cache_size: int = -1,
        attn_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = None,
        cnn_cache: List[Optional[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor,
               List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
               List[Optional[torch.Tensor]]]:
        """
        流式推理：逐 chunk 处理，带 KV 缓存。

        Args:
            x: Encoder input for this chunk (B, T_chunk, idim)
            offset: Global frame offset of this chunk (after subsampling)
            required_cache_size: How many past frames to keep in cache (-1 = all)
            attn_cache: List of (key, value) tensors for each layer's self-attention
            cnn_cache: List of cached conv outputs for each layer

        Returns:
            output: (B, T_chunk_out, d_model)
            new_attn_cache: Updated attention caches
            new_cnn_cache: Updated conv caches
        """
        # Subsample
        x = x.unsqueeze(1)                     # [B, 1, T, idim]
        x = self.subsample(x)                  # [B, C, T/4, idim/4]
        B, C, Tt, Ff = x.shape
        x = x.transpose(1, 2).contiguous().view(B, Tt, C * Ff)
        x = self.linear(x)                     # [B, Tt, d_model]

        # Positional encoding (offset-aware for global position)
        x = x + self.pos_enc.pe[:, offset:offset + Tt]

        new_attn_cache = []
        new_cnn_cache = []

        for i, blk in enumerate(self.blocks):
            if attn_cache is not None and i < len(attn_cache) and attn_cache[i] is not None:
                x, new_cache = blk(x, attn_cache=attn_cache[i])
                new_attn_cache.append(new_cache)
            else:
                x = blk(x)
                new_attn_cache.append(None)

            # CNN cache (placeholder — no state kept in current conv)
            if cnn_cache is not None and i < len(cnn_cache) and cnn_cache[i] is not None:
                pass  # ConvModule currently uses symmetric padding, no causal cache
            new_cnn_cache.append(None)

        return x, new_attn_cache, new_cnn_cache


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ======================================================================
# Attention Decoder：标准 Transformer 解码器
# ======================================================================

class AttentionDecoder(nn.Module):
    """
    Transformer 解码器（自回归）
    - 带 causal mask 的自注意力
    - 交叉注意力连接编码器输出
    """

    def __init__(self, vocab_size, d_model=144, n_head=4, d_ff=1024,
                 num_blocks=3, dropout=0.1, sos_id=-1, eos_id=-1):
        super().__init__()
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, ys, memory, ys_mask=None):
        """
        ys: [B, L]  目标token序列（含 <sos>）
        memory: [B, T', d_model]  编码器输出
        """
        B, L = ys.shape
        y = self.pos(self.embed(ys))
        # causal mask
        causal_mask = torch.tril(torch.ones(L, L, device=ys.device)).bool()
        for layer in self.layers:
            y = layer(y, memory, causal_mask)
        y = self.norm(y)
        logits = self.out(y)
        return logits  # [B, L, vocab]


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, n_head, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadCrossAttention(d_model, n_head, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, memory, causal_mask):
        # self-attention with causal mask
        y = y + self.dropout(self.self_attn(self.norm1(y), causal_mask))
        # cross-attention: y -> query, memory -> key/value (no causal mask needed)
        y = y + self.dropout(self.cross_attn(self.norm2(y), memory, memory))
        # FFN
        y = y + self.dropout(self.ff(self.norm3(y)))
        return y


# ======================================================================
# Transducer Decoder（RNN-T）
# ======================================================================

class TransducerDecoder(nn.Module):
    """
    RNN-T 解码器（Recurrent Neural Network Transducer）
    - Prediction Network: embedding + LSTM
    - Joint Network: 融合编码器输出和预测网络输出
    """

    def __init__(self, vocab_size, d_model=144, hidden_dim=144,
                 embed_dim=144, num_layers=1, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pred_rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        # Joint Network: 融合 encoder_out 和 pred_out
        self.joint_linear = nn.Linear(d_model + hidden_dim, vocab_size)

    def forward(self, encoder_out, targets, target_lengths):
        """
        encoder_out: [B, T, d_model]
        targets: [B, U]  含 <sos> 前缀 (不含 blank/eos)
        target_lengths: [B]  实际长度
        """
        B, T, D = encoder_out.shape
        U = targets.shape[1]

        # Prediction Network
        pred_emb = self.embed(targets)          # [B, U, embed_dim]
        pred_out, _ = self.pred_rnn(pred_emb)   # [B, U, hidden_dim]

        # Joint Network: 拼接编码器输出和预测网络输出
        # enc_expanded: [B, T, 1, D] -> [B, T, U, D]
        enc_expanded = encoder_out.unsqueeze(2).expand(-1, -1, U, -1)
        # pred_expanded: [B, 1, U, H] -> [B, T, U, H]
        pred_expanded = pred_out.unsqueeze(1).expand(-1, T, -1, -1)
        joint_input = torch.tanh(torch.cat([enc_expanded, pred_expanded], dim=-1))
        logits = self.joint_linear(joint_input)  # [B, T, U, vocab]

        # For the loss, torchaudio's rnnt_loss expects:
        # logits: [B, T, U, V], targets: [B, U-1], lengths etc.
        # The last U index is for the blank transition
        target_lengths_clamped = target_lengths.clamp(min=1, max=U-1)
        return logits, target_lengths_clamped

    def predict(self, encoder_out, tokens, state=None):
        """
        推理模式：逐步解码
        encoder_out: [B, T, D]
        tokens: [B, 1]  当前输入 token
        """
        pred_emb = self.embed(tokens)
        B = tokens.size(0)
        if state is None:
            num_layers = self.pred_rnn.num_layers
            h0 = torch.zeros(num_layers, B, self.pred_rnn.hidden_size, device=tokens.device)
            c0 = torch.zeros(num_layers, B, self.pred_rnn.hidden_size, device=tokens.device)
            state = (h0, c0)
        pred_out, state = self.pred_rnn(pred_emb, state)
        T_enc = encoder_out.size(1)
        enc_part = encoder_out.unsqueeze(2).expand(-1, -1, 1, -1)[:, -1:, :, :]
        pred_part = pred_out.unsqueeze(1).expand(-1, 1, -1, -1)
        joint_input = torch.tanh(torch.cat([enc_part, pred_part], dim=-1))
        logits = self.joint_linear(joint_input).squeeze(1)  # [B, 1, V]
        return logits, state


# ======================================================================
# 联合模型：Conformer + CTC + Attention + Transducer
# ======================================================================

class ConformerCTCATTNTransducer(nn.Module):
    """
    联合模型：共享 Conformer 编码器
    三个解码头：CTC / Attention / Transducer
    支持流式推理和流式训练
    """

    def __init__(self, vocab_size, idim=80, d_model=144, n_head=4,
                 d_ff=1024, enc_blocks=6, attn_blocks=3,
                 pred_dim=144, pred_layers=1,
                 ctc_weight=0.3, attn_weight=0.3, trans_weight=0.4,
                 dropout=0.1, chunk_size=0, right_context=0,
                 streaming_prob=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.ctc_weight = ctc_weight
        self.attn_weight = attn_weight
        self.trans_weight = trans_weight
        self.chunk_size = chunk_size
        self.right_context = right_context
        self.streaming_prob = streaming_prob

        # 共享编码器
        self.encoder = ConformerEncoder(idim, d_model, n_head, d_ff,
                                        enc_blocks, dropout,
                                        chunk_size=chunk_size,
                                        right_context=right_context,
                                        streaming_prob=streaming_prob)

        # CTC 分支
        self.ctc_linear = nn.Linear(d_model, vocab_size)

        # Attention 解码器
        self.attn_decoder = AttentionDecoder(
            vocab_size, d_model, n_head, d_ff, attn_blocks, dropout
        )

        # Transducer 解码器
        self.trans_decoder = TransducerDecoder(
            vocab_size, d_model, pred_dim, pred_dim, pred_layers, dropout
        )

    def forward(self, feats, feat_lens, attn_tokens_in, attn_tokens_out,
                trans_tokens, trans_token_lens, is_streaming=None,
                chunk_size=None, right_context=None):
        """
        前向计算所有三个损失

        feats: [B, T, idim]  Fbank 特征
        feat_lens: [B]  特征帧长（下采样后需要调整）
        attn_tokens_in: [B, L]  Attention 解码器输入 (含 <sos>)
        attn_tokens_out: [B, L]  Attention 解码器目标
        trans_tokens: [B, U]  Transducer 输入 token (含 <sos>)
        trans_token_lens: [B]  Transducer token 长度
        is_streaming: 若为 True 则使用流式训练（chunk mask）
        chunk_size: 覆盖默认 chunk_size
        right_context: 覆盖默认 right_context
        """
        # ---- 共享编码 ----
        encoder_out = self.encoder(feats, is_streaming=is_streaming,
                                   chunk_size=chunk_size, right_context=right_context)
        # [B, T', D]
        T_enc = encoder_out.size(1)
        enc_lens = ((feat_lens + 1) // 2 + 1) // 2  # 下采样 4x

        # --- CTC Loss ---
        ctc_logits = self.ctc_linear(encoder_out)  # [B, T', V]
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)
        # 去掉 attn_tokens_out 中的 padding (-1)
        ys_padded = attn_tokens_out.clamp(min=0)
        ys_lens = (attn_tokens_out != -1).sum(dim=1).clamp(min=1)
        ctc_loss = F.ctc_loss(
            ctc_log_probs.transpose(0, 1),
            ys_padded,
            enc_lens.clamp(min=1, max=T_enc),
            ys_lens,
            blank=0,
            zero_infinity=True,
        )

        # --- Attention Loss ---
        attn_logits = self.attn_decoder(attn_tokens_in, encoder_out)
        # [B, L, V] -> [B*L, V]
        attn_loss = F.cross_entropy(
            attn_logits.reshape(-1, self.vocab_size),
            attn_tokens_out.reshape(-1),
            ignore_index=-1,
        )

        # --- Transducer Loss ---
        trans_logits, trans_lens_clamped = self.trans_decoder(
            encoder_out, trans_tokens, trans_token_lens
        )
        # targets: trans_tokens[:, 1:]  去掉 <sos> (需 int32 类型)
        trans_targets = trans_tokens[:, 1:].int()  # [B, U-1], int32
        trans_logit_lengths = enc_lens.clamp(min=1, max=T_enc).int()
        trans_target_lengths = (trans_token_lens - 1).clamp(min=1, max=trans_tokens.size(1) - 1).int()
        if trans_targets.size(1) > 0 and trans_logits.size(2) > 0:
            transducer_loss = torchaudio.functional.rnnt_loss(
                logits=trans_logits,
                targets=trans_targets,
                logit_lengths=trans_logit_lengths,
                target_lengths=trans_target_lengths,
                blank=0,
                reduction="mean",
            )
        else:
            transducer_loss = torch.tensor(0.0, device=feats.device)

        # --- 联合损失 ---
        total_loss = (
            self.ctc_weight * ctc_loss
            + self.attn_weight * attn_loss
            + self.trans_weight * transducer_loss
        )

        return {
            "loss": total_loss,
            "ctc_loss": ctc_loss.detach(),
            "attn_loss": attn_loss.detach(),
            "transducer_loss": transducer_loss.detach(),
        }

    @torch.no_grad()
    def recognize_ctc_greedy(self, feats, idx2token):
        """CTC 贪心解码（过滤 blank=0 和 sos/eos）"""
        encoder_out = self.encoder(feats)
        ctc_logits = self.ctc_linear(encoder_out)
        preds = ctc_logits.argmax(dim=-1)  # [B, T']
        sos_eos_id = self.vocab_size - 1
        texts = []
        for b in range(preds.size(0)):
            tokens = []
            prev = -1
            for t in range(preds.size(1)):
                tok = preds[b, t].item()
                if tok not in (0, sos_eos_id) and tok != prev:
                    tokens.append(idx2token.get(tok, ""))
                prev = tok
            texts.append("".join(tokens))
        return texts

    @torch.no_grad()
    def recognize_attention(self, feats, max_len=20, sos_id=-1, eos_id=-1):
        """Attention 自回归解码"""
        if sos_id < 0:
            sos_id = self.vocab_size - 1
        if eos_id < 0:
            eos_id = self.vocab_size - 1
        encoder_out = self.encoder(feats)
        B = encoder_out.size(0)
        ys = torch.full((B, 1), sos_id, dtype=torch.long, device=feats.device)
        ended = [False] * B
        for _ in range(max_len):
            logits = self.attn_decoder(ys, encoder_out)
            next_tok = logits[:, -1, :].argmax(dim=-1)  # [B]
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            for b in range(B):
                if next_tok[b].item() == eos_id:
                    ended[b] = True
            if all(ended):
                break
        return ys  # [B, L+1]

    @torch.no_grad()
    def recognize_transducer(self, feats, max_len=50, sos_id=-1):
        """Transducer 贪心解码（blank 跳过，非 blank 发射）"""
        if sos_id < 0:
            sos_id = self.vocab_size - 1
        encoder_out = self.encoder(feats)  # [B, T_enc, D]
        B, T_enc, D = encoder_out.shape
        results = [[] for _ in range(B)]
        state = None
        # decoder input token starts with <sos>
        y = torch.full((B, 1), sos_id, dtype=torch.long, device=feats.device)
        t = 0  # encoder frame index

        for _ in range(max_len * 2):
            if t >= T_enc:
                break
            # Predict using current encoder frame
            logits, state = self.trans_decoder.predict(
                encoder_out[:, t:t+1, :], y, state
            )
            # argmax over vocab
            next_tok = logits[:, 0, :].argmax(dim=-1)  # [B]
            tok = next_tok[0].item()

            if tok == 0:  # blank -> move to next encoder frame
                t += 1
            elif tok == sos_id:  # sos/eos -> stop
                break
            else:  # non-blank token -> emit
                results[0].append(tok)
                y = torch.full((B, 1), tok, dtype=torch.long, device=feats.device)

        # Convert token ids to texts in the recognize function
        return results

    @torch.no_grad()
    def recognize_ctc_streaming(self, feats, idx2token, chunk_size=16,
                                 right_context=4):
        """
        流式 CTC 贪心解码：逐 chunk 处理编码器。

        Args:
            feats: (B, T, idim) Fbank 特征
            idx2token: index to token mapping
            chunk_size: 每 chunk 帧数（原始特征帧，将自动对齐到下采样后边界）
            right_context: 每 chunk 右侧上下文帧数

        Returns:
            texts: [str] 识别文本列表
        """
        from model.streaming_helper import init_attn_cache

        B, T_full, _ = feats.shape
        num_blocks = len(self.encoder.blocks)
        d_model = self.encoder.linear.out_features

        # 初始化 KV cache
        # cache 的最大长度设为按 chunk 数估计
        max_cache_len = T_full // 4 + 32
        attn_cache = init_attn_cache(num_blocks, B, max_cache_len, d_model, feats.device)

        all_ctc_logits = []
        offset = 0

        # 按 chunk 处理
        for start in range(0, T_full, chunk_size):
            end = min(T_full, start + chunk_size + right_context)
            chunk = feats[:, start:end, :]

            # 执行 forward_chunk
            chunk_out, attn_cache, _ = self.encoder.forward_chunk(
                chunk, offset=offset, attn_cache=attn_cache
            )
            T_out = chunk_out.size(1)
            offset += T_out

            # CTC logits 取前 chunk 对应的部分（去掉 right_context 贡献）
            ctc_logits = self.ctc_linear(chunk_out)  # (B, T_out, vocab)
            all_ctc_logits.append(ctc_logits)

            if end >= T_full:
                break

        # 合并所有 chunk 的 CTC logits
        ctc_logits_all = torch.cat(all_ctc_logits, dim=1)  # (B, T', vocab)
        preds = ctc_logits_all.argmax(dim=-1)  # (B, T')

        # 贪心解码 + 去重
        sos_eos_id = self.vocab_size - 1
        texts = []
        for b in range(B):
            tokens = []
            prev = -1
            for t in range(preds.size(1)):
                tok = preds[b, t].item()
                if tok not in (0, sos_eos_id) and tok != prev:
                    tokens.append(idx2token.get(tok, ""))
                prev = tok
            texts.append("".join(tokens))
        return texts

    @torch.no_grad()
    def recognize_transducer_streaming(self, feats, chunk_size=16,
                                        right_context=4, max_len=50, sos_id=-1):
        """
        流式 Transducer 解码：逐 chunk 处理编码器，输出 token 序列。

        Args:
            feats: (B, T, idim) Fbank 特征
            chunk_size: 每 chunk 帧数
            right_context: 每 chunk 右侧上下文帧数
            max_len: 最大输出长度
            sos_id: <sos> token id

        Returns:
            results: [token_ids] 每句的 token 序列
        """
        from model.streaming_helper import init_attn_cache

        if sos_id < 0:
            sos_id = self.vocab_size - 1

        B, T_full, _ = feats.shape
        num_blocks = len(self.encoder.blocks)
        d_model = self.encoder.linear.out_features
        max_cache_len = T_full // 4 + 32
        attn_cache = init_attn_cache(num_blocks, B, max_cache_len, d_model, feats.device)

        results = [[] for _ in range(B)]
        state = None
        y = torch.full((B, 1), sos_id, dtype=torch.long, device=feats.device)

        offset = 0
        # 逐 chunk 处理编码器，每帧逐个用 transducer 解码
        for start in range(0, T_full, chunk_size):
            end = min(T_full, start + chunk_size + right_context)
            chunk = feats[:, start:end, :]

            chunk_out, attn_cache, _ = self.encoder.forward_chunk(
                chunk, offset=offset, attn_cache=attn_cache
            )
            T_out = chunk_out.size(1)
            offset += T_out

            # 对当前 chunk 内的每帧，执行 transducer 解码
            t = 0
            while t < T_out:
                if len(results[0]) >= max_len:
                    break
                logits, state = self.trans_decoder.predict(
                    chunk_out[:, t:t+1, :], y, state
                )
                next_tok = logits[:, 0, :].argmax(dim=-1)
                tok = next_tok[0].item()

                if tok == 0:  # blank -> move to next encoder frame
                    t += 1
                elif tok == sos_id:  # stop
                    return results
                else:
                    results[0].append(tok)
                    y = torch.full((B, 1), tok, dtype=torch.long, device=feats.device)

            if len(results[0]) >= max_len:
                break

        return results
