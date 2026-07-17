# -*- coding: utf-8 -*-
"""
Conformer + CTC / Attention / Transducer 联合模型
=================================================
用于 Hugging Face Space 部署。
"""
import math

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
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), Swish(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.d_k = d_model // n_head
        self.h = n_head
        self.w_qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        q, k, v = self.w_qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.h, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class MultiHeadCrossAttention(nn.Module):
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
        q = self.w_q(query).view(Bq, Tq, self.h, self.d_k).transpose(1, 2)
        Bk, Tk, _ = key.shape
        k = self.w_k(key).view(Bk, Tk, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(Bk, Tk, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(Bq, Tq, D)
        return self.out(out)


class ConvModule(nn.Module):
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

    def forward(self, x, mask=None):
        x = x + 0.5 * self.ff1(self.norm_ff1(x))
        x = x + self.attn(self.norm_attn(x), mask)
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.ff2(self.norm_ff2(x))
        return self.norm_final(x)


# ======================================================================
# 编码器
# ======================================================================

class ConformerEncoder(nn.Module):
    def __init__(self, idim=80, d_model=144, n_head=4, d_ff=1024,
                 num_blocks=6, dropout=0.1, kernel_size=15):
        super().__init__()
        self.subsample = nn.Sequential(
            nn.Conv2d(1, d_model, 3, 2, padding=1), nn.ReLU(),
            nn.Conv2d(d_model, d_model, 3, 2, padding=1), nn.ReLU(),
        )
        sub_dim = d_model * (idim // 4)
        self.linear = nn.Linear(sub_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, n_head, d_ff, kernel_size, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)
        x = self.subsample(x)
        B, C, Tt, Ff = x.shape
        x = x.transpose(1, 2).contiguous().view(B, Tt, C * Ff)
        x = self.linear(x)
        x = self.pos_enc(x)
        for blk in self.blocks:
            x = blk(x, mask)
        return x


class PositionalEncoding(nn.Module):
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
# Attention Decoder
# ======================================================================

class AttentionDecoder(nn.Module):
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
        B, L = ys.shape
        y = self.pos(self.embed(ys))
        causal_mask = torch.tril(torch.ones(L, L, device=ys.device)).bool()
        for layer in self.layers:
            y = layer(y, memory, causal_mask)
        y = self.norm(y)
        return self.out(y)


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
        y = y + self.dropout(self.self_attn(self.norm1(y), causal_mask))
        y = y + self.dropout(self.cross_attn(self.norm2(y), memory, memory))
        y = y + self.dropout(self.ff(self.norm3(y)))
        return y


# ======================================================================
# Transducer Decoder
# ======================================================================

class TransducerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=144, hidden_dim=144,
                 embed_dim=144, num_layers=1, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pred_rnn = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.joint_linear = nn.Linear(d_model + hidden_dim, vocab_size)

    def forward(self, encoder_out, targets, target_lengths):
        B, T, D = encoder_out.shape
        U = targets.shape[1]
        pred_emb = self.embed(targets)
        pred_out, _ = self.pred_rnn(pred_emb)
        enc_expanded = encoder_out.unsqueeze(2).expand(-1, -1, U, -1)
        pred_expanded = pred_out.unsqueeze(1).expand(-1, T, -1, -1)
        joint_input = torch.tanh(torch.cat([enc_expanded, pred_expanded], dim=-1))
        logits = self.joint_linear(joint_input)
        target_lengths_clamped = target_lengths.clamp(min=1, max=U-1)
        return logits, target_lengths_clamped

    def predict(self, encoder_out, tokens, state=None):
        pred_emb = self.embed(tokens)
        B = tokens.size(0)
        if state is None:
            num_layers = self.pred_rnn.num_layers
            h0 = torch.zeros(num_layers, B, self.pred_rnn.hidden_size, device=tokens.device)
            c0 = torch.zeros(num_layers, B, self.pred_rnn.hidden_size, device=tokens.device)
            state = (h0, c0)
        pred_out, state = self.pred_rnn(pred_emb, state)
        enc_part = encoder_out.unsqueeze(2).expand(-1, -1, 1, -1)[:, -1:, :, :]
        pred_part = pred_out.unsqueeze(1).expand(-1, 1, -1, -1)
        joint_input = torch.tanh(torch.cat([enc_part, pred_part], dim=-1))
        logits = self.joint_linear(joint_input).squeeze(1)
        return logits, state


# ======================================================================
# 联合模型
# ======================================================================

class ConformerCTCATTNTransducer(nn.Module):
    def __init__(self, vocab_size, idim=80, d_model=144, n_head=4,
                 d_ff=1024, enc_blocks=6, attn_blocks=3,
                 pred_dim=144, pred_layers=1,
                 ctc_weight=0.3, attn_weight=0.3, trans_weight=0.4,
                 dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.ctc_weight = ctc_weight
        self.attn_weight = attn_weight
        self.trans_weight = trans_weight
        self.encoder = ConformerEncoder(idim, d_model, n_head, d_ff, enc_blocks, dropout)
        self.ctc_linear = nn.Linear(d_model, vocab_size)
        self.attn_decoder = AttentionDecoder(
            vocab_size, d_model, n_head, d_ff, attn_blocks, dropout
        )
        self.trans_decoder = TransducerDecoder(
            vocab_size, d_model, pred_dim, pred_dim, pred_layers, dropout
        )

    def forward(self, feats, feat_lens, attn_tokens_in, attn_tokens_out,
                trans_tokens, trans_token_lens):
        encoder_out = self.encoder(feats)
        T_enc = encoder_out.size(1)
        enc_lens = ((feat_lens + 1) // 2 + 1) // 2
        ctc_logits = self.ctc_linear(encoder_out)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)
        ys_padded = attn_tokens_out.clamp(min=0)
        ys_lens = (attn_tokens_out != -1).sum(dim=1).clamp(min=1)
        ctc_loss = F.ctc_loss(
            ctc_log_probs.transpose(0, 1), ys_padded,
            enc_lens.clamp(min=1, max=T_enc), ys_lens,
            blank=0, zero_infinity=True,
        )
        attn_logits = self.attn_decoder(attn_tokens_in, encoder_out)
        attn_loss = F.cross_entropy(
            attn_logits.reshape(-1, self.vocab_size),
            attn_tokens_out.reshape(-1), ignore_index=-1,
        )
        trans_logits, trans_lens_clamped = self.trans_decoder(
            encoder_out, trans_tokens, trans_token_lens
        )
        trans_targets = trans_tokens[:, 1:].int()
        trans_logit_lengths = enc_lens.clamp(min=1, max=T_enc).int()
        trans_target_lengths = (trans_token_lens - 1).clamp(min=1, max=trans_tokens.size(1) - 1).int()
        if trans_targets.size(1) > 0 and trans_logits.size(2) > 0:
            transducer_loss = torchaudio.functional.rnnt_loss(
                logits=trans_logits, targets=trans_targets,
                logit_lengths=trans_logit_lengths,
                target_lengths=trans_target_lengths,
                blank=0, reduction="mean",
            )
        else:
            transducer_loss = torch.tensor(0.0, device=feats.device)
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
        encoder_out = self.encoder(feats)
        ctc_logits = self.ctc_linear(encoder_out)
        preds = ctc_logits.argmax(dim=-1)
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
            next_tok = logits[:, -1, :].argmax(dim=-1)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            for b in range(B):
                if next_tok[b].item() == eos_id:
                    ended[b] = True
            if all(ended):
                break
        return ys

    @torch.no_grad()
    def recognize_transducer(self, feats, max_len=50, sos_id=-1):
        if sos_id < 0:
            sos_id = self.vocab_size - 1
        encoder_out = self.encoder(feats)
        B, T_enc, D = encoder_out.shape
        results = [[] for _ in range(B)]
        state = None
        y = torch.full((B, 1), sos_id, dtype=torch.long, device=feats.device)
        t = 0
        for _ in range(max_len * 2):
            if t >= T_enc:
                break
            logits, state = self.trans_decoder.predict(
                encoder_out[:, t:t+1, :], y, state
            )
            next_tok = logits[:, 0, :].argmax(dim=-1)
            tok = next_tok[0].item()
            if tok == 0:
                t += 1
            elif tok == sos_id:
                break
            else:
                results[0].append(tok)
                y = torch.full((B, 1), tok, dtype=torch.long, device=feats.device)
        return results
