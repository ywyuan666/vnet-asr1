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


def _apply_attention_mask(scores, mask):
    """Support keep masks (bool / 0-1) and additive masks (0 / -inf)."""
    if mask.dtype == torch.bool:
        return scores.masked_fill(~mask, float("-inf"))
    if torch.is_floating_point(mask):
        if torch.isinf(mask).any() or (mask < 0).any():
            return scores + mask
        return scores.masked_fill(mask == 0, float("-inf"))
    return scores.masked_fill(mask == 0, float("-inf"))


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.w1.weight, gain=1.0)
        nn.init.xavier_uniform_(self.w2.weight, gain=1.0)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)

    def forward(self, x):
        return self.w2(self.dropout(F.silu(self.w1(x))))


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
        batch, steps, dim = x.shape
        q, k, v = self.w_qkv(x).chunk(3, dim=-1)
        q = q.view(batch, steps, self.h, self.d_k).transpose(1, 2)
        k = k.view(batch, steps, self.h, self.d_k).transpose(1, 2)
        v = v.view(batch, steps, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = _apply_attention_mask(scores, mask)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, steps, dim)
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
        bq, tq, dim = query.shape
        bk, tk, _ = key.shape
        q = self.w_q(query).view(bq, tq, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(bk, tk, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(bk, tk, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = _apply_attention_mask(scores, mask)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bq, tq, dim)
        return self.out(out)


class ConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=15, dropout=0.1):
        super().__init__()
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
        )
        self.norm = nn.LayerNorm(d_model)
        self.pointwise2 = nn.Conv1d(d_model, d_model, 1)
        self.act = Swish()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.glu(self.pointwise1(x), dim=1)
        x = self.depthwise(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.act(x)
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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + 0.5 * self.dropout(self.ff1(self.norm_ff1(x)))
        effective_mask = mask
        if effective_mask is not None and effective_mask.dim() == 3:
            keep = effective_mask == 0
            effective_mask = keep.unsqueeze(-1) & keep.unsqueeze(-2)
        x = x + self.dropout(self.attn(self.norm_attn(x), effective_mask))
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.dropout(self.ff2(self.norm_ff2(x)))
        return self.norm_final(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ConformerEncoder(nn.Module):
    def __init__(self, idim=80, d_model=144, n_head=4, d_ff=1024,
                 num_blocks=6, dropout=0.1, kernel_size=15):
        super().__init__()
        self.subsample = nn.Sequential(
            nn.Conv2d(1, d_model, (3, 3), (2, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
        )
        self.output_dim = d_model
        self.time_downsample_factor = 2
        self.freq_weight = nn.Parameter(torch.ones(d_model, idim) / idim)
        self.pos_enc = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, n_head, d_ff, kernel_size, dropout)
            for _ in range(num_blocks)
        ])

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return (lengths + self.time_downsample_factor - 1) // self.time_downsample_factor

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)
        x = self.subsample(x)
        _, _, _, _ = x.shape
        x = (x * self.freq_weight[None, :, None, :]).sum(dim=3)
        x = x.transpose(1, 2)
        x = self.pos_enc(x)
        for blk in self.blocks:
            x = blk(x, mask)
        return x


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
        _, length = ys.shape
        y = self.pos(self.embed(ys))
        causal_mask = torch.tril(torch.ones(length, length, device=ys.device)).bool()
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
        self.norm_enc = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, memory, causal_mask):
        y = y + self.dropout(self.self_attn(self.norm1(y), causal_mask))
        memory_norm = self.norm_enc(memory)
        y = y + self.dropout(self.cross_attn(self.norm2(y), memory_norm, memory_norm))
        y = y + self.dropout(self.ff(self.norm3(y)))
        return y


class TransducerDecoder(nn.Module):
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
        self.joint_linear = nn.Linear(d_model + hidden_dim, vocab_size)

    def forward(self, encoder_out, targets, target_lengths):
        batch, steps, _ = encoder_out.shape
        target_steps = targets.shape[1]
        pred_emb = self.embed(targets)
        pred_out, _ = self.pred_rnn(pred_emb)
        enc_expanded = encoder_out.unsqueeze(2).expand(-1, -1, target_steps, -1)
        pred_expanded = pred_out.unsqueeze(1).expand(-1, steps, -1, -1)
        joint_input = torch.tanh(torch.cat([enc_expanded, pred_expanded], dim=-1))
        logits = self.joint_linear(joint_input)
        target_lengths_clamped = target_lengths.clamp(min=1, max=target_steps - 1)
        return logits, target_lengths_clamped

    def predict(self, encoder_out, tokens, state=None):
        pred_emb = self.embed(tokens)
        batch = tokens.size(0)
        if state is None:
            num_layers = self.pred_rnn.num_layers
            h0 = torch.zeros(num_layers, batch, self.pred_rnn.hidden_size, device=tokens.device)
            c0 = torch.zeros(num_layers, batch, self.pred_rnn.hidden_size, device=tokens.device)
            state = (h0, c0)
        pred_out, state = self.pred_rnn(pred_emb, state)
        enc_part = encoder_out.unsqueeze(2).expand(-1, -1, 1, -1)[:, -1:, :, :]
        pred_part = pred_out.unsqueeze(1).expand(-1, 1, -1, -1)
        joint_input = torch.tanh(torch.cat([enc_part, pred_part], dim=-1))
        logits = self.joint_linear(joint_input).squeeze(1)
        return logits, state


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
        enc_steps = encoder_out.size(1)
        enc_lens = self.encoder.output_lengths(feat_lens).clamp(min=1, max=enc_steps)

        ctc_logits = self.ctc_linear(encoder_out)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)
        ys_padded = attn_tokens_out.clamp(min=0)
        ys_lens = ((attn_tokens_out != -1).sum(dim=1) - 1).clamp(min=1)
        ctc_loss = F.ctc_loss(
            ctc_log_probs.transpose(0, 1),
            ys_padded,
            enc_lens,
            ys_lens,
            blank=0,
            zero_infinity=True,
        )

        attn_logits = self.attn_decoder(attn_tokens_in, encoder_out)
        attn_loss = F.cross_entropy(
            attn_logits.reshape(-1, self.vocab_size),
            attn_tokens_out.reshape(-1),
            ignore_index=-1,
        )

        if self.trans_weight > 0:
            trans_logits, _ = self.trans_decoder(encoder_out, trans_tokens, trans_token_lens)
            trans_targets = trans_tokens[:, 1:].int()
            trans_logit_lengths = enc_lens.int()
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

    def _resolve_decode_max_len(self, encoder_out, max_len):
        if max_len is None or max_len <= 0:
            return max(1, encoder_out.size(1))
        return max_len

    @torch.no_grad()
    def recognize_ctc_greedy(self, feats, idx2token):
        encoder_out = self.encoder(feats)
        ctc_logits = self.ctc_linear(encoder_out)
        preds = ctc_logits.argmax(dim=-1)
        sos_eos_id = self.vocab_size - 1
        texts = []
        for batch_idx in range(preds.size(0)):
            tokens = []
            prev = -1
            for step in range(preds.size(1)):
                tok = preds[batch_idx, step].item()
                if tok not in (0, sos_eos_id) and tok != prev:
                    tokens.append(idx2token.get(tok, ""))
                prev = tok
            texts.append("".join(tokens))
        return texts

    @torch.no_grad()
    def recognize_attention(self, feats, max_len=None, sos_id=-1, eos_id=-1):
        if sos_id < 0:
            sos_id = self.vocab_size - 1
        if eos_id < 0:
            eos_id = self.vocab_size - 1
        encoder_out = self.encoder(feats)
        max_len = self._resolve_decode_max_len(encoder_out, max_len)
        batch = encoder_out.size(0)
        ys = torch.full((batch, 1), sos_id, dtype=torch.long, device=feats.device)
        ended = [False] * batch
        for _ in range(max_len):
            logits = self.attn_decoder(ys, encoder_out)
            next_tok = logits[:, -1, :].argmax(dim=-1)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            for idx in range(batch):
                if next_tok[idx].item() == eos_id:
                    ended[idx] = True
            if all(ended):
                break
        return ys

    @torch.no_grad()
    def recognize_transducer(self, feats, max_len=None, sos_id=-1):
        if sos_id < 0:
            sos_id = self.vocab_size - 1
        encoder_out = self.encoder(feats)
        _, enc_steps, _ = encoder_out.shape
        max_len = self._resolve_decode_max_len(encoder_out, max_len)
        results = [[] for _ in range(encoder_out.size(0))]
        state = None
        y = torch.full((encoder_out.size(0), 1), sos_id, dtype=torch.long, device=feats.device)
        step_idx = 0
        steps = 0
        max_steps = max_len + enc_steps * 4

        while step_idx < enc_steps and len(results[0]) < max_len and steps < max_steps:
            steps += 1
            logits, state = self.trans_decoder.predict(encoder_out[:, step_idx:step_idx + 1, :], y, state)
            next_tok = logits[:, 0, :].argmax(dim=-1)
            tok = next_tok[0].item()
            if tok == 0:
                step_idx += 1
            elif tok == sos_id:
                break
            else:
                results[0].append(tok)
                y = torch.full((encoder_out.size(0), 1), tok, dtype=torch.long, device=feats.device)
        return results
