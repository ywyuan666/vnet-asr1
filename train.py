# -*- coding: utf-8 -*-
"""
train.py
========
自定义训练脚本：Conformer + CTC/Attention/Transducer 三任务联合训练。

不依赖 WeNet 框架，使用 PyTorch 原生训练循环。
支持：
  - 自定义数据加载
  - 三任务联合损失
  - Adam + Warmup LR
  - 每个 epoch 验证 + 保存 checkpoint
  - TensorBoard 日志
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.conformer_ctc_attn_transducer import ConformerCTCATTNTransducer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ======================================================================
# 数据集
# ======================================================================

class AsrDataset(Dataset):
    """WeNet 风格 data.list 数据加载"""

    def __init__(self, data_list_path, cmvn_path=None, max_length=2000):
        self.items = []
        with open(data_list_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    self.items.append(obj)
        self.max_length = max_length

        # 加载 CMVN (mean_stat/var_stat 是累加和，需除以 frame_num 得到真实均值和方差)
        self.cmvn_mean = None
        self.cmvn_var = None
        if cmvn_path and os.path.exists(cmvn_path):
            with open(cmvn_path) as f:
                cmvn = json.load(f)
            frame_num = cmvn["frame_num"]
            mean_stat = torch.tensor(cmvn["mean_stat"], dtype=torch.float32)
            var_stat = torch.tensor(cmvn["var_stat"], dtype=torch.float32)
            self.cmvn_mean = mean_stat / frame_num
            self.cmvn_var = var_stat / frame_num - self.cmvn_mean * self.cmvn_mean

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        wav_path = item["wav"]
        txt = item["txt"]

        # 加载音频（用 soundfile 避免 ffmpeg 依赖）
        import soundfile as sf
        data, sr = sf.read(wav_path)
        if sr != 16000:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            sr = 16000
        waveform = torch.from_numpy(data).float().unsqueeze(0)  # [1, T]
        waveform = waveform * (1 << 15)
        feat = kaldi.fbank(
            waveform,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=1.0,
            sample_frequency=16000,
        )  # [T, 80]

        # CMVN
        if self.cmvn_mean is not None:
            feat = (feat - self.cmvn_mean) / (self.cmvn_var.sqrt() + 1e-10)

        return feat, txt


def collate_fn(batch, vocab, sos_id, pad_id=-1):
    """自定义 collate 函数"""
    feats_list = []
    feat_lens = []
    texts = []
    for feat, txt in batch:
        T = feat.shape[0]
        feats_list.append(feat)
        feat_lens.append(T)
        texts.append(txt)

    # 填充特征到相同长度
    max_T = max(feat_lens)
    B = len(batch)
    D = feats_list[0].shape[-1]
    feats_padded = torch.zeros(B, max_T, D)
    for i, f in enumerate(feats_list):
        feats_padded[i, :f.shape[0]] = f

    # 将文字转为 token id
    token_ids = []
    for txt in texts:
        ids = [vocab.get(ch, 1) for ch in txt]  # <unk>=1
        token_ids.append(torch.tensor(ids, dtype=torch.long))

    max_L = max(len(t) for t in token_ids) + 1  # +1 for sos
    attn_in = torch.full((B, max_L), sos_id, dtype=torch.long)
    attn_out = torch.full((B, max_L), -1, dtype=torch.long)
    trans_in = torch.full((B, max_L), sos_id, dtype=torch.long)
    trans_lens = torch.zeros(B, dtype=torch.long)

    for i, ids in enumerate(token_ids):
        L = len(ids)
        attn_in[i, 0] = sos_id
        attn_in[i, 1:L+1] = ids
        attn_out[i, :L] = ids
        attn_out[i, L] = sos_id
        # Transducer: 含 <sos> 前缀
        trans_in[i, 0] = sos_id
        if L > 0:
            trans_in[i, 1:L+1] = ids
        trans_lens[i] = L + 1

    return feats_padded, torch.tensor(feat_lens), attn_in, attn_out, trans_in, trans_lens


# ======================================================================
# Warmup LR 调度器
# ======================================================================

class WarmupLR:
    def __init__(self, optimizer, warmup_steps=500, d_model=144):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self._step = 0

    def step(self):
        self._step += 1
        lr = self.d_model ** (-0.5) * min(
            self._step ** (-0.5), self._step * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


# ======================================================================
# 训练与验证
# ======================================================================

def train_epoch(model, loader, optimizer, scheduler, device, args, epoch):
    model.train()
    total_loss = 0
    total_ctc = 0
    total_attn = 0
    total_trans = 0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", file=sys.stdout)
    for batch in pbar:
        feats, feat_lens, attn_in, attn_out, trans_in, trans_lens = batch
        feats = feats.to(device)
        feat_lens = feat_lens.to(device)
        attn_in = attn_in.to(device)
        attn_out = attn_out.to(device)
        trans_in = trans_in.to(device)
        trans_lens = trans_lens.to(device)

        optimizer.zero_grad()
        outputs = model(feats, feat_lens, attn_in, attn_out, trans_in, trans_lens)
        loss = outputs["loss"]

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += outputs["loss"].item()
        total_ctc += outputs["ctc_loss"].item()
        total_attn += outputs["attn_loss"].item()
        total_trans += outputs["transducer_loss"].item()
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{outputs['loss'].item():.4f}",
            "ctc": f"{outputs['ctc_loss'].item():.4f}",
            "attn": f"{outputs['attn_loss'].item():.4f}",
            "trans": f"{outputs['transducer_loss'].item():.4f}",
        })

    return {
        "loss": total_loss / max(1, num_batches),
        "ctc": total_ctc / max(1, num_batches),
        "attn": total_attn / max(1, num_batches),
        "trans": total_trans / max(1, num_batches),
    }


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    total_ctc = 0
    total_attn = 0
    total_trans = 0
    num_batches = 0

    for batch in tqdm(loader, desc="Validating", file=sys.stdout):
        feats, feat_lens, attn_in, attn_out, trans_in, trans_lens = batch
        feats = feats.to(device)
        feat_lens = feat_lens.to(device)
        attn_in = attn_in.to(device)
        attn_out = attn_out.to(device)
        trans_in = trans_in.to(device)
        trans_lens = trans_lens.to(device)

        outputs = model(feats, feat_lens, attn_in, attn_out, trans_in, trans_lens)

        total_loss += outputs["loss"].item()
        total_ctc += outputs["ctc_loss"].item()
        total_attn += outputs["attn_loss"].item()
        total_trans += outputs["transducer_loss"].item()
        num_batches += 1

    return {
        "loss": total_loss / max(1, num_batches),
        "ctc": total_ctc / max(1, num_batches),
        "attn": total_attn / max(1, num_batches),
        "trans": total_trans / max(1, num_batches),
    }


# ======================================================================
# 主函数
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--cv_data", required=True)
    parser.add_argument("--dict", required=True)
    parser.add_argument("--cmvn", default=None)
    parser.add_argument("--model_dir", default="exp/conformer_ctc_attn_transducer")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epoch", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--d_model", type=int, default=144)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ctc_weight", type=float, default=0.3)
    parser.add_argument("--attn_weight", type=float, default=0.3)
    parser.add_argument("--trans_weight", type=float, default=0.4)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.model_dir, exist_ok=True)

    # 加载字典
    vocab = {}
    with open(args.dict, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                vocab[parts[0]] = int(parts[1])
    vocab_size = len(vocab)
    sos_id = vocab_size - 1  # <sos/eos> 是最后一个

    print(f"词表大小: {vocab_size}, <sos/eos> id: {sos_id}")

    # 数据集
    train_ds = AsrDataset(args.train_data, args.cmvn)
    cv_ds = AsrDataset(args.cv_data, args.cmvn)
    collate = lambda b: collate_fn(b, vocab, sos_id)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=0, pin_memory=(device.type == "cuda"),
    )
    cv_loader = DataLoader(
        cv_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
    )

    print(f"训练集: {len(train_ds)} 条, 验证集: {len(cv_ds)} 条")

    # 模型
    model = ConformerCTCATTNTransducer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        ctc_weight=args.ctc_weight,
        attn_weight=args.attn_weight,
        trans_weight=args.trans_weight,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupLR(optimizer, args.warmup_steps, args.d_model)

    # 训练
    best_loss = float("inf")
    for epoch in range(1, args.max_epoch + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, args, epoch)
        cv_metrics = validate(model, cv_loader, device)

        lr = scheduler.get_lr()
        log = (f"Epoch {epoch:3d}/{args.max_epoch} | "
               f"Train: loss={train_metrics['loss']:.4f} "
               f"(CTC={train_metrics['ctc']:.4f} "
               f"Attn={train_metrics['attn']:.4f} "
               f"Trans={train_metrics['trans']:.4f}) | "
               f"CV: loss={cv_metrics['loss']:.4f} "
               f"(CTC={cv_metrics['ctc']:.4f} "
               f"Attn={cv_metrics['attn']:.4f} "
               f"Trans={cv_metrics['trans']:.4f}) | "
               f"LR={lr:.2e}")
        print(log)

        # 保存日志
        with open(os.path.join(args.model_dir, "train.log"), "a", encoding="utf-8") as f:
            f.write(log + "\n")

        # 保存 checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cv_loss": cv_metrics["loss"],
            "train_loss": train_metrics["loss"],
            "config": vars(args),
        }
        ckpt_path = os.path.join(args.model_dir, f"epoch_{epoch}.pt")
        torch.save(ckpt, ckpt_path)

        # 保存最佳模型
        if cv_metrics["loss"] < best_loss:
            best_loss = cv_metrics["loss"]
            torch.save(ckpt, os.path.join(args.model_dir, "best.pt"))
            print(f"  >>> 新最佳模型: CV loss = {cv_metrics['loss']:.4f}")

    # 保存 final.pt（最后一个 epoch）
    final_path = os.path.join(args.model_dir, "final.pt")
    torch.save(torch.load(ckpt_path), final_path)
    print(f"\n训练完成！最终模型: {final_path}")


if __name__ == "__main__":
    main()
