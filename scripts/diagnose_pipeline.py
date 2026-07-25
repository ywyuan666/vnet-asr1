#!/usr/bin/env python3
"""数据管道诊断：单条过拟合 + 特征多样性检查"""

import json, torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import soundfile as sf
import torchaudio.compliance.kaldi as kaldi
import torch.nn as nn
import torch.nn.functional as F

def main():
    device = torch.device('cuda')

    # ===== 加载 =====
    with open('data/aishell/train/data.list') as f:
        lines = [json.loads(l.strip()) for l in f.readlines()[:3]]

    with open('data/aishell/units.txt') as f:
        vocab = {}
        for line in f:
            p = line.strip().split()
            if len(p) >= 2: vocab[p[0]] = int(p[1])

    with open('data/aishell/global_cmvn') as f:
        cmvn = json.load(f)
    fn = cmvn['frame_num']
    cmvn_mean = torch.tensor(cmvn['mean_stat']) / fn
    cmvn_var = torch.tensor(cmvn['var_stat']) / fn - cmvn_mean * cmvn_mean
    cmvn_std = cmvn_var.sqrt()

    # ===== 1. 特征多样性 =====
    print("=== 1. 特征多样性检查 ===")
    feats = []
    for item in lines:
        data, sr = sf.read(item['wav'])
        wav = torch.from_numpy(data).float().unsqueeze(0) * (1 << 15)
        feat = kaldi.fbank(wav, num_mel_bins=80, frame_length=25, frame_shift=10,
                           dither=0.0, sample_frequency=16000)
        feat = (feat - cmvn_mean) / (cmvn_std + 1e-10)
        feats.append(feat)
        print(f"  {item['key']}: shape={feat.shape}, mean={feat.mean():.4f}, std={feat.std():.4f}")

    for i in range(len(feats)):
        for j in range(i+1, len(feats)):
            # 对齐到 min 长度
            min_len = min(feats[i].size(0), feats[j].size(0))
            diff = (feats[i][:min_len] - feats[j][:min_len]).abs().mean()
            print(f"  diff({i},{j}) = {diff:.4f}  (应为 >> 0.01, 若 < 0.01 则特征完全相同)")

    # ===== 2. 单条过拟合 =====
    print("\n=== 2. 单条过拟合测试 ===")
    item = lines[0]
    data, sr = sf.read(item['wav'])
    wav = torch.from_numpy(data).float().unsqueeze(0) * (1 << 15)
    feat_raw = kaldi.fbank(wav, num_mel_bins=80, frame_length=25, frame_shift=10,
                           dither=0.0, sample_frequency=16000)
    feat_cmvn = (feat_raw - cmvn_mean) / (cmvn_std + 1e-10)
    ids = [vocab.get(ch, 1) for ch in item['txt']]
    print(f"  文本: {item['txt']}")
    print(f"  ids: {ids}, len={len(ids)}")

    feat_t = feat_cmvn.unsqueeze(0).to(device)
    targets = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    target_len = torch.tensor([len(ids)], dtype=torch.long).to(device)

    # 简单模型：Conv1d stride-4 + 1层 BiLSTM + CTC head
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(80, 64, 7, stride=4, padding=3)
            self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
            self.head = nn.Linear(128, len(vocab))

        def forward(self, x):
            x = x.transpose(1, 2)
            x = F.relu(self.conv(x))
            x = x.transpose(1, 2)
            x, _ = self.lstm(x)
            return F.log_softmax(self.head(x), dim=-1)

    model = TestModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    for ep in range(200):
        model.train()
        opt.zero_grad()
        logp = model(feat_t)
        il = torch.tensor([min(feat_t.size(1)//4, logp.size(1))], dtype=torch.long).to(device)
        loss = F.ctc_loss(logp.transpose(0, 1), targets, il, target_len, blank=0, zero_infinity=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if ep % 50 == 0 or ep < 10:
            model.eval()
            with torch.no_grad():
                lp = model(feat_t)
                preds = lp.argmax(dim=-1).squeeze(0)
                tokens = []; prev = -1
                for t in preds.tolist():
                    if t != 0 and t != prev: tokens.append(t)
                    prev = t
                idx2token = {v:k for k,v in vocab.items()}
                hyp = ''.join(idx2token.get(t,'?') for t in tokens)
                print(f"  ep {ep:3d}: loss={loss.item():.2f} | hyp={hyp}")

    # Final
    model.eval()
    with torch.no_grad():
        lp = model(feat_t)
        preds = lp.argmax(dim=-1).squeeze(0)
        tokens = []; prev = -1
        for t in preds.tolist():
            if t != 0 and t != prev: tokens.append(t)
            prev = t
        idx2token = {v:k for k,v in vocab.items()}
        hyp = ''.join(idx2token.get(t,'?') for t in tokens)
        correct = sum(1 for a,b in zip(hyp, item['txt']) if a == b)
        print(f"  最终: {correct}/{len(item['txt'])} 匹配, loss={loss.item():.2f}")

    print("\n=== 诊断结论 ===")
    print("如果单条 200 步能过拟合(字符匹配 >50%): 管道正常, Conformer 架构有隐藏 bug")
    print("如果单条 200 步不能过拟合: 特征有问题, 需要检查 fbank/CMVN")

if __name__ == '__main__':
    main()
