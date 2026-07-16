# -*- coding: utf-8 -*-
"""测试模型架构"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from model.conformer_ctc_attn_transducer import ConformerCTCATTNTransducer

model = ConformerCTCATTNTransducer(vocab_size=20)
B, T = 2, 100
feats = torch.randn(B, T, 80)
feat_lens = torch.tensor([T, T])
attn_in = torch.randint(1, 19, (B, 6))
attn_in[:, 0] = 19
attn_out = torch.randint(1, 19, (B, 6))
attn_out[:, -1] = 19
trans_in = torch.randint(1, 19, (B, 6))
trans_in[:, 0] = 19
trans_lens = torch.tensor([5, 6])

outputs = model(feats, feat_lens, attn_in, attn_out, trans_in, trans_lens)
print("=== 前向测试 ===")
for k, v in outputs.items():
    print(f"  {k}: {v.item():.4f}" if v.numel() == 1 else f"  {k}: {v}")

loss = outputs["loss"]
loss.backward()
print("反向传播: OK")

# 测试解码
idx2token = {i: f"T{i}" for i in range(20)}
ctc_result = model.recognize_ctc_greedy(torch.randn(1, 100, 80), idx2token)
print(f"CTC 解码: {ctc_result}")

n_params = sum(p.numel() for p in model.parameters())
print(f"\n参数量: {n_params:,}")
print("全部测试通过")
