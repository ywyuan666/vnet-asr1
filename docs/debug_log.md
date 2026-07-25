# vnet-asr1 调试日志

> 记录 AISHELL-1 训练过程中的所有 bug 发现、修复和调试过程。
> 训练环境：AutoDL RTX 5090, PyTorch 2.13.0+cu130, torchaudio 2.11.0+cu130

---

## Benchmark 基线

| 模型 | AISHELL-1 test CER | 说明 |
|------|--------------------|------|
| **FunASR Paraformer (large)** | **2.28%** | 百度开源，作为本项目性能上界参考 |
| vnet-asr1 Conformer (d_model=144, max_epoch=30) | 待优化 | 当前训练中 |
| WeNet U2++ Conformer (~46M params) | ~4.5% (公开数据) | 业内标准基线 |

---

## Bug #1：units.txt 格式不匹配

### 现象
训练时 `词表大小: 0, <sos/eos> id: -1`，模型创建失败

### 根因
`local/download_aishell.py` 中 `build_dict()` 输出的 `units.txt` 只有 token 名称（单列）：
```
<blank>
<unk>
一
丁
...
<sos/eos>
```
但 `train.py:332-336` 按 `token id` 两列格式解析：
```python
parts = line.strip().split()
if len(parts) >= 2:
    vocab[parts[0]] = int(parts[1])
```

### 修复
`build_dict()` 改为输出 `<token> <id>` 格式（`local/download_aishell.py:123-131`）：
```python
f.write("<blank> 0\n")
f.write("<unk> 1\n")
idx = 2
for ch in sorted_chars:
    f.write(f"{ch} {idx}\n")
    idx += 1
f.write(f"<sos/eos> {idx}\n")
```

- 提交：`1c5618b`、`f63102f`

---

## Bug #2：Cross-Attention 忽略编码器输出

### 现象
模型对所有输入识别出相同的文本（如"这是一个相当深刻的技术进步"），CER ~100%

### 测试过程
1. 验证编码器输出：两条不同音频的 encoder 输出 cosine 相似度正常，说明 encoder 工作
2. 验证解码器 top5 token：两条不同音频的 top5 token 完全一样
3. 关键测试：给 attention decoder 传入正常 encoder 输出 vs 全零 encoder 输出，两者 top1 token 完全相同 → **cross-attention 完全没用 encoder 信息**

### 根因
`DecoderLayer.forward()` 中 cross-attention 直接接收原始 encoder 输出（mean≈-0.01, std≈0.57），但 decoder 内部经过 LayerNorm（mean≈0, std≈1）。cross-attn 输出 std=0.046，self-attn 输出 std=0.329，**差了 7 倍**。residual connection 中 cross-attn 被 self-attn 淹没。

### 修复
在 `DecoderLayer.__init__` 中添加 `self.norm_enc = nn.LayerNorm(d_model)`，在 cross-attn 前对 encoder memory 做归一化：
```python
memory_norm = self.norm_enc(memory)
y = y + self.dropout(self.cross_attn(self.norm2(y), memory_norm, memory_norm))
```

- 提交：`f63102f`
- 验证：norm_enc 权重确认存在于 checkpoint 中（6 个 key）

---

## Bug #3：Transducer RNN-T loss 数值异常

### 现象
Transducer loss 持续 900-1000（正常随初始化应约 200-300），且 `torchaudio 2.11` 标记 `rnnt_loss` 为 deprecated

### 根因分析
1. `torchaudio.functional.rnnt_loss` 在 2.11 版本中已废弃（将在 2.9 移除）
2. 测试调用时直接报 `output length mismatch` 错误（compute.cpp:81）
3. Joint Network 的 `enc_expanded.unsqueeze(2).expand(...)` 产生 `[B, T, U, D]` 大张量，导致 ~2.6GB 额外显存和 OOM 风险

### 修复
当 `trans_weight=0` 时，完全跳过 Transducer decoder 的前向计算：
```python
if self.trans_weight > 0:
    # 计算 Transducer forward + rnnt_loss
    ...
else:
    transducer_loss = torch.tensor(0.0, device=feats.device)
```

- 提交：`9ac28cb`
- 效果：释放约 2.6GB 显存，训练速度提升

---

## 训练迭代历史

### 第 1 轮（d_model=144, 60 epochs, 无 CMVN/无 SpecAugment）
- CTC loss: 12.5 → 6.2（接近随机 8.1）
- Attention loss: 5.9 → 1.1（学习成功但推理失败——Bug #2）
- CER: >85%（三种模式）

### 第 2 轮（d_model=256, 60 epochs, CMVN + trans_weight=0）
- CTC loss: 6.8 → 6.2（停滞）
- Attention loss: 5.8 → 1.14（train 过拟合，CV=8.0）
- CER: >85%（三种模式）
- 发现 Bug #2（cross-attn 废了）

### 第 3 轮（d_model=256, 60 epochs, CMVN + SpecAugment, 含 norm_enc 修复）
- Enhancement: cross-attn 输出开始随输入变化（不再总输出同一句话）
- Attention loss: 5.8 → 1.06（train），CV 5.5 → 8.1（过拟合）
- 问题仍在曝光偏差导致推理陷入固定模板

### 第 4 轮（d_model=144, 30 epochs, CMVN + SpecAugment + SpeedPerturb, trans_weight=0）
- trans 项不再计算（0.0000），训练速度明显加快
- 最佳 epoch: 6（CV loss=5.60）
- Attention 解码：输出有变化但重复"但是一个小偷打工人的"模板
- CTC 解码：全部 blank 坍塌

---

## 当前状态

| 项目 | 状态 |
|------|------|
| 代码推送到 GitHub | ✅ |
| pytest 23 项全部通过 | ✅ |
| AISHELL-1 数据准备完成 | ✅ |
| FunASR Paraformer benchmark | ✅ 2.28% CER |
| Bug #1-3 修复 | ✅ |
| Bug #4: BatchNorm1d → LayerNorm | ✅ 已修复 |
| Bug #5: ConformerBlock 缺失 dropout | ✅ 已修复 |
| vnet-asr1 模型训练 | 🔧 修复后重新训练中 |

---

## Bug #4：ConvModule 的 BatchNorm1d 导致训练不稳定

### 现象
5 轮训练 CTC loss 始终卡在 6-7（随机 ~8.08），编码器学不到声学特征

### 根因
`ConvModule` 使用 `BatchNorm1d(d_model)`，但 ASR 任务中序列长度差异极大（短则 50 帧，长则 2000 帧）。BatchNorm 的 batch statistics 在变长序列上极不稳定：
- 训练时：每个 step 的 mini-batch 统计方差大，梯度噪音淹没信号
- 推理时：running stats 不匹配任意单条测试音频

Conformer 原始论文使用的是 **LayerNorm**，而非 BatchNorm。

### 修复
`model/conformer_ctc_attn_transducer.py:120-143` — ConvModule 中 `BatchNorm1d → LayerNorm`，需在 forward 中做 transpose（conv 用 (B, D, T)，LayerNorm 用 (B, T, D)）

---

## Bug #5：ConformerBlock 残差连接缺失 Dropout

### 现象
Attention 子层后的残差连接没有 dropout，导致过拟合快

### 修复
`model/conformer_ctc_attn_transducer.py:153-196` — ConformerBlock 新增 `self.dropout`，在 FFN 1/2 块和 Attention 输出后添加 dropout

## 调试经验总结

1. **dict 格式一致性**：数据流水线的每个环节（生成→解析）必须保证格式一致
2. **residual connection 尺度匹配**：pre-norm 架构下，encoder memory 需要与 decoder 内部激活尺度一致
3. **torchaudio API 版本兼容性**：`rnnt_loss` 在 2.11 已废弃，需关注 API 变更
4. **encodec-only 训练稳定性**：CTC 和 Attention 的权重比需要仔细调整，CTC 权重太低会导致声学特征学习不足
5. **baseline 先行**：在调试自研模型前，先跑开源的 SOTA 模型获取 baseline，为后续对比提供可靠参照
