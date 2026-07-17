---
title: VNet ASR
emoji: 🎤
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
---

# VNet ASR — Conformer + CTC/Attention/Transducer 语音识别

基于 Conformer 编码器 + 联合 CTC/Attention/Transducer 解码的语音识别演示。

## 三种解码模式

| 模式 | 特点 |
|------|------|
| **Attention** | 自回归解码，精度最高 (CER=0.87%) |
| **Transducer** | 流式友好，适合低延迟场景 |
| **CTC Greedy** | 非自回归，速度最快 |

## 模型信息

- 编码器: 6× Conformer Block, d_model=144, 4 heads
- 训练数据: 300条 TTS 合成中文语音命令
- 词汇: 34个中文汉字 + 特殊标记
