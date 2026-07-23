# Conformer + CTC / Attention / Transducer 三任务联合训练语音识别系统

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/yaweiyuan/vnet-asr1)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ywyuan666/vnet-asr1/blob/main/colab_demo.ipynb)

基于 **Conformer 编码器 + CTC / Attention / Transducer 三解码头** 的端到端语音识别系统。参考 WeNet 架构设计，在一个统一的模型框架中同时训练三个解码器，支持三种解码方式及流式推理。

---

## 特性

| # | 特性 | 说明 |
|---|------|------|
| 1 | **Conformer 编码器** | Transformer 自注意力 + 卷积模块（ConvModule），兼顾全局与局部声学特征 |
| 2 | **三头联合解码** | CTC + Attention + Transducer 同一框架，共享编码器 |
| 3 | **联合损失训练** | 总损失 = 0.3 x CTC + 0.3 x Attention + 0.4 x Transducer |
| 4 | **流式推理** | 支持 CTC 和 Transducer 流式解码，chunk_size 可配置 |
| 5 | **KV Cache 机制** | 流式推理时缓存历史注意力，避免重复计算 |
| 6 | **动态 Chunk 训练** | 训练时随机采样 chunk_size，单一模型支持多种延迟配置 |
| 7 | **TTS 数据生成** | 纯 Python edge-tts 合成，无需系统 ffmpeg 依赖 |
| 8 | **推理加速** | ONNX 导出 + TensorRT FP16/INT8 量化部署 |

---

## 模型架构

```
Fbank(80-dim) -> Conv2d 下采样(4x) -> PositionalEncoding -> 6 x ConformerBlock -> Encoder Memory
     |--- CTC Linear -> CTC Logits -> CTC Loss (weight=0.3)
     |--- Transformer Decoder(3层) -> Attn Logits -> CE Loss (weight=0.3)
     |--- Prediction LSTM(1层) + Joint Linear -> RNNT Logits -> RNN-T Loss (weight=0.4)
```

### Conformer Block (Macaron Structure)

```
x = x + 1/2 x FFN(LayerNorm(x))
x = x + MHSA(LayerNorm(x))
x = x + ConvModule(LayerNorm(x))
x = x + 1/2 x FFN(LayerNorm(x))
x = LayerNorm(x)
```

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| d_model | 144 | 模型维度 |
| n_head | 4 | 注意力头数 |
| d_ff | 1024 | 前馈网络隐藏维 |
| encoder_blocks | 6 | Conformer 块数 |
| decoder_blocks | 3 | Attention 解码器层数 |
| vocab_size | 35 | 词表大小（含 blank/unk/sos/eos） |
| 总参数量 | ~6.6M | 模型总参数量 |
| streaming_prob | 0.5 | 流式训练概率 |

---

## 性能结果

### TTS 合成语音指令测试集（300 条）

| 解码模式 | CER | 精确匹配率 | 说明 |
|---------|-----|-----------|------|
| **Attention** | **0.87%** | **96.67%** | 仅 1 个错误 |
| Transducer Greedy | 9.57% | 76.67% | 末尾字符偶有丢失 |
| CTC Greedy | 25.22% | 10.00% | 需 beam search 改善 |

### 推理速度基准

| 后端 | 精度 | 相对速度 | 模型大小 |
|------|------|---------|----------|
| PyTorch | FP32 | 1x | ~26 MB |
| ONNX | FP32 | 1.5 - 2x | ~13 MB |
| TensorRT | FP16 | 3 - 5x | ~7 MB |
| TensorRT | INT8 | 5 - 8x | ~4 MB |

---

## 快速开始

### 环境准备

```bash
# 创建虚拟环境
python -m venv .venv
# 或 conda
conda create -n vnet-asr python=3.10
conda activate vnet-asr

# 安装依赖
pip install -r requirements_ctc_attn_transducer.txt
```

### 一键运行

```powershell
# Windows PowerShell - 3 分钟体验完整流程
.\run_ctc_attn_transducer.ps1
```

脚本按阶段依次执行：

| Stage | 内容 |
| --- | --- |
| 0 | 用 edge-tts 合成语音指令数据集（300 条） |
| 1 | 准备 data.list 与字典 units.txt |
| 2 | 计算 CMVN 特征归一化统计量 |
| 3 | 训练 Conformer + CTC/Attention/Transducer 模型（200 epochs） |
| 4 | 在测试集上解码并计算 CER（3 种模式） |
| 5 | 单条音频识别演示 |

### 分步运行

#### 生成数据集

```bash
python local/generate_corpus_ctc_attn_transducer.py --out data/audio --repeat 5
```

#### 准备数据

```bash
python local/prepare_data.py --audio_dir data/audio --out_dir data
```

#### 计算 CMVN

```bash
python tools/make_cmvn.py --data_list data/train/data.list --out data/train/global_cmvn
```

#### 训练

```bash
python train.py \
  --train_data data/train/data.list \
  --cv_data data/dev/data.list \
  --dict data/dict/units.txt \
  --cmvn data/train/global_cmvn \
  --model_dir exp/conformer_ctc_attn_transducer \
  --batch_size 16 \
  --max_epoch 200 \
  --device cpu \
  --d_model 144 \
  --ctc_weight 0.3 \
  --attn_weight 0.3 \
  --trans_weight 0.4
```

#### 解码评测

```bash
python recognize_ctc_attn_transducer.py \
  --checkpoint exp/conformer_ctc_attn_transducer/best.pt \
  --test_data data/test/data.list \
  --dict data/dict/units.txt \
  --cmvn data/train/global_cmvn \
  --device cpu \
  --mode all
```

`--mode` 可选：`ctc_greedy`、`attention`、`transducer`、`all`

#### 单条识别

```bash
python infer_demo_ctc_attn_transducer.py \
  --checkpoint exp/conformer_ctc_attn_transducer/best.pt \
  --dict data/dict/units.txt \
  --cmvn data/train/global_cmvn \
  --wav some.wav \
  --mode attention
```

---

## 流式推理

### 流式 CTC 解码

```bash
python infer_demo_ctc_attn_transducer.py \
  --checkpoint exp/conformer_ctc_attn_transducer/best.pt \
  --dict data/dict/units.txt \
  --cmvn data/train/global_cmvn \
  --wav some.wav \
  --mode ctc_streaming \
  --chunk_size 16 \
  --right_context 4
```

### 流式 Transducer 解码

```bash
python infer_demo_ctc_attn_transducer.py \
  --checkpoint exp/conformer_ctc_attn_transducer/best.pt \
  --dict data/dict/units.txt \
  --cmvn data/train/global_cmvn \
  --wav some.wav \
  --mode transducer_streaming \
  --chunk_size 16 \
  --right_context 4
```

### 流式参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| chunk_size | 16 | 每 chunk 帧数（160ms），越小延迟越低 |
| right_context | 4 | 右侧上下文帧数（40ms），增大可恢复精度 |

---

## 推理加速与部署

### ONNX 导出

```bash
python tools/export_tensorrt.py \
  --checkpoint exp/conformer_ctc_attn_transducer/best.pt \
  --dict data/dict/units.txt \
  --onnx_only
```

导出内容：
- `model.onnx` - 编码器 ONNX
- `model_decoder.onnx` - Attention 解码器 ONNX
- `model_ctc.onnx` - CTC 头 ONNX

### TensorRT 导出

```bash
# FP16
python tools/export_tensorrt.py \
  --checkpoint exp/conformer_ctc_attn_transducer/best.pt \
  --dict data/dict/units.txt \
  --fp16 \
  --benchmark

# INT8
python tools/export_tensorrt.py \
  --checkpoint exp/conformer_ctc_attn_transducer/best.pt \
  --dict data/dict/units.txt \
  --int8 \
  --calib_data data/train/data.list \
  --benchmark
```

---

## 三种解码模式对比

| 解码模式 | 算法 | 优点 | 缺点 | 流式支持 |
|---------|------|------|------|----------|
| **Attention** | Transformer 自回归解码 | 精度最高 | 计算量较大 | 否 |
| **Transducer** | Prediction + Joint Network | 流式友好，精度适中 | 易丢尾字 | 是 |
| **CTC Greedy** | 帧级 argmax + 去重 | 速度最快 | 精度最低 | 是 |

---

## 基准对比

### vs WeNet

| 指标 | WeNet (Conformer) | vnet-asr1 | 说明 |
|------|-------------------|-----------|------|
| 编码器层数 | 12 | 6 | vnet-asr1 更轻量 |
| d_model | 256 | 144 | vnet-asr1 更小 |
| 参数量 | ~46M | ~6.6M | vnet-asr1 约 1/7 |
| 解码方式 | CTC + Attn 或 CTC + Trans | CTC + Attn + Trans | vnet-asr1 三合一 |
| 流式支持 | 是 (U2++) | 是 | 均支持 |

### 三种解码模式速度对比

| 模式 | 相对速度 | 适用场景 |
|------|---------|----------|
| CTC Greedy | 最快 | 实时低延迟，粗粒度假设 |
| Transducer | 适中 | 流式高精度 |
| Attention | 最慢 | 离线批处理，最高精度 |

---

## Hugging Face / Colab 在线体验

| 平台 | 链接 | 费用 |
|------|------|------|
| **Hugging Face Model Hub** | [yaweiyuan/vnet-asr1](https://huggingface.co/yaweiyuan/vnet-asr1) | 免费 |
| **Google Colab**（推荐） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ywyuan666/vnet-asr1/blob/main/colab_demo.ipynb) | 免费 |
| **GitHub 仓库** | [ywyuan666/vnet-asr1](https://github.com/ywyuan666/vnet-asr1) | 免费 |

---

## 目录结构

```
vnet-asr1/
├── model/                              # 模型定义
│   ├── conformer_ctc_attn_transducer.py  # 核心模型 (Conformer + 3解码头)
│   ├── streaming_helper.py               # 流式推理工具 (KV cache)
│   └── augmentation.py                   # 数据增强 (SpecAugment)
├── train.py                            # 训练脚本
├── recognize_ctc_attn_transducer.py    # 批量解码评测 (3种模式)
├── infer_demo_ctc_attn_transducer.py   # 单条推理演示
├── local/                              # 数据处理
│   ├── generate_corpus...              # edge-tts 合成语音指令
│   ├── download_aishell.py             # AISHELL-1 下载
│   └── prepare_data.py                 # WeNet 格式 data.list
├── tools/                              # 工具
│   ├── make_cmvn.py                    # CMVN 计算
│   ├── train_kenlm.py                  # LM 训练
│   ├── lm_rescore.py                   # LM 重打分
│   └── export_tensorrt.py              # TensorRT 导出
├── scripts/                            # 实验脚本
│   ├── ablation_study.py               # 消融实验
│   └── benchmark_baselines.py          # 基准对比
├── docs/                               # 文档
│   ├── system_design.md                # 系统设计
│   └── quick_start.md                  # 快速入门
├── conf/                               # 配置文件
├── server/                             # 推理服务
├── hf_space/                           # Hugging Face Space 部署
├── Dockerfile                          # Docker 部署
├── docker-compose.yml                  # Docker Compose
└── 实验报告.md                          # 完整实验报告
```

---

## 模型文件说明

- `model/conformer_ctc_attn_transducer.py` — 核心模型实现，包含：
  - `ConformerEncoder`（ConformerBlock x 6）
  - `MultiHeadSelfAttention` / `MultiHeadCrossAttention`（支持 KV cache）
  - `ConvModule`（逐深度卷积 + GLU）
  - `AttentionDecoder`（Transformer 解码器 x 3）
  - `TransducerDecoder`（Prediction LSTM + Joint Network）
  - `ConformerCTCATTNTransducer`（统一模型类）

- `train.py` — 自定义训练脚本（非 WeNet 框架）：
  - Adam 优化器 + WarmupLR 学习率调度
  - CTC / CrossEntropy / RNN-T 三损失联合训练
  - 动态流式训练（streaming_prob=0.5）
  - Checkpoint 保存（最佳模型 + 每 epoch 保存）
  - SpecAugment 数据增强

---

## 常见问题

- **训练时 CTC 损失不变 / 输出固定结果？** 检查 CMVN 是否正确计算（mean_stat 需除以 frame_num）。
- **Attention 解码始终输出相同文字？** 数据量不足，建议调大 `--repeat`。
- **Transducer 解码丢尾字？** 贪心算法的固有问题，可改用 beam search。
- **没有 GPU？** CPU 也能训练，调小 d_model（如 96）可加速。
- **流式推理和非流式结果不一致？** 确保训练时启用了 streaming_prob > 0。

---

## 参考

- WeNet: https://github.com/wenet-e2e/wenet
- Conformer: https://arxiv.org/abs/2005.08100
- RNN-Transducer: https://arxiv.org/abs/1211.3711
- SpecAugment: https://arxiv.org/abs/1904.08779
- TensorRT: https://developer.nvidia.com/tensorrt
