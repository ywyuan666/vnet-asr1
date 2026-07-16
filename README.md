# Conformer + CTC / Attention / Transducer 三任务联合训练语音识别系统

基于 **Conformer 编码器 + CTC / Attention / Transducer 三解码头** 的端到端语音识别系统。参考 WeNet 架构设计，在一个统一的模型框架中同时训练三个解码器，支持三种解码方式。

## 技术特性

| 特性 | 说明 |
| --- | --- |
| **Conformer 编码器** | Transformer 自注意力 + 卷积模块（ConvModule），兼顾全局与局部声学特征 |
| **CTC 解码头** | Connectionist Temporal Classification，非自回归一遍解码，帧级对齐 |
| **Attention 解码头** | Transformer 解码器，自回归生成文字，精度最高 |
| **Transducer 解码头** | RNN-T 架构（Prediction LSTM + Joint Network），流式友好 |
| **三任务联合训练** | 总损失 = 0.3×CTC + 0.3×Attention + 0.4×Transducer |
| **edge-tts 数据生成** | 纯 Python TTS 合成，无需系统 ffmpeg 依赖 |

## 模型架构

```
Fbank(80-dim) → Conv2d下采样(4x) → PositionalEncoding → 6×ConformerBlock → Encoder Memory
     ├── CTC Linear → CTC Logits → CTC Loss (权重0.3)
     ├── Transformer Decoder(3层) → Attn Logits → CE Loss (权重0.3)
     └── Prediction LSTM(1层) + Joint Linear → RNNT Logits → RNN-T Loss (权重0.4)
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

## 评测结果

在 300 条 TTS 合成语音指令测试集上：

| 解码模式 | CER | 精确匹配率 | 说明 |
|---------|-----|-----------|------|
| **Attention** | **0.87%** | **96.67%** | 仅 1 个错误（"下一首"→"上一首"） |
| Transducer Greedy | 9.57% | 76.67% | 末尾字符偶有丢失 |
| CTC Greedy | 25.22% | 10.00% | 字符重复/缺失，需 beam search 改善 |

## 目录结构

```
conformer-ctc-attn-transducer/
├── README.md
├── requirements_ctc_attn_transducer.txt    # Python 依赖
├── conf/
│   └── train_conformer_ctc_attn_transducer.yaml   # 训练配置
├── local/
│   ├── generate_corpus_ctc_attn_transducer.py    # edge-tts 合成语音指令数据集
│   └── prepare_data.py                # 生成 WeNet 格式 data.list 与字典
├── tools/
│   └── make_cmvn.py                   # 计算 CMVN 特征归一化统计量
├── model/
│   └── conformer_ctc_attn_transducer.py   # 核心模型（Conformer + 3个解码头）
├── train.py                            # 自定义训练脚本（非 WeNet 框架）
├── recognize_ctc_attn_transducer.py    # 批量解码测试集并计算 CER（3种模式）
├── infer_demo_ctc_attn_transducer.py   # 单条音频识别演示
├── run_ctc_attn_transducer.ps1         # Windows 一键 Pipeline（PowerShell）
├── debug_transducer.py                 # Transducer 调试脚本
├── test_model.py                       # 模型前向验证
└── 实验报告.md                          # 完整实验报告
```

## 环境准备

### 1. Python 与 PyTorch

- Python 3.9 ~ 3.11
- CPU 即可运行（数据集较小）；有 NVIDIA 显卡可显著加速
- 建议虚拟环境：`python -m venv .venv`

### 2. 安装依赖

```bash
pip install -r requirements_ctc_attn_transducer.txt
```

核心依赖：
- `torch>=2.0.0`、`torchaudio>=2.0.0`
- `edge-tts>=6.1.0`（TTS 语音合成）
- `miniaudio`、`soundfile`（纯 Python 音频编解码，无需 ffmpeg）
- `librosa`（音频重采样）

## 快速开始

### 一键运行

```powershell
# Windows PowerShell
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

- 12 条中文耳机语音指令 × 5 种发音人 × 5 轮 = 300 条音频
- 调大 `--repeat` 可扩充数据规模

#### 准备数据

```bash
python local/prepare_data.py --audio_dir data/audio --out_dir data
```

产出：`data/{train,dev,test}/data.list` 和 `data/dict/units.txt`

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

`--mode` 可选：`ctc_greedy`、`attention`、`transducer`、`all`（三种模式一起评测）

#### 单条识别

```bash
python infer_demo_ctc_attn_transducer.py \
  --checkpoint exp/conformer_ctc_attn_transducer/best.pt \
  --dict data/dict/units.txt \
  --cmvn data/train/global_cmvn \
  --wav some.wav \
  --mode attention
```

## 三种解码模式对比

| 解码模式 | 算法 | 优点 | 缺点 |
|---------|------|------|------|
| **Attention** | Transformer 自回归解码 | 精度最高，含语言模型信息 | 计算量较大，无法流式 |
| **Transducer** | Prediction Network + Joint Network | 流式友好，精度适中 | 贪心解码易丢尾字 |
| **CTC Greedy** | 帧级 argmax + 去重 | 速度最快 | 精度最低，需 beam search |

## 模型文件说明

- `model/conformer_ctc_attn_transducer.py` — 核心模型实现，包含：
  - `ConformerEncoder`（ConformerBlock × 6）
  - `MultiHeadSelfAttention` / `MultiHeadCrossAttention`
  - `ConvModule`（逐深度卷积 + GLU）
  - `AttentionDecoder`（Transformer 解码器 × 3）
  - `TransducerDecoder`（Prediction LSTM + Joint Network）
  - `ConformerCTCATTNTransducer`（统一模型类）

- `train.py` — 自定义训练脚本（非 WeNet 框架）：
  - Adam 优化器 + WarmupLR 学习率调度
  - CTC / CrossEntropy / RNN-T 三损失联合训练
  - Checkpoint 保存（最佳模型 + 每 epoch 保存）
  - CMVN 特征归一化（已修复累加和 bug）

## 常见问题

- **训练时 CTC 损失不变 / 输出固定结果？** 检查 CMVN 是否正确计算（`mean_stat` 需除以 `frame_num`）。
- **Attention 解码始终输出相同文字？** 数据量不足，建议调大 `--repeat`。
- **Transducer 解码丢尾字？** 贪心算法的固有问题，可改用 beam search。
- **没有 GPU？** CPU 也能训练，调小 `d_model`（如 96）可加速。

## 参考

- WeNet: https://github.com/wenet-e2e/wenet
- Conformer: https://arxiv.org/abs/2005.08100
- RNN-Transducer: https://arxiv.org/abs/1211.3711
