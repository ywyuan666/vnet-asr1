# 安克耳机语音指令识别系统（WeNet · U2++ Conformer）

> 一份**小白友好**、可**一键跑通**的端到端中文语音识别大作业 / 课程项目。
>
> 主题：为「安克（Anker）耳机」做一个**语音助手指令识别**模型，能听懂诸如
> 「下一首」「打开降噪」「接听电话」「调高音量」等中文语音命令。
>
> 模型：**WeNet 框架 + U2++ 架构**（Conformer 编码器 + CTC/Attention 联合训练 +
> **双向注意力解码器**（bitransformer）+ Attention Rescoring 重打分），支持流式 / 非流式统一识别。

---

## 0. 这个项目能学到什么（写进报告正好）

| 关键词 | 在本项目里对应什么 |
| --- | --- |
| **端到端 ASR** | 直接「语音波形 → 汉字」，不需要传统的 HMM-GMM、发音词典 |
| **Conformer** | 编码器 = Transformer 自注意力 + 卷积，兼顾全局与局部特征 |
| **CTC** | 一条「快但糙」的解码分支，负责对齐与一遍解码 |
| **Attention Decoder** | 一条「慢但准」的解码分支，自回归生成文字 |
| **U2++ / 双向解码器** | 同时训练**从左到右**和**从右到左**两个解码器，识别更准 |
| **Attention Rescoring** | 先用 CTC 出 N-best，再用注意力解码器重新打分挑最优 |
| **流式 / 非流式统一** | 同一个模型，既能边说边出字，也能整句一次性识别 |
| **数据增强** | 速度扰动（speed perturb）+ 频谱掩蔽（SpecAugment） |

---

## 1. 目录结构

```
anker-asr-wenet/
├── README.md                       # 你正在看的总教程
├── requirements.txt                # Python 依赖
├── conf/
│   └── train_u2pp_conformer.yaml   # ⭐ U2++ Conformer 训练配置（核心）
├── local/
│   ├── generate_anker_corpus.py    # ⭐ 用 TTS 自动合成「安克指令」语音数据集
│   └── prepare_data.py             # 生成 WeNet 需要的 data.list 与字典
├── tools/
│   └── make_cmvn.py                # 计算特征归一化(CMVN)统计量
├── model/
│   └── u2pp_conformer_min.py       # ⭐ 教学版：从零实现的精简 U2++（看懂原理用）
├── run.ps1                         # ⭐ Windows 一键脚本（PowerShell）
├── run.sh                          # Linux / WSL / Mac 一键脚本
├── recognize.py                    # 用训练好的模型批量解码测试集
├── infer_demo.py                   # ⭐ 单条音频 / 麦克风实时识别 demo
└── docs/
    ├── 原理讲解.md                 # Conformer / CTC / U2++ 原理详解（报告素材）
    └── 答辩大纲.md                 # 报告 / PPT / 答辩问答提纲
```

> 带 ⭐ 的是你最该看、答辩最可能被问到的文件。

---

## 2. 环境准备（按顺序做一遍即可）

### 2.1 安装 Python 与 PyTorch

- 推荐 **Python 3.9 ~ 3.11**。
- 没有显卡也能跑（数据集很小，CPU 训练几十分钟到一两小时）；有 NVIDIA 显卡会快很多。

### 2.2 安装 ffmpeg（用于把合成音频转成 16k 单声道 wav）

- **Windows**：用 winget 安装最简单（在 PowerShell 里执行）：
  ```powershell
  winget install --id=Gyan.FFmpeg -e
  ```
  装完**重开一个终端**，输入 `ffmpeg -version` 能出版本号即成功。
- **Linux**：`sudo apt install ffmpeg`
- **Mac**：`brew install ffmpeg`

### 2.3 安装本项目依赖

```powershell
# 进入项目目录
cd C:\Users\ywyuan\Desktop\anker-asr-wenet

# （可选但强烈推荐）建一个虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

> `requirements.txt` 里包含 `wenet`（WeNet 的训练/推理库）、`torch`、`torchaudio`、
> `edge-tts`（语音合成）、`soundfile` 等。

### 2.4 安装 WeNet 训练代码（重要）

WeNet 的「**推理库**」用 `pip install wenet` 就有；但「**训练脚本**」(`wenet.bin.train`) 需要 GitHub 上的完整代码。两种方式二选一：

**方式 A（推荐）：clone 官方仓库到本项目同级目录**
```powershell
cd C:\Users\ywyuan\Desktop
git clone https://github.com/wenet-e2e/wenet.git
cd wenet
pip install -e .          # 以「可编辑」方式安装，wenet 命令即可用
```

**方式 B：直接 pip 装（部分版本含训练入口）**
```powershell
pip install wenet
```

安装是否成功，用这条命令验证（能打印帮助即成功）：
```powershell
python -m wenet.bin.train --help
```

> ⚠️ WeNet 不同版本配置字段会有细微差别。本项目的 `conf/train_u2pp_conformer.yaml`
> 按主流版本编写；若你的版本报「unexpected key」之类的错，按报错把对应字段删/改即可，
> `docs/原理讲解.md` 里有逐项解释。Windows 上训练建议用 **WSL** 或直接在 Linux，
> 但小数据集在 Windows 原生 CPU 也能跑通。

---

## 3. 一键跑通（最省心）

Windows（PowerShell）：
```powershell
cd C:\Users\ywyuan\Desktop\anker-asr-wenet
.\run.ps1
```

Linux / Mac / WSL：
```bash
cd anker-asr-wenet
bash run.sh
```

`run` 脚本会按阶段（stage）依次执行，你也可以分步跑（见脚本顶部说明）：

| Stage | 做什么 |
| --- | --- |
| 0 | 用 TTS 合成「安克指令」语音数据集（生成 wav） |
| 1 | 准备 WeNet 数据清单 `data.list`、生成字典 `units.txt` |
| 2 | 计算 CMVN（特征均值方差，用于归一化） |
| 3 | **训练 U2++ Conformer 模型** |
| 4 | 在测试集上解码并计算字错误率 CER |
| 5 | 导出/演示：单条音频识别 |

---

## 4. 分步详解

### Stage 0：生成数据集
```powershell
python local/generate_anker_corpus.py --out data/audio --repeat 1
```
- 它会用多个**不同发音人**把每条安克指令合成成 wav（16k、单声道）。
- 指令清单写在脚本顶部的 `ANKER_COMMANDS`，你可以**自由增删**（增删后字典会自动更新）。
- 想要更多数据：把 `--repeat` 调大，会叠加不同语速/音调的版本。

### Stage 1：准备 WeNet 数据
```powershell
python local/prepare_data.py --audio_dir data/audio --out_dir data
```
产出：
- `data/train/data.list`、`data/dev/data.list`、`data/test/data.list`
  —— 每行一个 JSON：`{"key": "...", "wav": "...", "txt": "下一首"}`
- `data/dict/units.txt` —— 建模单元（汉字）字典，含 `<blank> <unk> <sos/eos>`

### Stage 2：CMVN
```powershell
python tools/make_cmvn.py --data_list data/train/data.list --out data/train/global_cmvn
```

### Stage 3：训练
```powershell
python -m wenet.bin.train `
  --config conf/train_u2pp_conformer.yaml `
  --data_type raw `
  --train_data data/train/data.list `
  --cv_data data/dev/data.list `
  --model_dir exp/u2pp_conformer `
  --cmvn data/train/global_cmvn `
  --num_workers 2
```
训练日志、每个 epoch 的模型 (`*.pt`) 会保存在 `exp/u2pp_conformer/`。

### Stage 4：解码评测
```powershell
python recognize.py `
  --config exp/u2pp_conformer/train.yaml `
  --checkpoint exp/u2pp_conformer/final.pt `
  --test_data data/test/data.list `
  --dict data/dict/units.txt `
  --mode attention_rescoring
```
会打印每条识别结果，并算出 **CER（字错误率）**。`--mode` 可选：
`ctc_greedy_search` / `ctc_prefix_beam_search` / `attention` / `attention_rescoring`（最准）。

### Stage 5：实时 / 单条识别 demo
```powershell
# 识别一个 wav 文件
python infer_demo.py --checkpoint exp/u2pp_conformer/final.pt --dict data/dict/units.txt --wav some.wav

# 用麦克风录 3 秒并识别（需要麦克风 + sounddevice）
python infer_demo.py --checkpoint exp/u2pp_conformer/final.pt --dict data/dict/units.txt --mic 3
```

---

## 5. 常见问题（小白必看）

- **Q：没有 GPU 能训练吗？** 能。数据集很小，CPU 也能跑完，只是慢一点。配置里 batch 已调小。
- **Q：CER 偏高 / 识别不准？** 这是合成数据 + 小模型的正常现象。提升办法：把 `--repeat` 调大造更多数据、增加 `max_epoch`、或减少指令数量让任务更简单。报告里如实写「受数据规模限制」即可，思路正确才是重点。
- **Q：报「找不到 wenet.bin.train」？** 回到 2.4，用方式 A 把 WeNet 仓库 clone 下来 `pip install -e .`。
- **Q：报 ffmpeg 相关错误？** 回到 2.2 装 ffmpeg 并重开终端。
- **Q：配置 yaml 报字段错误？** 不同 WeNet 版本字段不同，按报错提示删/改对应行；核心结构（conformer/bitransformer/ctc）不要动。

---

## 6. 写报告 / 答辩怎么用本项目

1. **系统框图**：麦克风/音频 → Fbank 特征 → Conformer 编码器 → (CTC 分支 + 双向 Attention 解码器) → Attention Rescoring → 文字。
2. **创新点/技术点**：U2++ 双向解码器、流式非流式统一、Attention Rescoring、数据增强。
3. **实验**：不同解码模式（ctc_greedy vs attention_rescoring）的 CER 对比表。
4. 原理细节直接引用 `docs/原理讲解.md`，PPT 结构用 `docs/答辩大纲.md`。

祝大作业顺利！🎧
