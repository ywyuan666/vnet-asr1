# 耳机语音指令识别系统（WeNet · U2++ Conformer）

面向蓝牙耳机场景的端到端中文语音指令识别系统，能够识别「下一首」「打开降噪」「接听电话」「调高音量」等中文语音命令，并据此触发耳机功能。

系统基于 **WeNet 框架 + U2++ 架构** 实现：Conformer 编码器 + CTC/Attention 联合训练 + 双向注意力解码器（bitransformer）+ Attention Rescoring 重打分，支持流式 / 非流式统一识别。

## 技术特性

| 特性 | 说明 |
| --- | --- |
| **端到端 ASR** | 直接「语音波形 → 汉字」，无需传统 HMM-GMM 与发音词典 |
| **Conformer 编码器** | Transformer 自注意力 + 卷积，兼顾全局与局部声学特征 |
| **CTC 分支** | 非自回归一遍解码，负责帧级对齐，速度快 |
| **Attention 解码器** | 自回归生成文字，精度高 |
| **U2++ 双向解码器** | 同时建模从左到右与从右到左两个方向，提升边界识别准确率 |
| **Attention Rescoring** | CTC 输出 N-best，再由注意力解码器重打分取最优 |
| **流式 / 非流式统一** | 同一模型既能边说边出字，也能整句一次性识别 |
| **数据增强** | 速度扰动（speed perturb）+ 频谱掩蔽（SpecAugment） |

## 目录结构

```
earphone-asr-wenet/
├── README.md
├── requirements.txt                # Python 依赖
├── conf/
│   └── train_u2pp_conformer.yaml   # U2++ Conformer 训练配置
├── local/
│   ├── generate_corpus.py          # 用 TTS 合成语音指令数据集
│   └── prepare_data.py             # 生成 WeNet 所需的 data.list 与字典
├── tools/
│   └── make_cmvn.py                # 计算特征归一化（CMVN）统计量
├── model/
│   └── u2pp_conformer_min.py       # 精简版 U2++ 结构实现（用于结构验证）
├── run.ps1                         # Windows 一键脚本（PowerShell）
├── run.sh                          # Linux / WSL / Mac 一键脚本
├── recognize.py                    # 批量解码测试集并计算 CER
├── infer_demo.py                   # 单条音频 / 麦克风实时识别
├── server/                         # Web Demo（FastAPI 后端 + 网页前端）
│   ├── app.py                      # 推理服务：/api/recognize、/api/health
│   └── static/index.html           # 网页：上传/录音 → 实时显示识别结果
├── Dockerfile                      # 容器镜像构建
├── docker-compose.yml              # 一键编排（构建+挂载模型+端口映射）
└── start_demo.sh                   # 一键启动 Demo（docker / local）
```

## 环境准备

### 1. Python 与 PyTorch

- 推荐 Python 3.9 ~ 3.11。
- CPU 即可运行（数据集较小）；有 NVIDIA 显卡可显著加速。

### 2. ffmpeg

用于将合成音频转换为 16k 单声道 wav：

- **Windows**：`winget install --id=Gyan.FFmpeg -e`（安装后重开终端）
- **Linux**：`sudo apt install ffmpeg`
- **Mac**：`brew install ffmpeg`

### 3. 安装依赖

```bash
# （推荐）创建虚拟环境
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 4. 安装 WeNet 训练代码

`pip install wenet` 提供推理库，但训练入口 `wenet.bin.train` 需要完整源码：

```bash
git clone https://github.com/wenet-e2e/wenet.git
cd wenet
pip install -e .
```

验证安装：

```bash
python -m wenet.bin.train --help
```

> 不同 WeNet 版本的配置字段略有差异。若运行时报 `unexpected key` 等错误，按提示调整 `conf/train_u2pp_conformer.yaml` 中的对应字段即可，核心结构（conformer / bitransformer / ctc）保持不变。

## 快速开始

一键运行完整流程：

```bash
# Windows
.\run.ps1

# Linux / Mac / WSL
bash run.sh
```

脚本按阶段（stage）依次执行，也可分步运行（见脚本顶部说明）：

| Stage | 内容 |
| --- | --- |
| 0 | 用 TTS 合成语音指令数据集 |
| 1 | 准备 WeNet 数据清单 `data.list` 与字典 `units.txt` |
| 2 | 计算 CMVN 特征归一化统计量 |
| 3 | 训练 U2++ Conformer 模型 |
| 4 | 在测试集上解码并计算字错误率 CER |
| 5 | 单条音频识别演示 |

## 分步说明

### 生成数据集

```bash
python local/generate_corpus.py --out data/audio --repeat 1
```

- 使用多个不同发音人将每条指令合成为 16k 单声道 wav。
- 指令清单定义在脚本顶部的 `COMMANDS`，可自由增删（字典会随之更新）。
- 调大 `--repeat` 可叠加不同语速 / 音调的样本，扩充数据规模。

### 准备 WeNet 数据

```bash
python local/prepare_data.py --audio_dir data/audio --out_dir data
```

产出：

- `data/{train,dev,test}/data.list`，每行一个 JSON：`{"key": "...", "wav": "...", "txt": "下一首"}`
- `data/dict/units.txt`，建模单元（汉字）字典，含 `<blank> <unk> <sos/eos>`

### 计算 CMVN

```bash
python tools/make_cmvn.py --data_list data/train/data.list --out data/train/global_cmvn
```

### 训练

```bash
python -m wenet.bin.train \
  --config conf/train_u2pp_conformer.yaml \
  --device cpu \
  --data_type raw \
  --train_data data/train/data.list \
  --cv_data data/dev/data.list \
  --model_dir exp/u2pp_conformer \
  --num_workers 2
```

训练日志与每个 epoch 的模型（`*.pt`）保存在 `exp/u2pp_conformer/`。字典路径和 CMVN 路径已经写在 YAML 的 `tokenizer_conf` / `cmvn_conf` 中，不再作为命令行参数传入。

### 解码评测

```bash
python recognize.py \
  --config exp/u2pp_conformer/train.yaml \
  --checkpoint exp/u2pp_conformer/final.pt \
  --test_data data/test/data.list \
  --dict data/dict/units.txt \
  --mode attention_rescoring
```

输出每条识别结果并计算 **CER（字错误率）**。`--mode` 可选 `ctc_greedy_search` / `ctc_prefix_beam_search` / `attention` / `attention_rescoring`（精度最高）。

### 单条 / 实时识别

```bash
# 识别 wav 文件
python infer_demo.py --checkpoint exp/u2pp_conformer/final.pt --dict data/dict/units.txt --wav some.wav

# 麦克风录音 3 秒并识别
python infer_demo.py --checkpoint exp/u2pp_conformer/final.pt --dict data/dict/units.txt --mic 3
```

## Web Demo（网页版实时识别）

提供一个网页界面：**上传音频或点麦克风录音，实时显示识别结果**，底层是真实模型推理。

前置条件：已完成训练（存在 `exp/u2pp_conformer/final.pt`）。若尚未训练，先运行 `bash run.sh`。

### 方式一：Docker（推荐）

```bash
# 构建镜像并后台启动，浏览器打开 http://localhost:8000
docker compose up -d --build

docker compose logs -f      # 查看日志
docker compose down         # 停止
```

模型通过 volume 挂载（`exp/`、`data/dict/`），换模型无需重建镜像。

### 方式二：本机 Python

```bash
pip install fastapi "uvicorn[standard]" python-multipart
uvicorn server.app:app --host 0.0.0.0 --port 8000
# 或： ./start_demo.sh local
```

### 接口

| 接口 | 方法 | 说明 |
| --- | --- | --- |
| `/` | GET | 网页前端 |
| `/api/recognize` | POST | 上传音频（form-data 字段 `audio`），返回识别文字 JSON |
| `/api/health` | GET | 健康检查，返回模型是否就绪 |

## 系统架构

```
麦克风 / 音频
   │  Fbank 特征提取 + CMVN
   ▼
Conformer 编码器
   │
   ├──► CTC 分支（快速一遍解码）
   └──► 双向 Attention 解码器（L2R + R2L）
   │
   ▼
Attention Rescoring（CTC 出 N-best，注意力解码器重打分）
   ▼
识别文字
```

## 常见问题

- **没有 GPU 能否训练？** 可以。数据集较小，CPU 也能完成训练，配置中 batch 已相应调小。
- **CER 偏高？** 合成数据 + 小模型下属正常现象。可调大 `--repeat` 扩充数据、增加 `max_epoch`，或精简指令集降低任务难度。
- **找不到 `wenet.bin.train`？** 按「安装 WeNet 训练代码」一节克隆源码并 `pip install -e .`。
- **ffmpeg 相关报错？** 安装 ffmpeg 后重开终端。
- **配置 yaml 字段报错？** 不同 WeNet 版本字段不同，按报错提示调整对应行，核心结构保持不变。
