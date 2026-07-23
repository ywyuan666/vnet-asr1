# vnet-asr1 快速入门

## 环境准备

```bash
# 1. 创建 conda 环境
conda create -n vnet-asr python=3.10
conda activate vnet-asr

# 2. 安装依赖
pip install -r requirements_ctc_attn_transducer.txt
```

## 快速体验（合成数据）

```powershell
# 3 分钟体验完整流程
.\run_ctc_attn_transducer.ps1 -Repeat 3 -Device cpu
```

## AISHELL-1 训练

```powershell
# 完整 AISHELL-1 评测
.\run_aishell.ps1 -Device cuda -MaxEpoch 60
```

## 流式推理

```bash
# 流式识别
python recognize_ctc_attn_transducer.py \
    --checkpoint exp/aishell_conformer/final.pt \
    --test_data data/aishell/test/data.list \
    --dict data/aishell/units.txt \
    --cmvn data/aishell/global_cmvn \
    --device cuda \
    --mode transducer \
    --streaming \
    --chunk_size 16
```

## 消融实验

```powershell
# 一键跑所有消融实验
.\scripts\run_ablations.ps1 -Device cuda -Epochs 30
```

## 推理加速

```bash
# ONNX 导出
python tools/export_tensorrt.py \
    --checkpoint exp/aishell_conformer/final.pt \
    --dict data/aishell/units.txt \
    --onnx_only

# TensorRT FP16
python tools/export_tensorrt.py \
    --checkpoint exp/aishell_conformer/final.pt \
    --dict data/aishell/units.txt \
    --fp16

# TensorRT INT8
python tools/export_tensorrt.py \
    --checkpoint exp/aishell_conformer/final.pt \
    --dict data/aishell/units.txt \
    --int8 \
    --calib_data data/aishell/train/data.list
```

## 项目结构

```
vnet-asr1/
├── model/                      # 模型定义
│   ├── conformer_ctc_attn_transducer.py  # 核心模型
│   ├── streaming_helper.py     # 流式推理工具
│   └── augmentation.py         # 数据增强
├── train.py                    # 训练脚本
├── recognize_ctc_attn_transducer.py  # 评测脚本
├── infer_demo_ctc_attn_transducer.py # 单条推理
├── local/                      # 数据处理
│   ├── download_aishell.py     # AISHELL-1 下载
│   ├── prepare_aishell.py      # AISHELL-1 准备
│   └── prepare_data.py         # 数据格式转换
├── tools/                      # 工具
│   ├── make_cmvn.py            # CMVN 计算
│   ├── train_kenlm.py          # LM 训练
│   ├── lm_rescore.py           # LM 重打分
│   └── export_tensorrt.py      # TensorRT 导出
├── scripts/                    # 实验脚本
│   ├── ablation_study.py       # 消融实验
│   └── benchmark_baselines.py  # 基准对比
├── tests/                      # 单元测试
│   ├── test_model.py
│   ├── test_data.py
│   └── test_inference.py
├── docs/                       # 文档
│   ├── system_design.md        # 系统设计
│   └── quick_start.md          # 快速入门
└── .github/workflows/ci.yml    # CI/CD
```
