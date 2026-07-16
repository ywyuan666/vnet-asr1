#!/bin/bash
# ============================================================
#  run_ctc_attn_transducer.sh
#  Conformer + CTC / Attention / Transducer 三任务联合训练
#  Linux / Mac / WSL 版本
# ------------------------------------------------------------
#  用法：
#     bash run_ctc_attn_transducer.sh                   # 从头跑到尾
#     bash run_ctc_attn_transducer.sh --stage 3         # 从第 3 阶段开始
#     bash run_ctc_attn_transducer.sh --stage 3 --stop 3 # 只跑第 3 阶段
# ============================================================
set -euo pipefail

# ---- 默认参数 ----
STAGE=0
STOP=5
REPEAT=5
DEVICE="cpu"

# ---- 解析参数 ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage) STAGE="$2"; shift 2 ;;
        --stop)  STOP="$2";  shift 2 ;;
        --repeat) REPEAT="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

EXP_DIR="exp/conformer_ctc_attn_transducer"
DICT="data/dict/units.txt"
CMVN="data/train/global_cmvn"

section() {
    echo ""
    echo "==== Stage $1 : $2 ===="
}

# -------- Stage 0: 合成数据 --------
if [[ $STAGE -le 0 && $STOP -ge 0 ]]; then
    section 0 "用 TTS 合成语音指令数据集"
    python local/generate_corpus_ctc_attn_transducer.py --out data/audio --repeat "$REPEAT"
fi

# -------- Stage 1: 准备数据清单 + 字典 --------
if [[ $STAGE -le 1 && $STOP -ge 1 ]]; then
    section 1 "准备 data.list 与字典"
    python local/prepare_data.py --audio_dir data/audio --out_dir data
fi

# -------- Stage 2: CMVN --------
if [[ $STAGE -le 2 && $STOP -ge 2 ]]; then
    section 2 "计算 CMVN 特征归一化统计量"
    python tools/make_cmvn.py --data_list data/train/data.list --out "$CMVN"
fi

# -------- Stage 3: 训练 --------
if [[ $STAGE -le 3 && $STOP -ge 3 ]]; then
    section 3 "训练 Conformer + CTC/Attention/Transducer 模型"
    mkdir -p "$EXP_DIR"
    python train.py \
        --train_data data/train/data.list \
        --cv_data data/dev/data.list \
        --dict "$DICT" \
        --cmvn "$CMVN" \
        --model_dir "$EXP_DIR" \
        --batch_size 16 \
        --max_epoch 200 \
        --device "$DEVICE" \
        --d_model 144 \
        --ctc_weight 0.3 \
        --attn_weight 0.3 \
        --trans_weight 0.4
fi

# -------- Stage 4: 解码评测 --------
if [[ $STAGE -le 4 && $STOP -ge 4 ]]; then
    section 4 "三种解码模式评测并计算 CER"
    CKPT="$EXP_DIR/final.pt"
    [[ -f "$CKPT" ]] || CKPT="$EXP_DIR/best.pt"
    python recognize_ctc_attn_transducer.py \
        --checkpoint "$CKPT" \
        --test_data data/test/data.list \
        --dict "$DICT" \
        --cmvn "$CMVN" \
        --device "$DEVICE" \
        --mode all
fi

# -------- Stage 5: 单条音频识别演示 --------
if [[ $STAGE -le 5 && $STOP -ge 5 ]]; then
    section 5 "单条音频识别演示"
    FIRST=$(head -1 data/test/data.list)
    WAV=$(echo "$FIRST" | python -c "import json,sys; print(json.load(sys.stdin)['wav'])")
    CKPT="$EXP_DIR/final.pt"
    [[ -f "$CKPT" ]] || CKPT="$EXP_DIR/best.pt"

    python infer_demo_ctc_attn_transducer.py \
        --checkpoint "$CKPT" \
        --dict "$DICT" \
        --cmvn "$CMVN" \
        --wav "$WAV" \
        --mode ctc_greedy

    python infer_demo_ctc_attn_transducer.py \
        --checkpoint "$CKPT" \
        --dict "$DICT" \
        --cmvn "$CMVN" \
        --wav "$WAV" \
        --mode attention

    python infer_demo_ctc_attn_transducer.py \
        --checkpoint "$CKPT" \
        --dict "$DICT" \
        --cmvn "$CMVN" \
        --wav "$WAV" \
        --mode transducer
fi

echo ""
echo "全部完成！"
