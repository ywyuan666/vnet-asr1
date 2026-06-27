#!/usr/bin/env bash
# ============================================================
#  run.sh  —  耳机语音识别 一键脚本 (Linux / Mac / WSL)
# ------------------------------------------------------------
#  用法：
#     bash run.sh                 # 从头跑到尾
#     stage=3 stop_stage=3 bash run.sh   # 只跑第 3 阶段
# ============================================================
set -e

stage=${stage:-0}
stop_stage=${stop_stage:-5}
repeat=${repeat:-2}

exp_dir="exp/u2pp_conformer"
dict="data/dict/units.txt"
cmvn="data/train/global_cmvn"
config="conf/train_u2pp_conformer.yaml"

echo_stage() { echo -e "\n==== Stage $1 : $2 ===="; }

# Stage 0: 合成数据
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo_stage 0 "用 TTS 合成语音指令数据集"
    python local/generate_corpus.py --out data/audio --repeat ${repeat}
fi

# Stage 1: 准备数据
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo_stage 1 "准备 data.list 与字典"
    python local/prepare_data.py --audio_dir data/audio --out_dir data
fi

# Stage 2: CMVN
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo_stage 2 "计算 CMVN"
    python tools/make_cmvn.py --data_list data/train/data.list --out ${cmvn}
fi

# Stage 3: 训练
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo_stage 3 "训练 U2++ Conformer 模型"
    mkdir -p ${exp_dir}
    python -m wenet.bin.train \
        --config ${config} \
        --data_type raw \
        --symbol_table ${dict} \
        --train_data data/train/data.list \
        --cv_data data/dev/data.list \
        --model_dir ${exp_dir} \
        --cmvn ${cmvn} \
        --num_workers 2 \
        --pin_memory
fi

# Stage 4: 解码评测
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo_stage 4 "在测试集上解码并计算 CER"
    ckpt=${exp_dir}/final.pt
    [ -f "${ckpt}" ] || ckpt=$(ls -1 ${exp_dir}/*.pt | sort | tail -n1)
    python recognize.py \
        --config ${exp_dir}/train.yaml \
        --checkpoint ${ckpt} \
        --test_data data/test/data.list \
        --dict ${dict} \
        --cmvn ${cmvn} \
        --mode attention_rescoring
fi

# Stage 5: 单条识别 demo
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo_stage 5 "随机挑一条测试音频做识别演示"
    wav=$(head -n1 data/test/data.list | python -c "import sys,json;print(json.loads(sys.stdin.read())['wav'])")
    ckpt=${exp_dir}/final.pt
    [ -f "${ckpt}" ] || ckpt=$(ls -1 ${exp_dir}/*.pt | sort | tail -n1)
    python infer_demo.py --checkpoint ${ckpt} --dict ${dict} --cmvn ${cmvn} \
        --config ${exp_dir}/train.yaml --wav "${wav}"
fi

echo -e "\n全部完成 ✅"
