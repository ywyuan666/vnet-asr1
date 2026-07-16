# ============================================================
#  run_ctc_attn_transducer.ps1
#  Conformer + CTC / Attention / Transducer 三任务联合训练
# ------------------------------------------------------------
#  用法：
#     .\run_ctc_attn_transducer.ps1                # 从头跑到尾
#     .\run_ctc_attn_transducer.ps1 -Stage 3       # 从第 3 阶段开始
#     .\run_ctc_attn_transducer.ps1 -Stage 3 -Stop 3  # 只跑第 3 阶段
# ============================================================
param(
    [int]$Stage = 0,
    [int]$Stop  = 5,
    [int]$Repeat = 5,
    [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"
$ExpDir   = "exp/conformer_ctc_attn_transducer"
$Dict     = "data/dict/units.txt"
$Cmvn     = "data/train/global_cmvn"

function Section($n, $msg) {
    Write-Host "`n==== Stage $n : $msg ====" -ForegroundColor Cyan
}

# -------- Stage 0: 合成数据 --------
if ($Stage -le 0 -and $Stop -ge 0) {
    Section 0 "用 TTS 合成语音指令数据集"
    python local/generate_corpus_ctc_attn_transducer.py --out data/audio --repeat $Repeat
}

# -------- Stage 1: 准备数据清单 + 字典 --------
if ($Stage -le 1 -and $Stop -ge 1) {
    Section 1 "准备 data.list 与字典"
    python local/prepare_data.py --audio_dir data/audio --out_dir data
}

# -------- Stage 2: CMVN --------
if ($Stage -le 2 -and $Stop -ge 2) {
    Section 2 "计算 CMVN 特征归一化统计量"
    python tools/make_cmvn.py --data_list data/train/data.list --out $Cmvn
}

# -------- Stage 3: 训练 Conformer + CTC/Attention/Transducer --------
if ($Stage -le 3 -and $Stop -ge 3) {
    Section 3 "训练 Conformer + CTC/Attention/Transducer 模型"
    New-Item -ItemType Directory -Force -Path $ExpDir | Out-Null
    python train.py `
        --train_data data/train/data.list `
        --cv_data data/dev/data.list `
        --dict $Dict `
        --cmvn $Cmvn `
        --model_dir $ExpDir `
        --batch_size 16 `
        --max_epoch 200 `
        --device $Device `
        --d_model 144 `
        --ctc_weight 0.3 `
        --attn_weight 0.3 `
        --trans_weight 0.4
}

# -------- Stage 4: 解码评测 --------
if ($Stage -le 4 -and $Stop -ge 4) {
    Section 4 "三种解码模式评测并计算 CER"
    $ckpt = Join-Path $ExpDir "final.pt"
    if (-not (Test-Path $ckpt)) {
        $ckpt = Join-Path $ExpDir "best.pt"
    }
    python recognize_ctc_attn_transducer.py `
        --checkpoint $ckpt `
        --test_data data/test/data.list `
        --dict $Dict `
        --cmvn $Cmvn `
        --device $Device `
        --mode all
}

# -------- Stage 5: 单条音频识别演示 --------
if ($Stage -le 5 -and $Stop -ge 5) {
    Section 5 "单条音频识别演示"
    $first = Get-Content data/test/data.list | Select-Object -First 1
    $wav = ($first | ConvertFrom-Json).wav
    $ckpt = Join-Path $ExpDir "final.pt"
    if (-not (Test-Path $ckpt)) {
        $ckpt = Join-Path $ExpDir "best.pt"
    }
    python infer_demo_ctc_attn_transducer.py `
        --checkpoint $ckpt `
        --dict $Dict `
        --cmvn $Cmvn `
        --wav $wav `
        --mode ctc_greedy

    python infer_demo_ctc_attn_transducer.py `
        --checkpoint $ckpt `
        --dict $Dict `
        --cmvn $Cmvn `
        --wav $wav `
        --mode attention

    python infer_demo_ctc_attn_transducer.py `
        --checkpoint $ckpt `
        --dict $Dict `
        --cmvn $Cmvn `
        --wav $wav `
        --mode transducer
}

Write-Host "`n全部完成 ✅" -ForegroundColor Green
