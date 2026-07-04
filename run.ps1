# ============================================================
#  run.ps1  —  耳机语音识别 一键脚本 (Windows PowerShell)
# ------------------------------------------------------------
#  用法：
#     .\run.ps1                 # 从头跑到尾
#     .\run.ps1 -Stage 3        # 从第 3 阶段开始跑
#     .\run.ps1 -Stage 3 -Stop 3  # 只跑第 3 阶段
# ============================================================
param(
    [int]$Stage = 0,     # 从哪个 stage 开始
    [int]$Stop  = 5,     # 跑到哪个 stage 结束
    [int]$Repeat = 2,    # 数据合成重复轮数(越大数据越多)
    [string]$Device = "cpu"  # Windows 默认 CPU；有 CUDA 可传 -Device cuda
)

$ErrorActionPreference = "Stop"
$ExpDir   = "exp/u2pp_conformer"
$Dict     = "data/dict/units.txt"
$Cmvn     = "data/train/global_cmvn"
$Config   = "conf/train_u2pp_conformer.yaml"

function Section($n, $msg) { Write-Host "`n==== Stage $n : $msg ====" -ForegroundColor Cyan }

# -------- Stage 0: 合成数据 --------
if ($Stage -le 0 -and $Stop -ge 0) {
    Section 0 "用 TTS 合成语音指令数据集"
    python local/generate_corpus.py --out data/audio --repeat $Repeat
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

# -------- Stage 3: 训练 U2++ Conformer --------
if ($Stage -le 3 -and $Stop -ge 3) {
    Section 3 "训练 U2++ Conformer 模型"
    New-Item -ItemType Directory -Force -Path $ExpDir | Out-Null
    # 注意：不同 WeNet 版本参数可能有差异，详见 README 2.4
    python -m wenet.bin.train `
        --config $Config `
        --device $Device `
        --data_type raw `
        --train_data data/train/data.list `
        --cv_data data/dev/data.list `
        --model_dir $ExpDir `
        --num_workers 2 `
        --pin_memory
}

# -------- Stage 4: 解码评测 --------
if ($Stage -le 4 -and $Stop -ge 4) {
    Section 4 "在测试集上解码并计算 CER"
    $ckpt = Join-Path $ExpDir "final.pt"
    if (-not (Test-Path $ckpt)) {
        # 没有 final.pt 就用最后一个 epoch
        $ckpt = (Get-ChildItem $ExpDir -Filter "*.pt" | Sort-Object Name | Select-Object -Last 1).FullName
    }
    python recognize.py `
        --config (Join-Path $ExpDir "train.yaml") `
        --checkpoint $ckpt `
        --test_data data/test/data.list `
        --dict $Dict `
        --cmvn $Cmvn `
        --mode attention_rescoring
}

# -------- Stage 5: 单条识别 demo --------
if ($Stage -le 5 -and $Stop -ge 5) {
    Section 5 "随机挑一条测试音频做识别演示"
    $first = Get-Content data/test/data.list | Select-Object -First 1
    $wav = ($first | ConvertFrom-Json).wav
    $ckpt = Join-Path $ExpDir "final.pt"
    if (-not (Test-Path $ckpt)) {
        $ckpt = (Get-ChildItem $ExpDir -Filter "*.pt" | Sort-Object Name | Select-Object -Last 1).FullName
    }
    python infer_demo.py --checkpoint $ckpt --dict $Dict --cmvn $Cmvn --wav $wav `
        --config (Join-Path $ExpDir "train.yaml")
}

Write-Host "`n全部完成 ✅" -ForegroundColor Green
