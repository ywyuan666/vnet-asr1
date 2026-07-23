# ============================================================
#  run_aishell.ps1
#  AISHELL-1 全流程 Pipeline：数据准备 -> 训练 -> 评测
# ------------------------------------------------------------
#  用法：
#     .\run_aishell.ps1                    # 从头跑到尾
#     .\run_aishell.ps1 -Stage 2           # 从第 2 阶段开始
#     .\run_aishell.ps1 -Stage 2 -Stop 2   # 只跑第 2 阶段
# ============================================================
param(
    [int]$Stage = 0,
    [int]$Stop  = 5,
    [string]$Device = "cpu",
    [int]$MaxEpoch = 60,
    [int]$BatchSize = 32,
    [int]$DModel = 144
)

$ErrorActionPreference = "Stop"
$DataDir  = "data/aishell"
$Dict     = "$DataDir/units.txt"
$Cmvn     = "$DataDir/global_cmvn"
$ExpDir   = "exp/aishell_conformer"

function Section($n, $msg) {
    Write-Host "`n==== Stage $n : $msg ====" -ForegroundColor Cyan
}

# -------- Stage 0: 下载并准备 AISHELL-1 --------
if ($Stage -le 0 -and $Stop -ge 0) {
    Section 0 "下载并准备 AISHELL-1 数据集"
    $skipExtract = "False"
    if (Test-Path "$DataDir/raw") { $skipExtract = "True" }
    python local/download_aishell.py --out_dir $DataDir --skip_extract:$skipExtract
}

# -------- Stage 1: CMVN --------
if ($Stage -le 1 -and $Stop -ge 1) {
    Section 1 "计算 CMVN 特征归一化统计量"
    python tools/make_cmvn.py --data_list "$DataDir/train/data.list" --out $Cmvn
}

# -------- Stage 2: 训练 Conformer + CTC/Attention/Transducer --------
if ($Stage -le 2 -and $Stop -ge 2) {
    Section 2 "AISHELL-1 训练 Conformer + CTC/Attention/Transducer"
    New-Item -ItemType Directory -Force -Path $ExpDir | Out-Null
    python train.py `
        --train_data "$DataDir/train/data.list" `
        --cv_data "$DataDir/dev/data.list" `
        --dict $Dict `
        --cmvn $Cmvn `
        --model_dir $ExpDir `
        --batch_size $BatchSize `
        --max_epoch $MaxEpoch `
        --device $Device `
        --d_model $DModel
}

# -------- Stage 3: 解码评测 --------
if ($Stage -le 3 -and $Stop -ge 3) {
    Section 3 "AISHELL-1 测试集解码评测 -- 三种解码模式"
    $ckpt = Join-Path $ExpDir "final.pt"
    if (-not (Test-Path $ckpt)) { $ckpt = Join-Path $ExpDir "best.pt" }
    if (-not (Test-Path $ckpt)) {
        Write-Host "  [警告] 未找到模型检查点，跳过解码" -ForegroundColor Yellow
    } else {
        python recognize_ctc_attn_transducer.py `
            --checkpoint $ckpt `
            --test_data "$DataDir/test/data.list" `
            --dict $Dict `
            --cmvn $Cmvn `
            --device $Device `
            --mode all
    }
}

# -------- Stage 4: 流式解码评测 --------
if ($Stage -le 4 -and $Stop -ge 4) {
    Section 4 "AISHELL-1 流式解码评测"
    $ckpt = Join-Path $ExpDir "final.pt"
    if (-not (Test-Path $ckpt)) { $ckpt = Join-Path $ExpDir "best.pt" }
    if (-not (Test-Path $ckpt)) {
        Write-Host "  [警告] 未找到模型检查点，跳过流式解码" -ForegroundColor Yellow
    } else {
        python recognize_ctc_attn_transducer.py `
            --checkpoint $ckpt `
            --test_data "$DataDir/test/data.list" `
            --dict $Dict `
            --cmvn $Cmvn `
            --device $Device `
            --mode attention `
            --streaming `
            --chunk_size 16
    }
}

# -------- Stage 5: 与 WeNet 对比报告 --------
if ($Stage -le 5 -and $Stop -ge 5) {
    Section 5 "生成 AISHELL-1 评测报告"
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host "  AISHELL-1 评测报告" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host "模型: Conformer + CTC/Attention/Transducer" -ForegroundColor Green
    Write-Host "参数量: ~6.6M" -ForegroundColor Green
    Write-Host "" -ForegroundColor Green
    Write-Host "解码模式       | CER (%)     | 说明" -ForegroundColor Green
    Write-Host "-------------|------------|---------------" -ForegroundColor Green
    Write-Host "CTC Greedy  |            | 最快，无语言模型" -ForegroundColor Green
    Write-Host "Attention   |            | 标准自回归解码" -ForegroundColor Green
    Write-Host "Transducer  |            | 流式友好解码" -ForegroundColor Green
    Write-Host "" -ForegroundColor Green
    Write-Host "上表数据请在 Stage 3 执行后填入。" -ForegroundColor Green
    Write-Host "流式 chunk_size=16, right_context=4" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
}

Write-Host "`nAISHELL-1 Pipeline 全部完成" -ForegroundColor Green
