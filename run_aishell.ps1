# ============================================================
#  run_aishell.ps1
#  AISHELL-1 Pipeline: data prep -> training -> evaluation
# -----------------------------------------------------------
#  Usage:
#     .\run_aishell.ps1                       # run all stages
#     .\run_aishell.ps1 -Stage 2              # start from stage 2
#     .\run_aishell.ps1 -Stage 2 -Stop 2      # run only stage 2
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
    Write-Host ""
    Write-Host "==== Stage $n : $msg ====" -ForegroundColor Cyan
}

# -------- Stage 0: Download & prepare AISHELL-1 --------
if ($Stage -le 0 -and $Stop -ge 0) {
    Section 0 "Download and prepare AISHELL-1 dataset"
    $skipExtract = "False"
    if (Test-Path "$DataDir/raw") { $skipExtract = "True" }
    python local/download_aishell.py --out_dir $DataDir --skip_extract:$skipExtract
}

# -------- Stage 1: CMVN --------
if ($Stage -le 1 -and $Stop -ge 1) {
    Section 1 "Compute CMVN normalization stats"
    python tools/make_cmvn.py --data_list "$DataDir/train/data.list" --out $Cmvn
}

# -------- Stage 2: Train Conformer + CTC/Attention/Transducer --------
if ($Stage -le 2 -and $Stop -ge 2) {
    Section 2 "Train Conformer + CTC/Attention/Transducer on AISHELL-1"
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

# -------- Stage 3: Decode evaluation --------
if ($Stage -le 3 -and $Stop -ge 3) {
    Section 3 "Decode AISHELL-1 test set -- three modes"
    $ckpt = Join-Path $ExpDir "final.pt"
    if (-not (Test-Path $ckpt)) { $ckpt = Join-Path $ExpDir "best.pt" }
    if (-not (Test-Path $ckpt)) {
        Write-Host "  [WARN] No checkpoint found, skip decoding" -ForegroundColor Yellow
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

# -------- Stage 4: Streaming decode evaluation --------
if ($Stage -le 4 -and $Stop -ge 4) {
    Section 4 "AISHELL-1 Streaming decode evaluation"
    $ckpt = Join-Path $ExpDir "final.pt"
    if (-not (Test-Path $ckpt)) { $ckpt = Join-Path $ExpDir "best.pt" }
    if (-not (Test-Path $ckpt)) {
        Write-Host "  [WARN] No checkpoint found, skip streaming decode" -ForegroundColor Yellow
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

# -------- Stage 5: Generate AISHELL-1 benchmark report --------
if ($Stage -le 5 -and $Stop -ge 5) {
    Section 5 "Generate AISHELL-1 eval report"
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host "  AISHELL-1 EVALUATION REPORT" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host "Model: Conformer + CTC/Attention/Transducer" -ForegroundColor Green
    Write-Host "Params: ~6.6M" -ForegroundColor Green
    Write-Host "" -ForegroundColor Green
    Write-Host "Decode Mode    | CER (%)     | Notes" -ForegroundColor Green
    Write-Host "---------------|-------------|------------------" -ForegroundColor Green
    Write-Host "CTC Greedy     |             | Fastest, no LM" -ForegroundColor Green
    Write-Host "Attention      |             | Standard AR dec." -ForegroundColor Green
    Write-Host "Transducer     |             | Streaming friendly" -ForegroundColor Green
    Write-Host "" -ForegroundColor Green
    Write-Host "Fill CER values after Stage 3 completes." -ForegroundColor Green
    Write-Host "Streaming chunk_size=16, right_context=4" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
}

Write-Host ""
Write-Host "AISHELL-1 Pipeline done." -ForegroundColor Green
