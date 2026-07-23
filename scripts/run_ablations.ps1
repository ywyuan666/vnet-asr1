# ============================================================
#  run_ablations.ps1
#  一键运行所有消融实验
# ------------------------------------------------------------
#  用法：
#     .\scripts\run_ablations.ps1 -Device cuda
#     .\scripts\run_ablations.ps1 -Mode model_size -DryRun
# ============================================================
param(
    [string]$Device = "cpu",
    [string]$Mode = "all",
    [int]$Epochs = 30,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$dryRunFlag = if ($DryRun) { "--dry_run" } else { "" }

Write-Host "=== vnet-asr1 消融实验 ===" -ForegroundColor Cyan
Write-Host "设备: $Device" -ForegroundColor Gray
Write-Host "模式: $Mode" -ForegroundColor Gray
Write-Host "Epochs: $Epochs" -ForegroundColor Gray
Write-Host "Dry Run: $DryRun" -ForegroundColor Gray

python scripts/ablation_study.py `
    --mode $Mode `
    --device $Device `
    --epochs $Epochs `
    $dryRunFlag

if (-not $DryRun) {
    python scripts/benchmark_baselines.py
}
