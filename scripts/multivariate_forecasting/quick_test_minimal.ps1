# Ultra-Fast Minimal Test Script
# Absolute minimum configuration for fastest possible execution
# Run with: .\scripts\multivariate_forecasting\quick_test_minimal.ps1

# Change to the project root directory (AI-Project-FAST)
# Find the AI-Project-FAST directory by looking for run.py
$scriptPath = $PSScriptRoot
if (-not $scriptPath) {
    $scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
}

# Navigate up until we find run.py
$currentPath = $scriptPath
$maxDepth = 5
$found = $false

for ($i = 0; $i -lt $maxDepth; $i++) {
    if (Test-Path (Join-Path $currentPath "run.py")) {
        Set-Location $currentPath
        $found = $true
        break
    }
    $currentPath = Split-Path -Parent $currentPath
    if (-not $currentPath) { break }
}

if (-not $found) {
    Write-Host "Error: Could not find run.py. Searched from: $scriptPath" -ForegroundColor Red
    exit 1
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Ultra-Fast Minimal Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Minimal configuration for fastest execution:" -ForegroundColor Yellow
Write-Host "  Dataset: ETTh1 (7 variates)" -ForegroundColor Green
Write-Host "  Input: 24 time steps" -ForegroundColor Green
Write-Host "  Prediction: 12 time steps" -ForegroundColor Green
Write-Host "  Model: Tiny (d_model=32, 1 layer)" -ForegroundColor Green
Write-Host "  Epochs: 1" -ForegroundColor Green
Write-Host ""

$model_name = "iTransformer"

python -u run.py `
  --is_training 1 `
  --root_path ../dataset/ETT-small/ `
  --data_path ETTh1.csv `
  --model_id minimal_test `
  --model $model_name `
  --data ETTh1 `
  --features M `
  --seq_len 24 `
  --label_len 12 `
  --pred_len 12 `
  --e_layers 1 `
  --enc_in 7 `
  --dec_in 7 `
  --c_out 7 `
  --d_model 32 `
  --d_ff 64 `
  --n_heads 2 `
  --batch_size 64 `
  --train_epochs 1 `
  --learning_rate 0.0001 `
  --patience 1 `
  --num_workers 0 `
  --itr 1

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Minimal test completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
