# Fast Execution Script for Quick Testing
# This script runs a minimal configuration for fast testing
# Run with: .\scripts\quick_test.ps1

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
Write-Host "Fast Test - iTransformer" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Dataset: ETTh1 (7 variates - smallest)" -ForegroundColor Green
Write-Host "  Input: 48 time steps" -ForegroundColor Green
Write-Host "  Prediction: 24 time steps" -ForegroundColor Green
Write-Host "  Model size: Small (d_model=64)" -ForegroundColor Green
Write-Host "  Epochs: 2 (for speed)" -ForegroundColor Green
Write-Host "  Batch size: 32" -ForegroundColor Green
Write-Host ""

$model_name = "iTransformer"

python -u run.py `
  --is_training 1 `
  --root_path ../dataset/ETT-small/ `
  --data_path ETTh1.csv `
  --model_id quick_test_ETTh1 `
  --model $model_name `
  --data ETTh1 `
  --features M `
  --seq_len 48 `
  --label_len 24 `
  --pred_len 24 `
  --e_layers 1 `
  --enc_in 7 `
  --dec_in 7 `
  --c_out 7 `
  --d_model 64 `
  --d_ff 128 `
  --n_heads 2 `
  --batch_size 32 `
  --train_epochs 2 `
  --learning_rate 0.0001 `
  --patience 2 `
  --num_workers 0 `
  --itr 1

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Fast test completed!" -ForegroundColor Green
Write-Host "Check results in: ./checkpoints/quick_test_ETTh1/" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

