# Quick Start Script for iTransformer
# Simple example to get started quickly
# Run with: .\scripts\quick_start.ps1

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
Write-Host "iTransformer Quick Start" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Running quick test on ETTh1 dataset..." -ForegroundColor Yellow
Write-Host "Dataset: ETTh1 (7 variates)" -ForegroundColor Green
Write-Host "Input length: 96 time steps" -ForegroundColor Green
Write-Host "Prediction length: 96 time steps" -ForegroundColor Green
Write-Host ""

$model_name = "iTransformer"

python -u run.py `
  --is_training 1 `
  --root_path ../dataset/ETT-small/ `
  --data_path ETTh1.csv `
  --model_id quick_start_ETTh1 `
  --model $model_name `
  --data ETTh1 `
  --features M `
  --seq_len 96 `
  --pred_len 96 `
  --e_layers 2 `
  --enc_in 7 `
  --dec_in 7 `
  --c_out 7 `
  --d_model 128 `
  --d_ff 128 `
  --batch_size 16 `
  --train_epochs 5 `
  --learning_rate 0.0001 `
  --num_workers 0 `
  --itr 1

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Quick start completed!" -ForegroundColor Cyan
Write-Host "Check results in: ./checkpoints/quick_start_ETTh1/" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

