# Complete Weather Prediction Test Script
# This script trains, tests, and visualizes weather predictions
# Run with: .\scripts\test_weather_prediction.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Weather Prediction Test Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to the project root directory (AI-Project-FAST)
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath
Set-Location $projectRoot

Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
Write-Host ""

# ============================================================================
# SETTINGS - Change these as needed
# ============================================================================
# Set to $true to run training and testing, $false to skip to visualization
$RUN_TRAINING_AND_TESTING = $false

# Configuration
$model_id = "weather_test"
$seq_len = 96
$pred_len = 96
$train_epochs = 5  # Reduced for faster testing
$batch_size = 32   # Smaller batch for testing
# ============================================================================

Write-Host "Configuration:" -ForegroundColor Green
Write-Host "  Model ID: $model_id"
Write-Host "  Sequence Length: $seq_len (input: past $seq_len hours)"
Write-Host "  Prediction Length: $pred_len (forecast: next $pred_len hours)"
Write-Host "  Training Epochs: $train_epochs"
Write-Host "  Batch Size: $batch_size"
Write-Host "  Run Training/Testing: $RUN_TRAINING_AND_TESTING"
Write-Host ""

if ($RUN_TRAINING_AND_TESTING) {
    # Step 1: Training
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Step 1: Training the Model" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""

    Write-Host "Starting training..." -ForegroundColor Yellow
    & python -u run.py `
      --is_training 1 `
      --root_path ../dataset/weather/ `
      --data_path weather.csv `
      --model_id $model_id `
      --model iTransformer `
      --data custom `
      --features M `
      --seq_len $seq_len `
      --label_len 48 `
      --pred_len $pred_len `
      --e_layers 3 `
      --enc_in 21 `
      --dec_in 21 `
      --c_out 21 `
      --d_model 512 `
      --d_ff 512 `
      --batch_size $batch_size `
      --train_epochs $train_epochs `
      --num_workers 0 `
      --itr 1

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Training failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }

    Write-Host ""
    Write-Host "Training completed!" -ForegroundColor Green
    Write-Host ""

    # Step 2: Testing
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Step 2: Testing the Model" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""

    Write-Host "Starting testing..." -ForegroundColor Yellow
    & python -u run.py `
      --is_training 0 `
      --root_path ../dataset/weather/ `
      --data_path weather.csv `
      --model_id $model_id `
      --model iTransformer `
      --data custom `
      --features M `
      --seq_len $seq_len `
      --label_len 48 `
      --pred_len $pred_len `
      --e_layers 3 `
      --enc_in 21 `
      --dec_in 21 `
      --c_out 21 `
      --d_model 512 `
      --d_ff 512 `
      --batch_size $batch_size `
      --num_workers 0 `
      --itr 1

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Testing failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }

    Write-Host ""
    Write-Host "Testing completed!" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "Skipping Training and Testing" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "RUN_TRAINING_AND_TESTING is set to `$false" -ForegroundColor Yellow
    Write-Host "Proceeding directly to visualization..." -ForegroundColor Yellow
    Write-Host ""
}

# Step 3: Visualize Results
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Step 3: Visualizing Results" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$visualizeScript = Join-Path $projectRoot "scripts\visualize_weather_results.py"

# Find the actual results directory (it has a longer name than just model_id)
$resultsDir = Join-Path $projectRoot "results"
if (-not (Test-Path $resultsDir)) {
    Write-Host "Results directory not found: $resultsDir" -ForegroundColor Red
    Write-Host "Please check if training and testing completed successfully." -ForegroundColor Yellow
    exit 1
}

# Find directories that start with the model_id
$matchingDirs = Get-ChildItem -Path $resultsDir -Directory | Where-Object { $_.Name -like "$model_id*" }

if ($matchingDirs.Count -eq 0) {
    Write-Host "No results directory found starting with: $model_id" -ForegroundColor Red
    Write-Host "Available directories in results:" -ForegroundColor Yellow
    Get-ChildItem -Path $resultsDir -Directory | ForEach-Object { Write-Host "  - $($_.Name)" -ForegroundColor White }
    exit 1
}

# Use the first matching directory (or the most recent one)
$actualResultsPath = $matchingDirs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$actualModelId = $actualResultsPath.Name

Write-Host "Found results directory: $actualModelId" -ForegroundColor Green
Write-Host ""

Write-Host "Loading and visualizing results..." -ForegroundColor Yellow
& python $visualizeScript --model_id $actualModelId

if ($LASTEXITCODE -ne 0) {
    Write-Host "Visualization failed. Trying basic visualization..." -ForegroundColor Yellow
    
    # Fallback: Basic visualization
    python -c @"
import numpy as np
import matplotlib.pyplot as plt
import os

model_id = '$actualModelId'
results_path = f'./results/{model_id}/'

if os.path.exists(results_path):
    preds = np.load(os.path.join(results_path, 'pred.npy'))
    trues = np.load(os.path.join(results_path, 'true.npy'))
    metrics = np.load(os.path.join(results_path, 'metrics.npy'))
    
    print(f'\nResults for: {model_id}')
    print(f'Predictions shape: {preds.shape}')
    print(f'True values shape: {trues.shape}')
    print(f'\nMetrics:')
    print(f'  MAE: {metrics[0]:.4f}')
    print(f'  MSE: {metrics[1]:.4f}')
    print(f'  RMSE: {metrics[2]:.4f}')
    print(f'  MAPE: {metrics[3]:.4f}%')
    print(f'  MSPE: {metrics[4]:.4f}%')
    
    # Simple plot
    plt.figure(figsize=(14, 6))
    plt.plot(trues[0, :, 0], label='Actual', marker='o')
    plt.plot(preds[0, :, 0], label='Predicted', marker='x')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Weather Prediction - Sample 0, Variable 0')
    plt.legend()
    plt.grid(True)
    plot_path = f'./results/{model_id}/prediction_plot.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved to: {plot_path}')
    plt.close()
else:
    print(f'Results not found at {results_path}')
"@
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Test Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved in: ./results/$actualModelId/" -ForegroundColor Green
Write-Host "  - pred.npy: Predictions" -ForegroundColor White
Write-Host "  - true.npy: Actual values" -ForegroundColor White
Write-Host "  - metrics.npy: Performance metrics" -ForegroundColor White
Write-Host "  - *.png: Visualization plots" -ForegroundColor White
Write-Host ""
Write-Host "To view detailed results, run:" -ForegroundColor Yellow
Write-Host "  python scripts\visualize_weather_results.py --model_id $actualModelId" -ForegroundColor White
Write-Host ""

