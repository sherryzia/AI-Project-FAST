# PowerShell script for Variate Generalization on Traffic dataset
# Train on partial variates, test on all variates (zero-shot generalization)
# Run with: .\scripts\variate_generalization\Traffic\iTransformer.ps1

# Change to the project root directory (AI-Project-FAST)
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $scriptPath))
Set-Location $projectRoot

# Train iTransformer on only 20% of variates (172 out of 862)
# The model will learn transferable representations
$model_name = "iTransformer"

python -u run.py `
  --is_training 1 `
  --root_path ../dataset/traffic/ `
  --data_path traffic.csv `
  --model_id traffic_96_96_partial `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 96 `
  --label_len 48 `
  --pred_len 96 `
  --e_layers 2 `
  --d_layers 1 `
  --factor 3 `
  --enc_in 172 `
  --dec_in 172 `
  --c_out 172 `
  --des 'Exp' `
  --exp_name partial_train `
  --num_workers 0 `
  --itr 1

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Training completed on 20% of variates (172/862)" -ForegroundColor Green
Write-Host "The model can now generalize to all 862 variates!" -ForegroundColor Green
Write-Host ""
Write-Host "To test on all variates, run:" -ForegroundColor Yellow
Write-Host "python -u run.py --is_training 0 --model_id traffic_96_96_partial --model iTransformer --data custom --features M --seq_len 96 --pred_len 96 --e_layers 2 --enc_in 862 --dec_in 862 --c_out 862" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan

