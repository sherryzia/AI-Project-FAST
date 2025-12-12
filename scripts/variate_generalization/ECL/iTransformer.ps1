# PowerShell script for Variate Generalization on ECL (Electricity) dataset
# Train on partial variates, test on all variates (zero-shot generalization)
# Run with: .\scripts\variate_generalization\ECL\iTransformer.ps1

# Change to the project root directory (AI-Project-FAST)
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $scriptPath))
Set-Location $projectRoot

# Train iTransformer on only 20% of variates (64 out of 321)
# The model will learn transferable representations
$model_name = "iTransformer"

python -u run.py `
  --is_training 1 `
  --root_path ../dataset/electricity/ `
  --data_path electricity.csv `
  --model_id ECL_96_96_partial `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 96 `
  --label_len 48 `
  --pred_len 96 `
  --e_layers 2 `
  --d_layers 1 `
  --factor 3 `
  --enc_in 64 `
  --dec_in 64 `
  --c_out 64 `
  --des 'Exp' `
  --exp_name partial_train `
  --num_workers 0 `
  --itr 1

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Training completed on 20% of variates (64/321)" -ForegroundColor Green
Write-Host "The model can now generalize to all 321 variates!" -ForegroundColor Green
Write-Host ""
Write-Host "To test on all variates, run:" -ForegroundColor Yellow
Write-Host "python -u run.py --is_training 0 --model_id ECL_96_96_partial --model iTransformer --data custom --features M --seq_len 96 --pred_len 96 --e_layers 2 --enc_in 321 --dec_in 321 --c_out 321" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan

