# PowerShell script for iTransformer on Exchange Rate dataset
# Run with: .\scripts\multivariate_forecasting\Exchange\iTransformer.ps1

# Change to the project root directory (AI-Project-FAST)
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $scriptPath))
Set-Location $projectRoot

$model_name = "iTransformer"

# Prediction length 96
python -u run.py `
  --is_training 1 `
  --root_path ../dataset/exchange_rate/ `
  --data_path exchange_rate.csv `
  --model_id Exchange_96_96 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 96 `
  --pred_len 96 `
  --e_layers 2 `
  --enc_in 8 `
  --dec_in 8 `
  --c_out 8 `
  --des 'Exp' `
  --d_model 128 `
  --d_ff 128 `
  --num_workers 0 `
  --itr 1

# Prediction length 192
python -u run.py `
  --is_training 1 `
  --root_path ../dataset/exchange_rate/ `
  --data_path exchange_rate.csv `
  --model_id Exchange_96_192 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 96 `
  --pred_len 192 `
  --e_layers 2 `
  --enc_in 8 `
  --dec_in 8 `
  --c_out 8 `
  --des 'Exp' `
  --d_model 128 `
  --d_ff 128 `
  --num_workers 0 `
  --itr 1

# Prediction length 336
python -u run.py `
  --is_training 1 `
  --root_path ../dataset/exchange_rate/ `
  --data_path exchange_rate.csv `
  --model_id Exchange_96_336 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 96 `
  --pred_len 336 `
  --e_layers 2 `
  --enc_in 8 `
  --dec_in 8 `
  --c_out 8 `
  --des 'Exp' `
  --d_model 128 `
  --d_ff 128 `
  --num_workers 0 `
  --itr 1

# Prediction length 720
python -u run.py `
  --is_training 1 `
  --root_path ../dataset/exchange_rate/ `
  --data_path exchange_rate.csv `
  --model_id Exchange_96_720 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 96 `
  --pred_len 720 `
  --e_layers 2 `
  --enc_in 8 `
  --dec_in 8 `
  --c_out 8 `
  --des 'Exp' `
  --d_model 128 `
  --d_ff 128 `
  --num_workers 0 `
  --itr 1

