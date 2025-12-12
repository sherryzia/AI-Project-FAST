# PowerShell script for testing iTransformer with increasing lookback windows on ECL dataset
# Run with: .\scripts\increasing_lookback\ECL\iTransformer.ps1

# Change to the project root directory (AI-Project-FAST)
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $scriptPath))
Set-Location $projectRoot

$model_name = "iTransformer"

# Lookback window: 48
python -u run.py `
  --is_training 1 `
  --root_path ../dataset/electricity/ `
  --data_path electricity.csv `
  --model_id ECL_48_96 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 48 `
  --pred_len 96 `
  --e_layers 3 `
  --enc_in 321 `
  --dec_in 321 `
  --c_out 321 `
  --des 'Exp' `
  --d_model 512 `
  --d_ff 512 `
  --batch_size 16 `
  --learning_rate 0.0005 `
  --num_workers 0 `
  --itr 1

# Lookback window: 96
python -u run.py `
  --is_training 1 `
  --root_path ../dataset/electricity/ `
  --data_path electricity.csv `
  --model_id ECL_96_96 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 96 `
  --pred_len 96 `
  --e_layers 3 `
  --enc_in 321 `
  --dec_in 321 `
  --c_out 321 `
  --des 'Exp' `
  --d_model 512 `
  --d_ff 512 `
  --batch_size 16 `
  --learning_rate 0.0005 `
  --num_workers 0 `
  --itr 1

# Lookback window: 192
python -u run.py `
  --is_training 1 `
  --root_path ../dataset/electricity/ `
  --data_path electricity.csv `
  --model_id ECL_192_96 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 192 `
  --pred_len 96 `
  --e_layers 3 `
  --enc_in 321 `
  --dec_in 321 `
  --c_out 321 `
  --des 'Exp' `
  --d_model 512 `
  --d_ff 512 `
  --batch_size 16 `
  --learning_rate 0.0005 `
  --num_workers 0 `
  --itr 1

# Lookback window: 336
python -u run.py `
  --is_training 1 `
  --root_path ../dataset/electricity/ `
  --data_path electricity.csv `
  --model_id ECL_336_96 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 336 `
  --pred_len 96 `
  --e_layers 3 `
  --enc_in 321 `
  --dec_in 321 `
  --c_out 321 `
  --des 'Exp' `
  --d_model 512 `
  --d_ff 512 `
  --batch_size 16 `
  --learning_rate 0.0005 `
  --num_workers 0 `
  --itr 1

