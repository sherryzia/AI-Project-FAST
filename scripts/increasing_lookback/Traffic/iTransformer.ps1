# PowerShell script for testing iTransformer with increasing lookback windows on Traffic dataset
# Run with: .\scripts\increasing_lookback\Traffic\iTransformer.ps1

# Change to the project root directory (AI-Project-FAST)
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $scriptPath))
Set-Location $projectRoot

$model_name = "iTransformer"

# Lookback window: 48
python -u run.py `
  --is_training 1 `
  --root_path ../dataset/traffic/ `
  --data_path traffic.csv `
  --model_id traffic_48_96 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 48 `
  --pred_len 96 `
  --e_layers 4 `
  --enc_in 862 `
  --dec_in 862 `
  --c_out 862 `
  --des 'Exp' `
  --d_model 512 `
  --d_ff 512 `
  --batch_size 16 `
  --learning_rate 0.001 `
  --num_workers 0 `
  --itr 1

# Lookback window: 96
python -u run.py `
  --is_training 1 `
  --root_path ../dataset/traffic/ `
  --data_path traffic.csv `
  --model_id traffic_96_96 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 96 `
  --pred_len 96 `
  --e_layers 4 `
  --factor 3 `
  --enc_in 862 `
  --dec_in 862 `
  --c_out 862 `
  --des 'Exp' `
  --d_model 512 `
  --d_ff 512 `
  --batch_size 16 `
  --learning_rate 0.001 `
  --num_workers 0 `
  --itr 1

# Lookback window: 192
python -u run.py `
  --is_training 1 `
  --root_path ../dataset/traffic/ `
  --data_path traffic.csv `
  --model_id traffic_192_96 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 192 `
  --pred_len 96 `
  --e_layers 4 `
  --enc_in 862 `
  --dec_in 862 `
  --c_out 862 `
  --des 'Exp' `
  --d_model 512 `
  --d_ff 512 `
  --batch_size 16 `
  --learning_rate 0.001 `
  --num_workers 0 `
  --itr 1

# Lookback window: 336
python -u run.py `
  --is_training 1 `
  --root_path ../dataset/traffic/ `
  --data_path traffic.csv `
  --model_id traffic_336_96 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 336 `
  --pred_len 96 `
  --e_layers 4 `
  --enc_in 862 `
  --dec_in 862 `
  --c_out 862 `
  --des 'Exp' `
  --d_model 512 `
  --d_ff 512 `
  --batch_size 16 `
  --learning_rate 0.001 `
  --num_workers 0 `
  --itr 1

# Lookback window: 720
python -u run.py `
  --is_training 1 `
  --root_path ../dataset/traffic/ `
  --data_path traffic.csv `
  --model_id traffic_720_96 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 720 `
  --pred_len 96 `
  --e_layers 4 `
  --enc_in 862 `
  --dec_in 862 `
  --c_out 862 `
  --des 'Exp' `
  --d_model 512 `
  --d_ff 512 `
  --batch_size 16 `
  --learning_rate 0.001 `
  --num_workers 0 `
  --itr 1

