# PowerShell script for comparing Transformer vs iTransformer on Traffic dataset
# Run with: .\scripts\boost_performance\Traffic\iTransformer.ps1
# Edit $model_name to switch between "Transformer" and "iTransformer"

# Change to the project root directory (AI-Project-FAST)
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $scriptPath))
Set-Location $projectRoot

# Uncomment to run vanilla Transformer first
# $model_name = "Transformer"
$model_name = "iTransformer"

# Prediction length 96
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

# Prediction length 192
python -u run.py `
  --is_training 1 `
  --root_path ../dataset/traffic/ `
  --data_path traffic.csv `
  --model_id traffic_96_192 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 96 `
  --pred_len 192 `
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

# Prediction length 336
python -u run.py `
  --is_training 1 `
  --root_path ../dataset/traffic/ `
  --data_path traffic.csv `
  --model_id traffic_96_336 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 96 `
  --pred_len 336 `
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

# Prediction length 720
python -u run.py `
  --is_training 1 `
  --root_path ../dataset/traffic/ `
  --data_path traffic.csv `
  --model_id traffic_96_720 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 96 `
  --pred_len 720 `
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

