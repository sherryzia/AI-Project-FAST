# PowerShell script for iTransformer on ETTh1 dataset
# Run with: .\scripts\multivariate_forecasting\ETT\iTransformer_ETTh1.ps1

# Change to the project root directory (AI-Project-FAST)
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $scriptPath))
Set-Location $projectRoot

$model_name = "iTransformer"

# Prediction length 96
python -u run.py `
  --is_training 1 `
  --root_path ../dataset/ETT-small/ `
  --data_path ETTh1.csv `
  --model_id ETTh1_96_96 `
  --model $model_name `
  --data ETTh1 `
  --features M `
  --seq_len 96 `
  --pred_len 96 `
  --e_layers 2 `
  --enc_in 7 `
  --dec_in 7 `
  --c_out 7 `
  --des 'Exp' `
  --d_model 256 `
  --d_ff 256 `
  --num_workers 0 `
  --itr 1

# Prediction length 192
python -u run.py `
  --is_training 1 `
  --root_path ../dataset/ETT-small/ `
  --data_path ETTh1.csv `
  --model_id ETTh1_96_192 `
  --model $model_name `
  --data ETTh1 `
  --features M `
  --seq_len 96 `
  --pred_len 192 `
  --e_layers 2 `
  --enc_in 7 `
  --dec_in 7 `
  --c_out 7 `
  --des 'Exp' `
  --d_model 256 `
  --d_ff 256 `
  --num_workers 0 `
  --itr 1

# Prediction length 336
python -u run.py `
  --is_training 1 `
  --root_path ../dataset/ETT-small/ `
  --data_path ETTh1.csv `
  --model_id ETTh1_96_336 `
  --model $model_name `
  --data ETTh1 `
  --features M `
  --seq_len 96 `
  --pred_len 336 `
  --e_layers 2 `
  --enc_in 7 `
  --dec_in 7 `
  --c_out 7 `
  --des 'Exp' `
  --d_model 512 `
  --d_ff 512 `
  --itr 1

# Prediction length 720
python -u run.py `
  --is_training 1 `
  --root_path ../dataset/ETT-small/ `
  --data_path ETTh1.csv `
  --model_id ETTh1_96_720 `
  --model $model_name `
  --data ETTh1 `
  --features M `
  --seq_len 96 `
  --pred_len 720 `
  --e_layers 2 `
  --enc_in 7 `
  --dec_in 7 `
  --c_out 7 `
  --des 'Exp' `
  --d_model 512 `
  --d_ff 512 `
  --itr 1

