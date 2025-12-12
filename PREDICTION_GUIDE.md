# iTransformer Prediction Guide

## 📊 Understanding Your Results

### What the Numbers Mean

From your test run, you got:
- **MSE: 0.5308** (Mean Squared Error)
- **MAE: 0.4747** (Mean Absolute Error)

**What this means:**
- **MSE**: Average squared difference between predicted and actual values (lower is better)
- **MAE**: Average absolute difference between predicted and actual values (lower is better)
- **Lower values = Better predictions**

### Your Test Results Explained

```
Test Loss: 0.5308
MSE: 0.5308
MAE: 0.4747
```

This means:
- On average, your predictions are off by **0.47 units** (MAE)
- The model predicted **12 time steps** into the future for **7 variates**
- Tested on **2,869 samples**

---

## 📁 Where Your Results Are Saved

After running a script, results are saved in:

### 1. **Predictions & True Values**
```
./results/<model_id>/
  ├── pred.npy      # Your predictions (NumPy array)
  ├── true.npy      # Actual values (for comparison)
  └── metrics.npy   # All metrics [MAE, MSE, RMSE, MAPE, MSPE]
```

### 2. **Model Checkpoint**
```
./checkpoints/<model_id>/
  └── checkpoint.pth  # Trained model (can reload for predictions)
```

### 3. **Visualizations**
```
./test_results/<model_id>/
  └── *.pdf  # Plot files showing predictions vs actual
```

### 4. **Text Results**
```
result_long_term_forecast.txt  # Summary of all runs
```

---

## 🔮 How to Use for Real Predictions

### Scenario 1: Predict Weather

**Step 1: Prepare Your Data**
- CSV file with:
  - First column: `date` (timestamp)
  - Remaining columns: Weather variables (temperature, humidity, pressure, etc.)

**Step 2: Train the Model**
```powershell
python run.py `
  --is_training 1 `
  --root_path ../dataset/weather/ `
  --data_path weather.csv `
  --model_id weather_forecast `
  --model iTransformer `
  --data custom `
  --features M `
  --seq_len 96 `
  --pred_len 24 `
  --e_layers 3 `
  --enc_in 21 `
  --dec_in 21 `
  --c_out 21 `
  --d_model 512 `
  --d_ff 512 `
  --batch_size 32 `
  --train_epochs 10 `
  --num_workers 0 `
  --itr 1
```

**Step 3: Get Predictions**
After training, predictions are automatically saved in `./results/weather_forecast/pred.npy`

**Step 4: Load and Use Predictions**
```python
import numpy as np

# Load predictions
predictions = np.load('./results/weather_forecast/pred.npy')
# Shape: [num_samples, pred_len, num_variates]
# Example: [2869, 24, 21] = 2869 samples, 24 time steps ahead, 21 weather variables

# predictions[0] = first sample's forecast (24 time steps, 21 variables)
# predictions[0, :, 0] = first variable (e.g., temperature) for 24 time steps
```

---

### Scenario 2: Predict Traffic

**Step 1: Train on Traffic Data**
```powershell
python run.py `
  --is_training 1 `
  --root_path ../dataset/traffic/ `
  --data_path traffic.csv `
  --model_id traffic_forecast `
  --model iTransformer `
  --data custom `
  --features M `
  --seq_len 96 `
  --pred_len 96 `
  --e_layers 4 `
  --enc_in 862 `
  --dec_in 862 `
  --c_out 862 `
  --d_model 512 `
  --d_ff 512 `
  --batch_size 16 `
  --learning_rate 0.001 `
  --num_workers 0 `
  --itr 1
```

**Step 2: Predictions Format**
- **Shape**: `[num_samples, pred_len, num_variates]`
- **Example**: `[3413, 96, 862]` = 3413 samples, 96 time steps ahead, 862 traffic sensors

---

## 📈 Understanding Prediction Output

### Prediction Array Structure

```python
predictions.shape = [num_samples, pred_len, num_variates]

# Example from your test:
predictions.shape = [2869, 12, 7]

# This means:
# - 2869 different time windows
# - Each predicts 12 time steps into the future
# - Each time step has 7 variables (for ETTh1 dataset)
```

### How to Interpret

**For Weather Prediction:**
```python
predictions[0, :, 0]  # Temperature forecast for next 24 hours
predictions[0, :, 1]  # Humidity forecast for next 24 hours
predictions[0, :, 2]  # Pressure forecast for next 24 hours
# ... etc for all 21 weather variables
```

**For Traffic Prediction:**
```python
predictions[0, :, 0]  # Traffic sensor 1 forecast for next 96 time steps
predictions[0, :, 1]  # Traffic sensor 2 forecast for next 96 time steps
# ... etc for all 862 sensors
```

---

## 🎯 Real-World Usage Examples

### Example 1: Weather Forecast API

```python
import numpy as np
import torch

# Load trained model
model = torch.load('./checkpoints/weather_forecast/checkpoint.pth')

# Prepare new data (last 96 hours of weather data)
new_data = load_recent_weather_data()  # Shape: [1, 96, 21]

# Predict next 24 hours
with torch.no_grad():
    forecast = model(new_data)  # Shape: [1, 24, 21]

# Extract specific forecasts
temperature_forecast = forecast[0, :, 0]  # Next 24 hours temperature
humidity_forecast = forecast[0, :, 1]     # Next 24 hours humidity
```

### Example 2: Traffic Prediction Service

```python
# Load predictions from saved file
traffic_predictions = np.load('./results/traffic_forecast/pred.npy')

# Get forecast for specific sensor
sensor_5_forecast = traffic_predictions[0, :, 4]  # Next 96 time steps for sensor 5

# Convert to actual traffic counts (if data was normalized)
# Use inverse_transform from the dataset
```

---

## 📝 Prediction Parameters Explained

| Parameter | Meaning | Example |
|-----------|---------|---------|
| `seq_len` | How many past time steps to use | 96 = use last 96 hours |
| `pred_len` | How many future steps to predict | 24 = predict next 24 hours |
| `enc_in` | Number of input variables | 21 = 21 weather variables |
| `c_out` | Number of output variables | 21 = predict all 21 variables |

---

## 🔍 Checking Your Predictions

### View Saved Predictions

```python
import numpy as np
import matplotlib.pyplot as plt

# Load results
preds = np.load('./results/minimal_test/pred.npy')
trues = np.load('./results/minimal_test/true.npy')
metrics = np.load('./results/minimal_test/metrics.npy')

print(f"Predictions shape: {preds.shape}")
print(f"True values shape: {trues.shape}")
print(f"MAE: {metrics[0]:.4f}")
print(f"MSE: {metrics[1]:.4f}")

# Plot first variable of first sample
plt.figure(figsize=(12, 6))
plt.plot(trues[0, :, 0], label='Actual', marker='o')
plt.plot(preds[0, :, 0], label='Predicted', marker='x')
plt.legend()
plt.title('Prediction vs Actual')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.show()
```

---

## 🚀 Quick Prediction Workflow

### For Weather:
1. **Train**: Use weather dataset script
2. **Predict**: Model automatically generates predictions
3. **Use**: Load `pred.npy` for your forecast values

### For Traffic:
1. **Train**: Use traffic dataset script  
2. **Predict**: Get forecasts for all sensors
3. **Use**: Extract specific sensor predictions

### For Custom Data:
1. **Format**: CSV with timestamp + variables
2. **Train**: Set `--data custom` and specify `--enc_in`
3. **Predict**: Same process, get your forecasts

---

## 💡 Key Takeaways

1. **MSE/MAE**: Lower = Better predictions
2. **Predictions**: Saved as NumPy arrays in `./results/`
3. **Format**: `[samples, time_steps, variables]`
4. **Use Cases**: Weather, traffic, stock prices, energy demand, etc.
5. **Model**: Saved in `./checkpoints/` for future predictions

---

## 📞 Next Steps

1. **Train on your data**: Use the appropriate script
2. **Check results**: Look in `./results/<model_id>/`
3. **Load predictions**: Use `np.load()` to get your forecasts
4. **Visualize**: Plot predictions vs actuals
5. **Deploy**: Use the saved model for real-time predictions

