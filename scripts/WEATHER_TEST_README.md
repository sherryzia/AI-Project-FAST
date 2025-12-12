# Weather Prediction Test Script

## Quick Start

Run the complete weather prediction test (training + testing + visualization):

```powershell
.\scripts\test_weather_prediction.ps1
```

This script will:
1. ✅ Train the iTransformer model on weather data
2. ✅ Test the model and generate predictions
3. ✅ Visualize results with multiple plots
4. ✅ Display metrics and example predictions

---

## What You'll Get

### Results Location
```
./results/weather_test/
  ├── pred.npy              # All predictions
  ├── true.npy              # Actual values
  ├── metrics.npy           # Performance metrics
  ├── prediction_sample0_var0.png        # Single variable plot
  ├── prediction_sample0_multi_var.png  # Multiple variables plot
  ├── error_distribution.png             # Error analysis
  └── variable_wise_errors.png          # Per-variable errors
```

### Metrics Displayed
- **MAE** (Mean Absolute Error): Average prediction error
- **MSE** (Mean Squared Error): Average squared error
- **RMSE** (Root Mean Squared Error): Standard deviation of errors
- **MAPE** (Mean Absolute Percentage Error): Percentage error
- **MSPE** (Mean Squared Percentage Error): Squared percentage error

---

## Customization

### Change Prediction Length
Edit `test_weather_prediction.ps1`:
```powershell
$pred_len = 192  # Predict next 192 hours instead of 96
```

### Change Training Duration
```powershell
$train_epochs = 10  # Train for more epochs
```

### Visualize Specific Results
```powershell
python scripts\visualize_weather_results.py --model_id weather_test --variable_idx 0 --sample_idx 5
```

---

## Weather Variables

The weather dataset has **21 variables**:
- Temperature, Humidity, Pressure
- Wind Speed, Wind Direction
- Visibility, Dew Point, Cloud Cover
- Precipitation, Solar Radiation
- UV Index, Air Quality, Pollen Count
- And more...

---

## Example Output

```
========================================
  Weather Prediction Test Script
========================================

Configuration:
  Model ID: weather_test
  Sequence Length: 96 (input: past 96 hours)
  Prediction Length: 96 (forecast: next 96 hours)
  Training Epochs: 5
  Batch Size: 32

========================================
Step 1: Training the Model
========================================
Starting training...
[Training progress...]
Training completed!

========================================
Step 2: Testing the Model
========================================
Starting testing...
[Testing progress...]
Testing completed!

========================================
Step 3: Visualizing Results
========================================
Loading and visualizing results...

Results for: weather_test
Predictions shape: (2869, 96, 21)
True values shape: (2869, 96, 21)

Metrics:
  MAE: 0.1234
  MSE: 0.2345
  RMSE: 0.4843
  MAPE: 2.34%
  MSPE: 5.67%

Plots created:
  - prediction_sample0_var0.png
  - prediction_sample0_multi_var.png
  - error_distribution.png
  - variable_wise_errors.png
```

---

## Troubleshooting

### Dataset Not Found
Make sure the weather dataset is at:
```
../dataset/weather/weather.csv
```

### Out of Memory
Reduce batch size in the script:
```powershell
$batch_size = 16  # Smaller batch size
```

### Training Takes Too Long
Reduce epochs:
```powershell
$train_epochs = 3  # Fewer epochs for faster testing
```

---

## Next Steps

After running the test:

1. **Check the plots** in `./results/weather_test/`
2. **Load predictions** in your code:
   ```python
   import numpy as np
   preds = np.load('./results/weather_test/pred.npy')
   ```
3. **Use for forecasting**: The model can now predict future weather!

---

## Advanced Usage

### Train with Different Parameters
Modify the script or run directly:
```powershell
python run.py --is_training 1 --model_id weather_custom --seq_len 192 --pred_len 336 ...
```

### Compare Multiple Models
Run the script multiple times with different `model_id` values and compare results.

