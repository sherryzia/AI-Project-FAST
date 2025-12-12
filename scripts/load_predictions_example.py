"""
Example script to load and use predictions from iTransformer
Run this after training to see your predictions
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_visualize_predictions(model_id='minimal_test', variable_idx=0, sample_idx=0):
    """
    Load predictions and visualize them
    
    Args:
        model_id: The model_id used when training (e.g., 'minimal_test', 'weather_96_96')
        variable_idx: Which variable to plot (0 = first variable)
        sample_idx: Which sample to plot (0 = first sample)
    """
    
    # Path to results
    results_path = f'./results/{model_id}/'
    
    if not os.path.exists(results_path):
        print(f"Results not found at {results_path}")
        print("Make sure you've run training first!")
        return
    
    # Load predictions and true values
    preds = np.load(os.path.join(results_path, 'pred.npy'))
    trues = np.load(os.path.join(results_path, 'true.npy'))
    metrics = np.load(os.path.join(results_path, 'metrics.npy'))
    
    print("=" * 60)
    print(f"Results for: {model_id}")
    print("=" * 60)
    print(f"\nPredictions shape: {preds.shape}")
    print(f"True values shape: {trues.shape}")
    print(f"\nMetrics:")
    print(f"  MAE (Mean Absolute Error): {metrics[0]:.4f}")
    print(f"  MSE (Mean Squared Error):  {metrics[1]:.4f}")
    print(f"  RMSE (Root Mean Squared Error): {metrics[2]:.4f}")
    print(f"  MAPE (Mean Absolute Percentage Error): {metrics[3]:.4f}%")
    print(f"  MSPE (Mean Squared Percentage Error): {metrics[4]:.4f}%")
    
    # Plot predictions vs actual
    plt.figure(figsize=(14, 6))
    
    # Get the data for the specified sample and variable
    true_values = trues[sample_idx, :, variable_idx]
    pred_values = preds[sample_idx, :, variable_idx]
    
    time_steps = range(len(true_values))
    
    plt.plot(time_steps, true_values, label='Actual', marker='o', linewidth=2, markersize=6)
    plt.plot(time_steps, pred_values, label='Predicted', marker='x', linewidth=2, markersize=8)
    
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(f'Prediction vs Actual - Sample {sample_idx}, Variable {variable_idx}', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = f'./results/{model_id}/prediction_plot_sample{sample_idx}_var{variable_idx}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    plt.show()
    
    # Show some example predictions
    print(f"\nExample Predictions (Sample {sample_idx}, Variable {variable_idx}):")
    print("-" * 60)
    for i in range(min(10, len(true_values))):
        print(f"Time Step {i:2d}: Actual = {true_values[i]:8.4f}, Predicted = {pred_values[i]:8.4f}, Error = {abs(true_values[i] - pred_values[i]):8.4f}")
    
    return preds, trues, metrics


def get_forecast_for_new_data(model_id='minimal_test', num_steps_ahead=12):
    """
    Get forecast values for practical use
    
    Args:
        model_id: The model_id used when training
        num_steps_ahead: How many steps ahead you want to forecast
    """
    
    results_path = f'./results/{model_id}/'
    
    if not os.path.exists(results_path):
        print(f"Results not found at {results_path}")
        return None
    
    preds = np.load(os.path.join(results_path, 'pred.npy'))
    
    # Get the most recent prediction (last sample)
    latest_forecast = preds[-1, :num_steps_ahead, :]
    
    print(f"\nLatest Forecast (Next {num_steps_ahead} time steps):")
    print("=" * 60)
    print(f"Shape: {latest_forecast.shape} = [time_steps, variables]")
    print("\nForecast values:")
    print(latest_forecast)
    
    return latest_forecast


if __name__ == '__main__':
    # Example 1: Visualize predictions from your test run
    print("Loading predictions from minimal_test...")
    load_and_visualize_predictions(model_id='minimal_test', variable_idx=0, sample_idx=0)
    
    # Example 2: Get forecast values
    print("\n" + "=" * 60)
    forecast = get_forecast_for_new_data(model_id='minimal_test', num_steps_ahead=12)
    
    print("\n" + "=" * 60)
    print("To use for your own model, change 'minimal_test' to your model_id")
    print("Example: load_and_visualize_predictions(model_id='weather_96_96', variable_idx=0)")

