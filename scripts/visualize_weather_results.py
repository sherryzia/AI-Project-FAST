"""
Comprehensive Weather Prediction Visualization Script
Visualizes results from weather prediction model
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

# Weather variable names (21 variables)
WEATHER_VARIABLES = [
    "Temperature", "Humidity", "Pressure", "Wind Speed", "Wind Direction",
    "Visibility", "Dew Point", "Cloud Cover", "Precipitation", "Solar Radiation",
    "UV Index", "Air Quality", "Pollen Count", "Barometric Pressure", "Heat Index",
    "Wind Chill", "Rainfall", "Snowfall", "Fog", "Thunderstorm", "Hail"
]

def load_results(model_id):
    """Load prediction results"""
    results_path = f'./results/{model_id}/'
    
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results not found at {results_path}. Please run training and testing first.")
    
    preds = np.load(os.path.join(results_path, 'pred.npy'))
    trues = np.load(os.path.join(results_path, 'true.npy'))
    metrics = np.load(os.path.join(results_path, 'metrics.npy'))
    
    return preds, trues, metrics, results_path

def print_metrics(metrics, model_id):
    """Print performance metrics"""
    print("=" * 70)
    print(f"  Weather Prediction Results: {model_id}")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'Value':<20} {'Description'}")
    print("-" * 70)
    print(f"{'MAE (Mean Absolute Error)':<30} {metrics[0]:<20.4f} Average absolute difference")
    print(f"{'MSE (Mean Squared Error)':<30} {metrics[1]:<20.4f} Average squared difference")
    print(f"{'RMSE (Root Mean Squared Error)':<30} {metrics[2]:<20.4f} Square root of MSE")
    print(f"{'MAPE (Mean Absolute % Error)':<30} {metrics[3]:<20.4f}% Average percentage error")
    print(f"{'MSPE (Mean Squared % Error)':<30} {metrics[4]:<20.4f}% Average squared percentage error")
    print("=" * 70)

def plot_single_variable(preds, trues, variable_idx, sample_idx, results_path, model_id):
    """Plot predictions vs actual for a single variable"""
    true_values = trues[sample_idx, :, variable_idx]
    pred_values = preds[sample_idx, :, variable_idx]
    time_steps = range(len(true_values))
    
    plt.figure(figsize=(14, 6))
    plt.plot(time_steps, true_values, label='Actual', marker='o', linewidth=2, markersize=5, alpha=0.7)
    plt.plot(time_steps, pred_values, label='Predicted', marker='x', linewidth=2, markersize=7, alpha=0.8)
    
    # Calculate error for this sample
    mae_sample = np.mean(np.abs(true_values - pred_values))
    mse_sample = np.mean((true_values - pred_values) ** 2)
    
    plt.xlabel('Time Step (Hours)', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(f'Weather Prediction - Sample {sample_idx}, Variable {variable_idx}\n'
              f'MAE: {mae_sample:.4f}, MSE: {mse_sample:.4f}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(results_path, f'prediction_sample{sample_idx}_var{variable_idx}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

def plot_multiple_variables(preds, trues, variable_indices, sample_idx, results_path, model_id):
    """Plot multiple variables in subplots"""
    num_vars = len(variable_indices)
    cols = min(3, num_vars)
    rows = (num_vars + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if num_vars == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, var_idx in enumerate(variable_indices):
        ax = axes[idx]
        true_values = trues[sample_idx, :, var_idx]
        pred_values = preds[sample_idx, :, var_idx]
        time_steps = range(len(true_values))
        
        ax.plot(time_steps, true_values, label='Actual', marker='o', markersize=3, alpha=0.7)
        ax.plot(time_steps, pred_values, label='Predicted', marker='x', markersize=4, alpha=0.8)
        ax.set_xlabel('Time Step (Hours)', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(f'Variable {var_idx}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(num_vars, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Weather Predictions - Sample {sample_idx} (Multiple Variables)', fontsize=14)
    plt.tight_layout()
    
    plot_path = os.path.join(results_path, f'prediction_sample{sample_idx}_multi_var.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

def plot_error_distribution(preds, trues, results_path):
    """Plot error distribution"""
    errors = preds - trues
    mae_per_sample = np.mean(np.abs(errors), axis=(1, 2))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(mae_per_sample, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Mean Absolute Error per Sample', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Error Distribution Across Samples', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(mae_per_sample)
    plt.ylabel('Mean Absolute Error', fontsize=11)
    plt.title('Error Box Plot', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_path, 'error_distribution.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

def plot_variable_wise_errors(preds, trues, results_path):
    """Plot errors for each variable"""
    errors = preds - trues
    mae_per_variable = np.mean(np.abs(errors), axis=(0, 1))
    
    plt.figure(figsize=(14, 6))
    variable_indices = range(len(mae_per_variable))
    plt.bar(variable_indices, mae_per_variable, alpha=0.7, edgecolor='black')
    plt.xlabel('Variable Index', fontsize=12)
    plt.ylabel('Mean Absolute Error', fontsize=12)
    plt.title('Error per Variable (Lower is Better)', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plot_path = os.path.join(results_path, 'variable_wise_errors.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

def print_prediction_examples(preds, trues, num_samples=3, num_variables=3):
    """Print example predictions"""
    print("\n" + "=" * 70)
    print("  Example Predictions")
    print("=" * 70)
    
    for sample_idx in range(min(num_samples, len(preds))):
        print(f"\nSample {sample_idx}:")
        print("-" * 70)
        print(f"{'Time':<8} {'Variable':<10} {'Actual':<12} {'Predicted':<12} {'Error':<12} {'Error %':<10}")
        print("-" * 70)
        
        for var_idx in range(min(num_variables, preds.shape[2])):
            for time_idx in range(min(10, preds.shape[1])):
                actual = trues[sample_idx, time_idx, var_idx]
                predicted = preds[sample_idx, time_idx, var_idx]
                error = abs(actual - predicted)
                error_pct = (error / abs(actual)) * 100 if actual != 0 else 0
                
                print(f"{time_idx:<8} {var_idx:<10} {actual:<12.4f} {predicted:<12.4f} {error:<12.4f} {error_pct:<10.2f}%")
            print()

def main():
    parser = argparse.ArgumentParser(description='Visualize weather prediction results')
    parser.add_argument('--model_id', type=str, default='weather_test', 
                       help='Model ID to visualize (default: weather_test)')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index to plot (default: 0)')
    parser.add_argument('--variable_idx', type=int, default=0,
                       help='Variable index to plot (default: 0)')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize (default: 5)')
    
    args = parser.parse_args()
    
    try:
        # Load results
        print("Loading results...")
        preds, trues, metrics, results_path = load_results(args.model_id)
        
        print(f"\nData loaded successfully!")
        print(f"Predictions shape: {preds.shape}")
        print(f"True values shape: {trues.shape}")
        print(f"Number of samples: {preds.shape[0]}")
        print(f"Prediction length: {preds.shape[1]} hours")
        print(f"Number of variables: {preds.shape[2]}")
        
        # Print metrics
        print_metrics(metrics, args.model_id)
        
        # Print example predictions
        print_prediction_examples(preds, trues, num_samples=args.num_samples, num_variables=3)
        
        # Create visualizations
        print("\n" + "=" * 70)
        print("  Creating Visualizations")
        print("=" * 70)
        
        plots_created = []
        
        # 1. Single variable plot
        print(f"\n1. Plotting sample {args.sample_idx}, variable {args.variable_idx}...")
        plot_path = plot_single_variable(preds, trues, args.variable_idx, args.sample_idx, 
                                        results_path, args.model_id)
        plots_created.append(plot_path)
        print(f"   Saved: {plot_path}")
        
        # 2. Multiple variables plot
        print(f"\n2. Plotting multiple variables for sample {args.sample_idx}...")
        var_indices = list(range(min(6, preds.shape[2])))  # First 6 variables
        plot_path = plot_multiple_variables(preds, trues, var_indices, args.sample_idx, 
                                           results_path, args.model_id)
        plots_created.append(plot_path)
        print(f"   Saved: {plot_path}")
        
        # 3. Error distribution
        print("\n3. Plotting error distribution...")
        plot_path = plot_error_distribution(preds, trues, results_path)
        plots_created.append(plot_path)
        print(f"   Saved: {plot_path}")
        
        # 4. Variable-wise errors
        print("\n4. Plotting variable-wise errors...")
        plot_path = plot_variable_wise_errors(preds, trues, results_path)
        plots_created.append(plot_path)
        print(f"   Saved: {plot_path}")
        
        # Summary
        print("\n" + "=" * 70)
        print("  Visualization Complete!")
        print("=" * 70)
        print(f"\nAll plots saved in: {results_path}")
        print(f"\nPlots created:")
        for plot in plots_created:
            print(f"  - {os.path.basename(plot)}")
        
        print("\nTo view a specific variable, use:")
        print(f"  python scripts/visualize_weather_results.py --model_id {args.model_id} --variable_idx <var> --sample_idx <sample>")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run the training and testing first:")
        print("  .\\scripts\\test_weather_prediction.ps1")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())

