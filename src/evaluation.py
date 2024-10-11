# src/evaluation.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch

from src.config import get_device  # Import get_device from config.py

def evaluate_model(ticker, model, X_test, y_test, target_scaler, plots_dir, ticker_dir):
    """
    Evaluates the trained LSTM model on the test set, computes performance metrics, 
    and generates relevant plots.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'GOOGL').

    model : torch.nn.Module
        Trained LSTM model.

    X_test : torch.Tensor
        Test set features.

    y_test : torch.Tensor
        Actual target values for the test set.

    target_scaler : sklearn.preprocessing object
        Scaler used to inverse transform the target variable.

    plots_dir : str
        Directory path to save the generated plots.

    ticker_dir : str
        Directory path specific to the ticker for saving metrics and plots.

    Returns
    -------
    None
    """
    device = get_device()
    
    # Move data to the appropriate device
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Generate predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)

    # Convert tensors to NumPy arrays and inverse transform
    y_pred_np = y_pred.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    y_pred_inversed = target_scaler.inverse_transform(y_pred_np).flatten()
    y_test_inversed = target_scaler.inverse_transform(y_test_np).flatten()

    # Compute performance metrics
    rmse = np.sqrt(mean_squared_error(y_test_inversed, y_pred_inversed))
    mae = mean_absolute_error(y_test_inversed, y_pred_inversed)
    r2 = r2_score(y_test_inversed, y_pred_inversed)
    
    def smape(A, F):
        """Calculate Symmetric Mean Absolute Percentage Error."""
        return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F) + 1e-8))
    
    smape_value = smape(y_test_inversed, y_pred_inversed)

    # Calculate Directional Accuracy
    direction_actual = np.sign(np.diff(y_test_inversed))
    direction_predicted = np.sign(np.diff(y_pred_inversed))
    directional_accuracy = np.mean(direction_actual == direction_predicted) * 100

    # Display metrics
    print(f'{ticker} - RMSE: {rmse:.2f}')
    print(f'{ticker} - MAE: {mae:.2f}')
    print(f'{ticker} - R-squared: {r2:.4f}')
    print(f'{ticker} - SMAPE: {smape_value:.2f}%')
    print(f'{ticker} - Directional Accuracy: {directional_accuracy:.2f}%')

    # Save metrics to a text file
    metrics_path = os.path.join(ticker_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f'RMSE: {rmse:.2f}\n')
        f.write(f'MAE: {mae:.2f}\n')
        f.write(f'R-squared: {r2:.4f}\n')
        f.write(f'SMAPE: {smape_value:.2f}%\n')
        f.write(f'Directional Accuracy: {directional_accuracy:.2f}%\n')
    print(f"Metrics saved to {metrics_path}")

    # Plot Residuals vs. Predicted Values
    residuals = y_test_inversed - y_pred_inversed
    plt.figure(figsize=(10, 5))
    plt.scatter(y_pred_inversed, residuals, alpha=0.5)
    plt.hlines(y=0, xmin=y_pred_inversed.min(), xmax=y_pred_inversed.max(), colors='red')
    plt.title(f'{ticker} Residuals vs. Predicted Values')
    plt.xlabel('Predicted Prices')
    plt.ylabel('Residuals')
    plt.savefig(os.path.join(plots_dir, 'residuals_vs_predicted.png'))
    plt.close()
    print("Residuals vs. Predicted Values plot saved.")

    # Plot Histogram of Residuals
    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=50, alpha=0.7)
    plt.title(f'{ticker} Residuals Distribution')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(plots_dir, 'residuals_histogram.png'))
    plt.close()
    print("Residuals histogram saved.")

    # Plot Actual vs. Predicted Prices
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_inversed, label='Actual Price')
    plt.plot(y_pred_inversed, label='Predicted Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Days')
    plt.ylabel('Price USD ($)')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted.png'))
    plt.close()
    print("Actual vs. Predicted Prices plot saved.")
