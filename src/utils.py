# src/utils.py

import os
import torch
import joblib
from src.models.lstm_model import LSTMModel
from src.config import device, logger  # Import device and logger from config.py

def load_model_and_scalers(ticker, model_dir='saved_models'):
    """
    Loads the trained LSTM model and associated scalers for a given ticker.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        model_dir (str): Path to the directory where models are saved.

    Returns:
        tuple: A tuple containing the loaded model, feature scaler, and target scaler.
               (model, feature_scaler, target_scaler)

    Raises:
        FileNotFoundError: If the model or scaler files are not found.
        Exception: For any other exceptions during loading.
    """
    # Construct the path to the ticker's subdirectory
    ticker_dir = os.path.join(model_dir, ticker)

    # Define paths to the model and scaler files
    model_path = os.path.join(ticker_dir, 'model.pth')
    scaler_path = os.path.join(ticker_dir, 'scalers.pkl')

    # Check if the ticker's directory exists
    if not os.path.isdir(ticker_dir):
        logger.error(f"Directory for ticker '{ticker}' does not exist at {ticker_dir}.")
        raise FileNotFoundError(f"Directory for ticker '{ticker}' does not exist at {ticker_dir}.")

    # Check if the model file exists
    if not os.path.isfile(model_path):
        logger.error(f"Model file not found for ticker '{ticker}' at {model_path}.")
        raise FileNotFoundError(f"Model file not found for ticker '{ticker}' at {model_path}.")

    # Check if the scaler file exists
    if not os.path.isfile(scaler_path):
        logger.error(f"Scaler file not found for ticker '{ticker}' at {scaler_path}.")
        raise FileNotFoundError(f"Scaler file not found for ticker '{ticker}' at {scaler_path}.")

    # Load scalers
    try:
        with open(scaler_path, 'rb') as f:
            scalers = joblib.load(f)
        feature_scaler = scalers['feature_scaler']
        target_scaler = scalers['target_scaler']
        logger.info(f"Scalers loaded successfully for ticker '{ticker}'.")
    except Exception as e:
        logger.error(f"Error loading scalers for ticker '{ticker}': {e}")
        raise Exception(f"Error loading scalers for ticker '{ticker}': {e}")

    # Initialize the model with the correct input size
    input_size = len(feature_scaler.scale_)
    hidden_size = 128
    num_layers = 2
    output_size = 1
    dropout = 0.2

    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)

    # Load the model state
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info(f"Model for '{ticker}' loaded successfully from {model_path}.")
    except Exception as e:
        logger.error(f"Error loading model for ticker '{ticker}': {e}")
        raise Exception(f"Error loading model for ticker '{ticker}': {e}")

    return model, feature_scaler, target_scaler
