# src/training.py

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import joblib

from src.models.lstm_model import LSTMModel
from src.data_processing import prepare_data
from src.config import get_device  # Import get_device from config.py

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def train_model_for_ticker(ticker, df, model_dir='saved_models'):
    """
    Trains an LSTM model for a specific stock ticker, saves the trained model along with 
    associated scalers and loss values, and generates a plot of training and validation losses.

    Parameters
    ----------
    ticker : str
        The stock ticker symbol (e.g., 'AAPL', 'GOOGL') for which the model is being trained.
    
    df : pandas.DataFrame
        The DataFrame containing the historical stock data. It is expected to have a DateTime 
        index and relevant feature columns required for training.
    
    model_dir : str, optional
        The base directory where the trained models and related files will be saved. 
        Defaults to `'saved_models'`.
    
    Returns
    -------
    model : torch.nn.Module
        The trained LSTM model loaded with the best state (i.e., the state that achieved 
        the lowest validation loss).
    
    X_test : torch.Tensor
        The test set features used for evaluating the model's performance.
    
    y_test : torch.Tensor
        The actual target values corresponding to `X_test`.
    
    target_scaler : sklearn.preprocessing object
        The scaler object used to normalize the target variable. Useful for inverse 
        transforming predictions back to the original scale.
    
    plots_dir : str
        The directory path where the training and validation loss plot is saved.
    
    ticker_dir : str
        The directory path specific to the ticker where all related files (model, scalers, 
        loss values) are stored.
    
    Raises
    ------
    FileNotFoundError
        If the necessary directories cannot be created or accessed.
    
    Notes
    -----
    - Ensure that the required libraries (`torch`, `joblib`, `matplotlib`, etc.) are installed.
    - The function employs early stopping based on validation loss to prevent overfitting.
    """

    # Determine the computation device (CPU or GPU)
    device = get_device()

    # Prepare the dataset for training, validation, and testing
    X_train, X_val, X_test, y_train, y_val, y_test, feature_scaler, target_scaler = prepare_data(df, device=device)

    # Define hyperparameters for the LSTM model
    input_size = X_train.shape[2]      # Number of input features
    hidden_size = 128                  # Number of features in the hidden state
    num_layers = 2                     # Number of stacked LSTM layers
    output_size = 1                    # Number of output features
    num_epochs = 300                   # Maximum number of training epochs
    learning_rate = 0.0001             # Learning rate for the optimizer
    dropout = 0.2                      # Dropout rate for regularization
    patience = 200                     # Number of epochs with no improvement for early stopping

    # Initialize the LSTM model, loss function, and optimizer
    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize variables for early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    # Lists to record training and validation loss over epochs
    train_loss_values = []
    val_loss_values = []

    # Training loop
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        optimizer.zero_grad()  # Reset gradients

        # Forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(X_train)
        loss = criterion(outputs, y_train.view(-1, 1))  # Compute loss

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        optimizer.step()  # Update model parameters

        # Record training loss
        train_loss = loss.item()
        train_loss_values.append(train_loss)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val.view(-1, 1)).item()
            val_loss_values.append(val_loss)

        # Check for improvement in validation loss for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()  # Save the best model state
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break  # Exit training loop if no improvement

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Load the best model weights after training
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Define directory paths for saving model and related files
    ticker_dir = os.path.join(model_dir, ticker)
    plots_dir = os.path.join(ticker_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)  # Create directories if they don't exist

    # Save the trained model's state dictionary
    model_path = os.path.join(ticker_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved at: {model_path}")

    # Save the feature and target scalers for future data preprocessing
    scaler_path = os.path.join(ticker_dir, 'scalers.pkl')
    with open(scaler_path, 'wb') as f:
        joblib.dump({
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler
        }, f)
    print(f"Scalers saved at: {scaler_path}")

    # Save the training and validation loss values for analysis or plotting
    loss_values_path = os.path.join(ticker_dir, 'loss_values.npy')
    np.save(loss_values_path, np.array([train_loss_values, val_loss_values]))
    print(f"Loss values saved at: {loss_values_path}")

    # Generate and save a plot of training and validation loss over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_loss_values) + 1), train_loss_values, label='Training Loss')
    plt.plot(range(1, len(val_loss_values) + 1), val_loss_values, label='Validation Loss')
    plt.title(f'{ticker} Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plot_path = os.path.join(plots_dir, 'train_val_loss.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Training and validation loss plot saved at: {plot_path}")

    return model, X_test, y_test, target_scaler, plots_dir, ticker_dir
