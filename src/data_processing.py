# src/data_processing.py

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def prepare_data(df, sequence_length=60, validation_split=0.2, test_split=0.1, device='cpu'):
    """
    Prepares data for LSTM modeling by adding features, scaling, creating sequences, and splitting into datasets.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing stock data with a DateTime index and required feature columns.
    
    sequence_length : int, optional
        Number of past time steps to include in each input sequence. Default is 60.
    
    validation_split : float, optional
        Proportion of the dataset to include in the validation split. Default is 0.2.
    
    test_split : float, optional
        Proportion of the dataset to include in the test split. Default is 0.1.
    
    device : str, optional
        Device to which tensors will be moved ('cpu' or 'cuda'). Default is 'cpu'.
    
    Returns
    -------
    X_train : torch.Tensor
        Training set features.
    
    X_val : torch.Tensor
        Validation set features.
    
    X_test : torch.Tensor
        Test set features.
    
    y_train : torch.Tensor
        Training set targets.
    
    y_val : torch.Tensor
        Validation set targets.
    
    y_test : torch.Tensor
        Test set targets.
    
    feature_scaler : sklearn.preprocessing.MinMaxScaler
        Scaler fitted on the feature data.
    
    target_scaler : sklearn.preprocessing.MinMaxScaler
        Scaler fitted on the target data.
    """
    # Ensure the index is datetime
    df.index = pd.to_datetime(df.index)

    # Add date-related features
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df['Day_of_Week'] = df.index.dayofweek

    # Define feature columns and target
    features = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
        'SMA_20', 'SMA_50', 'EMA_20', 'RSI',
        'Volume_MA_20', 'High_MA_20', 'Low_MA_20', 'Open_MA_20',
        'Day', 'Month', 'Year', 'Day_of_Week'
    ]
    target = 'Adj Close'

    # Remove rows with missing values
    df.dropna(inplace=True)

    # Initialize scalers and scale features and target
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(df[features])
    scaled_target = target_scaler.fit_transform(df[[target]])

    # Function to create input sequences and corresponding targets
    def create_sequences(features_data, target_data, seq_length):
        xs, ys = [], []
        for i in range(len(features_data) - seq_length):
            xs.append(features_data[i:i + seq_length])
            ys.append(target_data[i + seq_length])
        return np.array(xs), np.array(ys)

    # Create sequences
    X, y = create_sequences(scaled_features, scaled_target, sequence_length)

    # Determine split sizes
    total_samples = len(X)
    test_size = int(test_split * total_samples)
    val_size = int(validation_split * total_samples)
    train_size = total_samples - val_size - test_size

    # Split into training, validation, and test sets
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    # Convert numpy arrays to PyTorch tensors and move to specified device
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_scaler, target_scaler
