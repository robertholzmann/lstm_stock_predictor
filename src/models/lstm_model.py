# src/models/lstm_model.py

import torch
import torch.nn as nn
from src.config import device  # Import device from config.py

class LSTMModel(nn.Module):
    """
    A Long Short-Term Memory (LSTM) network for time series prediction.

    This model processes sequential data to predict future values based on past observations.
    
    Attributes
    ----------
    hidden_size : int
        Number of features in the hidden state of the LSTM.
    
    num_layers : int
        Number of stacked LSTM layers.
    
    lstm : nn.LSTM
        LSTM layer that processes the input sequences.
    
    dropout_layer : nn.Dropout
        Dropout layer applied after the LSTM to prevent overfitting.
    
    fc : nn.Linear
        Fully connected layer that maps the LSTM outputs to the desired output size.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        """
        Initializes the LSTMModel with the given parameters.

        Parameters
        ----------
        input_size : int
            Number of input features per time step.
        
        hidden_size : int
            Number of features in the hidden state of the LSTM.
        
        num_layers : int
            Number of stacked LSTM layers.
        
        output_size : int
            Number of output features.
        
        dropout : float
            Dropout probability for the LSTM and dropout layers.
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer with specified number of layers and dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # Dropout only applied if num_layers > 1
        )

        # Dropout layer to prevent overfitting
        self.dropout_layer = nn.Dropout(p=dropout)

        # Fully connected layer to map LSTM outputs to desired output size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Performs a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_size).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_size).
        """
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Apply dropout to the last time step's output
        out = self.dropout_layer(out[:, -1, :])

        # Pass through the fully connected layer to get the final output
        out = self.fc(out)
        return out
