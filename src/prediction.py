# src/prediction.py

import pandas as pd
import torch
import yfinance as yf
from datetime import timedelta
from src.technical_indicators import add_technical_indicators

def predict_future_prices(ticker, query_dates, model, feature_scaler, target_scaler, recent_data, sequence_length=60, device='cpu'):
    """
    Predict future Adjusted Close prices for a given ticker and list of dates.

    Args:
        ticker (str): Stock ticker symbol.
        query_dates (list): List of dates as strings (e.g., ['2024-10-01', '2024-10-02']).
        model (nn.Module): Trained LSTM model.
        feature_scaler (MinMaxScaler): Scaler used for feature scaling.
        target_scaler (MinMaxScaler): Scaler used for target scaling.
        recent_data (pd.DataFrame): DataFrame containing historical data up to the day before predictions.
        sequence_length (int): Number of past days used for each prediction.
        device (torch.device): Device to perform computations on.

    Returns:
        dict: Dictionary with dates as keys and dictionaries containing predicted price,
              actual price (if available), and error percentage.
    """
    # Convert query_dates to datetime objects
    query_dates = [pd.to_datetime(date) for date in query_dates]

    # Filter out non-trading days (weekends)
    trading_days = [date for date in query_dates if date.weekday() < 5]

    if not trading_days:
        print("No trading days in the provided query dates.")
        return {}

    # Prepare date range for fetching actual data
    start_date = (trading_days[0] - timedelta(days=sequence_length + 50)).strftime('%Y-%m-%d')
    end_date = (trading_days[-1] + timedelta(days=1)).strftime('%Y-%m-%d')  # Include the last day

    # Fetch actual data for technical indicators and comparison
    actual_data = yf.download(ticker, start=start_date, end=end_date)

    # Ensure recent_data includes up to the day before the first prediction date
    last_sequence_date = trading_days[0] - timedelta(days=1)
    if last_sequence_date > recent_data.index[-1]:
        # Fetch additional data if needed
        additional_data = yf.download(
            ticker,
            start=(recent_data.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d'),
            end=(last_sequence_date + timedelta(days=1)).strftime('%Y-%m-%d')
        )
        if not additional_data.empty:
            recent_data = pd.concat([recent_data, additional_data])
            recent_data.sort_index(inplace=True)
            print(f"Fetched additional data for {ticker} up to {last_sequence_date.strftime('%Y-%m-%d')}")

    # Add technical indicators to recent_data
    recent_data = add_technical_indicators(recent_data)
    recent_data.dropna(inplace=True)

    # Initialize predictions dictionary
    predictions = {}

    # Features to be used
    features = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
        'SMA_20', 'SMA_50', 'EMA_20', 'RSI',
        'Volume_MA_20', 'High_MA_20', 'Low_MA_20', 'Open_MA_20',
        'Day', 'Month', 'Year', 'Day_of_Week'
    ]

    # Initialize last_sequence with recent_data
    last_sequence = recent_data.copy(deep=True)

    for date in trading_days:
        date_str = date.strftime('%Y-%m-%d')
        print(f"\nPredicting for {ticker} on {date_str}")

        # Prepare the input sequence without the new row
        temp_sequence = last_sequence[-sequence_length:].copy()

        # Add date-related features
        temp_sequence.loc[:, 'Day'] = temp_sequence.index.day
        temp_sequence.loc[:, 'Month'] = temp_sequence.index.month
        temp_sequence.loc[:, 'Year'] = temp_sequence.index.year
        temp_sequence.loc[:, 'Day_of_Week'] = temp_sequence.index.dayofweek

        # Fill any missing values
        temp_sequence = temp_sequence.ffill().bfill()

        # Prepare input features
        X_input = temp_sequence[features].values

        # Scale features
        scaled_features = feature_scaler.transform(X_input)

        # Ensure input shape is correct
        X_input_tensor = torch.tensor(scaled_features.reshape(1, sequence_length, -1), dtype=torch.float32).to(device)

        # Make prediction
        model.eval()
        with torch.no_grad():
            y_pred = model(X_input_tensor)

        # Inverse transform prediction
        y_pred_inversed = target_scaler.inverse_transform(y_pred.cpu().numpy())

        predicted_adj_close = y_pred_inversed.flatten()[0]
        print(f"Predicted Adjusted Close: ${predicted_adj_close:.2f}")

        # Get actual Adj Close if available
        if date in actual_data.index:
            actual_adj_close = actual_data.at[date, 'Adj Close']
            error_pct = ((predicted_adj_close - actual_adj_close) / actual_adj_close) * 100
            print(f"Actual Adjusted Close: ${actual_adj_close:.2f}, Error: {error_pct:.2f}%")
        else:
            actual_adj_close = None
            error_pct = None
            print(f"Actual Adjusted Close not available for {date_str}")

        # Store prediction
        predictions[date_str] = {
            'Predicted Price': predicted_adj_close,
            'Actual Price': actual_adj_close,
            'Error Percentage': error_pct
        }

        # Now, create new_row for the date
        new_row = pd.DataFrame(index=[date], columns=recent_data.columns)

        # Estimate features for the new date
        n_days = 20  # Window for rolling means

        # Use actual data if available
        if date in actual_data.index:
            actual_row = actual_data.loc[date]
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                new_row.at[date, col] = actual_row[col]
            print(f"Used actual market data for {date_str}")
        else:
            # Estimate using rolling means
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                new_row.at[date, col] = last_sequence[col].iloc[-n_days:].mean()
            print(f"Estimated market data for {date_str} using rolling means")

        # Set 'Adj Close' in new_row to actual or predicted value
        new_adj_close = actual_adj_close if actual_adj_close is not None else predicted_adj_close
        new_row.at[date, 'Adj Close'] = new_adj_close
        print(f"Set 'Adj Close' for {date_str} to {new_adj_close}")

        # Append new_row to last_sequence
        last_sequence = pd.concat([last_sequence, new_row])

        # Recalculate technical indicators on last_sequence
        last_sequence = add_technical_indicators(last_sequence)

        # Fill any missing values in last_sequence
        last_sequence = last_sequence.ffill().bfill()

    return predictions


def predict_future_prices_multiple_tickers(tickers, query_dates, models_info, ticker_data, sequence_length=60, device='cpu'):
    """
    Predict future Adjusted Close prices for multiple tickers.

    Args:
        tickers (list): List of stock ticker symbols.
        query_dates (list): List of dates as strings for prediction.
        models_info (dict): Dictionary with ticker symbols as keys and a dictionary containing 'model',
                            'feature_scaler', and 'target_scaler' as values.
        ticker_data (dict): Dictionary with ticker symbols as keys and their corresponding historical DataFrames.
        sequence_length (int): Number of past days used for each prediction.
        device (torch.device): Device to perform computations on.

    Returns:
        dict: Dictionary with ticker symbols as keys and their prediction dictionaries as values.
    """
    all_predictions = {}

    for ticker in tickers:
        print(f"\nPredicting future prices for {ticker}...")
        if ticker not in models_info:
            print(f"No model information found for {ticker}. Skipping.")
            continue

        model_info = models_info[ticker]
        model = model_info['model']
        feature_scaler = model_info['feature_scaler']
        target_scaler = model_info['target_scaler']
        recent_data = ticker_data[ticker]

        predictions = predict_future_prices(
            ticker,
            query_dates,
            model,
            feature_scaler,
            target_scaler,
            recent_data,
            sequence_length,
            device  # Use the passed device
        )
        all_predictions[ticker] = predictions

    return all_predictions
