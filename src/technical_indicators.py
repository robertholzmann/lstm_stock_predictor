# src/technical_indicators.py

import pandas as pd

def add_technical_indicators(df):
    """
    Enhances the stock DataFrame by adding key technical indicators.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing stock data with at least the following columns:
        ['Adj Close', 'Volume', 'High', 'Low', 'Open'].

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame augmented with additional technical indicator columns:
        - SMA_20: 20-day Simple Moving Average of Adjusted Close.
        - SMA_50: 50-day Simple Moving Average of Adjusted Close.
        - EMA_20: 20-day Exponential Moving Average of Adjusted Close.
        - RSI: 14-day Relative Strength Index.
        - Volume_MA_20: 20-day Moving Average of Volume.
        - High_MA_20: 20-day Moving Average of High prices.
        - Low_MA_20: 20-day Moving Average of Low prices.
        - Open_MA_20: 20-day Moving Average of Open prices.
    """
    df = df.copy()

    # Calculate Simple Moving Averages
    df['SMA_20'] = df['Adj Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Adj Close'].rolling(window=50).mean()

    # Calculate Exponential Moving Average
    df['EMA_20'] = df['Adj Close'].ewm(span=20, adjust=False).mean()

    # Calculate Relative Strength Index (RSI)
    delta = df['Adj Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    window = 14
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Calculate Moving Averages for Volume and Price
    ma_window = 20
    df['Volume_MA_20'] = df['Volume'].rolling(window=ma_window).mean()
    df['High_MA_20'] = df['High'].rolling(window=ma_window).mean()
    df['Low_MA_20'] = df['Low'].rolling(window=ma_window).mean()
    df['Open_MA_20'] = df['Open'].rolling(window=ma_window).mean()

    return df
