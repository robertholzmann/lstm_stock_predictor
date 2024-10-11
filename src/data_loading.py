# src/data_loading.py

import yfinance as yf
import pandas as pd

def download_ticker_data(ticker, start_date, end_date, save_csv=False):
    """
    Downloads historical stock data for a given ticker symbol between specified dates.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    if save_csv:
        df.to_csv(f'{ticker}_data.csv')
    return df
