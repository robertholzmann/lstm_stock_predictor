# app.py

import streamlit as st
import os
import torch
import pandas as pd
from datetime import datetime, timedelta
from PIL import Image  # For image handling

# Import necessary functions
from src.data_loading import download_ticker_data
from src.utils import load_model_and_scalers
from src.prediction import predict_future_prices_multiple_tickers
from src.technical_indicators import add_technical_indicators
from src.config import get_device  # Import get_device from config.py

def load_metrics(ticker, model_dir='saved_models'):
    """
    Loads performance metrics from the metrics.txt file for a given ticker.
    """
    metrics_path = os.path.join(model_dir, ticker, 'metrics.txt')
    if os.path.isfile(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = f.read()
        return metrics
    else:
        return "No metrics available."

def display_plots(ticker, model_dir='saved_models'):
    """
    Displays the 'actual_vs_predicted.png' plot for a given ticker.
    """
    plots_dir = os.path.join(model_dir, ticker, 'plots')
    plot_file = 'actual_vs_predicted.png'
    plot_path = os.path.join(plots_dir, plot_file)
    if os.path.isfile(plot_path):
        image = Image.open(plot_path)
        st.image(image, use_column_width=True)  # Removed the caption to eliminate text below the image

# Apply custom CSS for better visuals (optional)
def local_css():
    st.markdown("""
        <style>
        .title {
            color: #4B8BBE;
            text-align: center;
        }
        .subheader {
            color: #FFA500;
        }
        /* Customize the sidebar */
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        /* Style the table headers */
        thead th {
            background-color: #4B8BBE;
            color: white;
            font-weight: bold;
        }
        /* Hide index column in tables */
        .dataframe tbody tr th {
            display: none;
        }
        .dataframe thead th:first-child {
            display: none;
        }
        /* Button Styling */
        .stButton>button {
            color: white;
            background-color: #4B8BBE;
        }
        </style>
        """, unsafe_allow_html=True)

# Initialize Streamlit app
st.set_page_config(page_title="Stock Price Prediction App", layout="wide")
local_css()

st.title("📈 Stock Price Prediction App")

# Create a sidebar for user inputs with a new header
st.sidebar.header("🔍 Prediction Options")  # Changed from "⚙️ Prediction Settings"

# User inputs
tickers = st.sidebar.multiselect(
    "Select Stocks to Predict:",
    ['AAPL', 'GOOG', 'META', 'MSFT', 'AMZN'],
    default=['AAPL']
)

# Automatically set start_date to 5 years ago from today
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)  # Approximate 5 years

# Number of days ahead input
num_days_ahead = st.sidebar.number_input(
    "Number of Days Ahead to Predict:",
    min_value=1,
    max_value=30,
    value=7,
    step=1
)

# Run Prediction Button
if st.sidebar.button("🚀 Run Prediction"):
    device = get_device()  # Get the device configuration
    # Removed the line that displays the device to the user

    # Display selected date range for user confirmation (optional)
    st.sidebar.success(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")

    # Download and prepare data
    ticker_data = {}
    for ticker in tickers:
        with st.spinner(f"🔄 Downloading data for {ticker}..."):
            df = download_ticker_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            df = add_technical_indicators(df)
            df.dropna(inplace=True)
            ticker_data[ticker] = df

    # Generate future dates
    future_dates = []
    last_trained_date = end_date
    current_date = last_trained_date + timedelta(days=1)
    while len(future_dates) < num_days_ahead:
        if current_date.weekday() < 5:  # Weekdays only
            future_dates.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    # Load models and make predictions
    loaded_models = {}
    for ticker in tickers:
        with st.spinner(f"📦 Loading model for {ticker}..."):
            try:
                model, feature_scaler, target_scaler = load_model_and_scalers(ticker)
                loaded_models[ticker] = {
                    'model': model,
                    'feature_scaler': feature_scaler,
                    'target_scaler': target_scaler
                }
            except Exception as e:
                st.error(f"❌ Failed to load model for {ticker}: {e}")

    # Make predictions
    with st.spinner("🤖 Making predictions..."):
        all_predictions = predict_future_prices_multiple_tickers(
            tickers,
            future_dates,
            loaded_models,
            ticker_data,
            sequence_length=60,
            device=device  # Pass the device to prediction functions
        )

    # Display predictions and plots
    for ticker, predictions in all_predictions.items():
        st.markdown(f"### 📊 Predictions for {ticker}:")
        if predictions:
            df_predictions = pd.DataFrame.from_dict(predictions, orient='index', columns=['Predicted Price'])
            # Add 'Dates' as a separate column and remove the default index
            df_predictions.index.name = 'Dates'
            df_predictions = df_predictions.reset_index(drop=True)
            df_predictions.insert(0, 'Dates', list(predictions.keys()))
            # Format the predicted prices with '$' and two decimal places
            df_predictions['Predicted Price'] = df_predictions['Predicted Price'].apply(lambda x: f"${x:,.2f}")
            st.table(df_predictions)  # Use table for better formatting

            # Display metrics (commented out)
            # metrics = load_metrics(ticker)
            # st.text(metrics)

            # Display only 'actual_vs_predicted.png' plot
            display_plots(ticker)
        else:
            st.write("No predictions available.")

    st.success("✅ Predictions completed successfully!")
