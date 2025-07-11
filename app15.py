import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime
import traceback
import os

# Set page configuration
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Load stocks from JSON file
def load_stocks_from_json(json_path="stocks.json"):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            return data.get("stocks", [])
    except Exception as e:
        st.error(f"Failed to load stocks from {json_path}: {e}")
        return []

@st.cache_data
def calculate_indicators(data, ma_window=20, ma_type='SMA', wma_exp_factor=1.0):
    try:
        data = data.copy()
        if ma_type == 'SMA':
            data['MA'] = data['Close'].rolling(window=ma_window).mean()
        elif ma_type == 'EMA':
            data['MA'] = data['Close'].ewm(span=ma_window, adjust=False).mean()
        else:
            # WMA: recent values have higher weights
            weights = np.exp(np.linspace(0, wma_exp_factor, ma_window))[::-1]
            weights /= weights.sum()
            data['MA'] = data['Close'].rolling(window=ma_window).apply(lambda x: np.sum(x * weights), raw=True)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        ema12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = ema12 - ema26
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['BB_Mid'] = data['Close'].rolling(window=ma_window).mean()
        data['BB_Std'] = data['Close'].rolling(window=ma_window).std()
        data['BB_Upper'] = data['BB_Mid'] + 2 * data['BB_Std']
        data['BB_Lower'] = data['BB_Mid'] - 2 * data['BB_Std']
        return data[['Close', 'MA', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower']].dropna()
    except Exception as e:
        st.error(f"Indicator Error: {e}")
        return None

@st.cache_data
def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        return data if not data.empty else None
    except Exception as e:
        st.error(f"Fetch Error: {e}")
        return None

@st.cache_resource
def train_lstm_model(X_train, y_train, X_test, y_test, sequence_length, n_features, epochs):
    try:
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, n_features)))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(7))  # Output: 7 days
        model.compile(optimizer='adam', loss='mean_squared_error')
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), verbose=0)
        return model, history
    except Exception as e:
        st.error(f"Training Error: {e}")
        return None, None

def calculate_ma_prediction(data, window=20, days=7, ma_type='SMA', wma_exp_factor=1.0):
    try:
        if ma_type == 'SMA':
            ma = data['Close'].rolling(window=window).mean()
        elif ma_type == 'EMA':
            ma = data['Close'].ewm(span=window, adjust=False).mean()
        else:
            weights = np.exp(np.linspace(0, wma_exp_factor, window))[::-1]
            weights /= weights.sum()
            ma = data['Close'].rolling(window=window).apply(lambda x: np.sum(x * weights), raw=True)
        ma = ma.dropna()
        n_points = min(30, len(ma))
        if n_points < 2:
            return np.array([np.nan] * days)
        x = np.arange(n_points).reshape(-1, 1)
        y = ma[-n_points:].values
        model = LinearRegression().fit(x, y)
        return model.predict(np.arange(n_points, n_points + days).reshape(-1, 1)).flatten()
    except Exception as e:
        st.error(f"MA Prediction Error: {e}")
        return np.array([np.nan] * days)

# UI
st.title("📈 Stock Price Prediction (LSTM + Indicators)")

# Load tickers from JSON
stock_options = load_stocks_from_json()

col1, col2 = st.columns(2)
with col1:
    if stock_options:
        ticker = st.selectbox("Select Stock Ticker", stock_options + ["Other"])
        if ticker == "Other":
            ticker = st.text_input("Enter Custom Ticker", value="AAPL").upper()
    else:
        ticker = st.text_input("Enter Stock Ticker", value="AAPL").upper()

with col2:
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("today"))
    sequence_length = st.slider("Sequence Length", 10, 100, 60)
    epochs = st.slider("Training Epochs", 3, 20, 5)
    ma_window = st.slider("Moving Average Window", 10, 50, 20)
    ma_type = st.selectbox("Moving Average Type", ["SMA", "EMA", "WMA"])
    wma_exp_factor = st.slider("WMA Exponent (if WMA)", 0.5, 5.0, 1.0, 0.1)

# Fetch & Predict
if st.button("📊 Fetch Data and Predict"):
    st.info("Fetching stock data...")
    df = fetch_stock_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    if df is None or len(df) < sequence_length + ma_window:
        st.error("Not enough data for prediction.")
    else:
        indicators = calculate_indicators(df, ma_window, ma_type, wma_exp_factor)
        if indicators is None:
            st.error("Failed to calculate indicators.")
        else:
            ma_preds = calculate_ma_prediction(df, ma_window, 7, ma_type, wma_exp_factor)
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(indicators.values)

            def create_sequences(data, seq_len):
                X, y = [], []
                for i in range(len(data) - seq_len - 7):
                    X.append(data[i:i+seq_len])
                    y.append(data[i+seq_len:i+seq_len+7, 0])  # 0: 'Close'
                return np.array(X), np.array(y)

            X, y = create_sequences(scaled, sequence_length)
            if len(X) < 10:
                st.error("Too few samples to train.")
            else:
                train_len = int(len(X) * 0.8)
                X_train, X_test = X[:train_len], X[train_len:]
                y_train, y_test = y[:train_len], y[train_len:]
                st.info("Training model...")
                model, hist = train_lstm_model(X_train, y_train, X_test, y_test, sequence_length, X.shape[2], epochs)

                if model:
                    last_seq = scaled[-sequence_length:].reshape(1, sequence_length, -1)
                    pred_scaled = model.predict(last_seq, verbose=0)[0]
                    # Inverse transform: only 'Close' is predicted, other features set to 0
                    pred_prices = [scaler.inverse_transform([[p]+[0]*6])[0][0] for p in pred_scaled]
                    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=7, freq='B')

                    df_pred = pd.DataFrame({
                        'Date': future_dates,
                        'LSTM Prediction': pred_prices,
                        f'{ma_type} Prediction': ma_preds
                    })

                    st.subheader("🔮 7-Day Forecast")
                    st.dataframe(df_pred)

                    st.subheader("📈 Prediction Plot")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(df.index[-30:], df['Close'][-30:], label="Close", color="blue")
                    ax.plot(future_dates, pred_prices, label="LSTM", color="green", marker='o')
                    ax.plot(future_dates, ma_preds, label=ma_type, color="orange", marker='x')
                    ax.legend()
                    ax.set_title(f"{ticker} Price Forecast")
                    st.pyplot(fig)