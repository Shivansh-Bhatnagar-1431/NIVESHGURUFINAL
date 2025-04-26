import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from itertools import product
import datetime

# Set up page
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("📈 Stock Price Prediction using LSTM")

# Get today's date
today = datetime.date.today().strftime('%Y-%m-%d')

# Select stocks
tickers = {
    'IOC.NS': 'Indian Oil',
    'IRFC.NS': 'IRFC',
    'BAJAJCON.NS': 'Bajaj Consumer',
    'UNIONBANK.NS': 'Union Bank',
}

# Dataset preparation
def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Hyperparameter tuning + training
@st.cache_data(show_spinner=True)
def train_and_tune_model(scaled_data, time_step=60):
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], time_step, 1)
    X_test = X_test.reshape(X_test.shape[0], time_step, 1)

    param_grid = {
        'lstm_units': [50, 100],
        'dropout_rate': [0.2, 0.3],
        'epochs': [30, 50],
        'batch_size': [32, 64]
    }

    best_model, best_mse, best_params = None, float('inf'), None
    best_test_predict, best_y_test = None, None

    for lstm_units, dropout_rate, epochs, batch_size in product(*param_grid.values()):
        model = Sequential([
            LSTM(lstm_units, return_sequences=True, input_shape=(time_step, 1)),
            Dropout(dropout_rate),
            LSTM(lstm_units, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(lstm_units),
            Dropout(dropout_rate),
            Dense(1)
        ])
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)

        test_predict = model.predict(X_test)
        mse = mean_squared_error(y_test, test_predict)

        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_params = (lstm_units, dropout_rate, epochs, batch_size)
            best_test_predict = test_predict
            best_y_test = y_test

    return best_model, best_params, X_test, best_y_test, best_test_predict, train_size

# Main app
selected_stock = st.selectbox("Select a stock", list(tickers.keys()), format_func=lambda x: tickers[x])

if selected_stock:
    st.subheader(f"Training model for: {tickers[selected_stock]} ({selected_stock})")

    df = yf.download(selected_stock, start='2015-01-01', end=today)
    
    if df.empty or len(df) < 100:
        st.error("Not enough data available to train the model.")
    else:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

        with st.spinner("Training and tuning the model... This may take a minute ⏳"):
            model, params, X_test, y_test, test_predict, train_size = train_and_tune_model(scaled_data)

        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        test_predict = scaler.inverse_transform(test_predict).flatten()
        test_dates = df.index[train_size + 60 + 1 : train_size + 60 + 1 + len(test_predict)]

        # Future prediction
        last_60_days = scaled_data[-60:]
        future_input = list(last_60_days.flatten())
        future_preds_scaled = []

        for _ in range(15):
            x_input = np.array(future_input[-60:]).reshape(1, 60, 1)
            pred = model.predict(x_input, verbose=0)
            future_preds_scaled.append(pred[0][0])
            future_input.append(pred[0][0])

        future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()
        future_dates = pd.date_range(start=test_dates[-1] + pd.Timedelta(days=1), periods=15)

        # Display actual vs predicted last 10
        st.subheader("Actual vs Predicted (last 10 test points)")
        comparison_df = pd.DataFrame({
            'Actual': y_test_actual[-10:],
            'Predicted': test_predict[-10:],
            'Difference': np.abs(y_test_actual[-10:] - test_predict[-10:])
        })
        comparison_df['Status'] = comparison_df['Difference'].apply(lambda x: 'Close' if x <= 10 else 'Far')
        st.dataframe(comparison_df)

        # Plot
        st.subheader("📈 Price Prediction Graph")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(test_dates, y_test_actual, label='Actual Price')
        ax.plot(test_dates, test_predict, label='Predicted Price')
        ax.plot(future_dates, future_preds, label='Future Prediction (15 days)', linestyle='dashed')
        ax.set_xlabel('Date')
        ax.set_ylabel('Stock Price')
        ax.set_title(f"{tickers[selected_stock]} Price Prediction")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
