import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pickle
import datetime

# Get today's date
today = datetime.date.today().strftime('%Y-%m-%d')

# Select 5 stocks (within Rs. 100–200 range approximately)
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

# Load model and scaler
def load_model_and_scaler(ticker):
    model = load_model(f'models/{ticker}_model.h5')
    with open(f'scalers/{ticker}_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Ask user which stock they want to visualize
st.title("Stock Price Prediction")
st.write("Select a stock to visualize its predictions:")

selected_ticker = st.selectbox('Choose a stock:', list(tickers.keys()))

# Get the stock data and prepare the dataset
df = yf.download(selected_ticker, start='2015-01-01', end=today)

if df.empty or len(df) < 100:
    st.error(f"Not enough data for {selected_ticker}. Try a different stock.")
else:
    # Scale the data and prepare for prediction
    scaler = load_model_and_scaler(selected_ticker)[1]
    scaled_data = scaler.transform(df['Close'].values.reshape(-1, 1))

    # Create dataset
    X, Y = create_dataset(scaled_data)

    # Load model
    model = load_model_and_scaler(selected_ticker)[0]

    # Make predictions on historical data
    X = X.reshape(X.shape[0], X.shape[1], 1)
    test_predict = model.predict(X)

    # Inverse transform the predictions
    test_predict = scaler.inverse_transform(test_predict)
    y_actual = scaler.inverse_transform(Y.reshape(-1, 1))

    # Get the dates for the test data
    test_dates = df.index[60:len(test_predict) + 60]

    # Future prediction (next 15 days)
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

    # Display actual vs predicted
    st.write("Actual vs Predicted Prices for last 10 days:")
    for actual, predicted in zip(y_actual[-10:], test_predict[-10:]):
        diff = abs(actual - predicted)
        status = "Close" if diff <= 10 else "Far"
        st.write(f"Actual: {actual[0]:.2f}, Predicted: {predicted[0]:.2f} --> {status}")

    # Plotting
    plt.figure(figsize=(14, 6))
    plt.plot(test_dates, y_actual.flatten(), label='Actual Price')
    plt.plot(test_dates, test_predict.flatten(), label='Predicted Price')
    plt.plot(future_dates, future_preds, label='Future Prediction (15 days)', linestyle='dashed')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f"{tickers[selected_ticker]} ({selected_ticker}) Stock Price Prediction")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
