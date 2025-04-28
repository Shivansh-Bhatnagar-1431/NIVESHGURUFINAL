import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import logging

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Streamlit app configuration
st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("📈 Stock Price Prediction using LSTM")

# Available stocks
stocks = {
    'IOC.NS': 'Indian Oil',
    'IRFC.NS': 'IRFC',
    'BAJAJCON.NS': 'Bajaj Consumer',
    'UNIONBANK.NS': 'Union Bank'
}

# File paths
MODEL_DIR = "models"
SCALER_DIR = "scalers"
CSV_DIR = "csv"

# Configure TensorFlow to use GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        st.info("GPU acceleration enabled")
    except RuntimeError as e:
        st.warning(f"GPU configuration error: {e}")

# Cache data loading
@st.cache_data
def load_stock_data(ticker):
    csv_path = os.path.join(CSV_DIR, f"{ticker.replace('.NS', '')}.csv")
    if not os.path.exists(csv_path):
        st.error(f"CSV file not found at {csv_path}")
        return None
    try:
        # Skip the first 3 rows (header, ticker, and empty row)
        df = pd.read_csv(csv_path, skiprows=3)
        # Rename columns to match our expected format
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            st.error(f"CSV missing required columns: {required_columns}")
            return None
        if len(df) < 61:
            st.error(f"CSV has insufficient data ({len(df)} rows). Need at least 61 rows.")
            return None
        st.info(f"Loaded data for {stocks[ticker]} with {len(df)} rows")
        return df
    except Exception as e:
        st.error(f"Error reading {csv_path}: {e}")
        return None

# Cache preprocessing
@st.cache_data
def preprocess_data(df, time_step=60):
    # Select features
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].values
    
    # Scale all features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    
    def create_dataset(data, time_step):
        X, Y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), :])  # All features
            Y.append(data[i + time_step, 3])      # Target: Close (index 3)
        return np.array(X), np.array(Y)
    
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    return X_train, y_train, X_test, y_test, scaler, train_size, scaled_data

# Cache model training with optimized parameters
@st.cache_resource
def train_model(X_train, y_train, X_test, y_test, time_step, n_features):
    with st.spinner("Training model (this may take a few minutes)..."):
        # Create an even simpler model architecture
        model = Sequential([
            LSTM(units=32, return_sequences=True, input_shape=(time_step, n_features)),
            Dropout(0.1),
            LSTM(units=32),
            Dropout(0.1),
            Dense(units=1)
        ])
        
        # Use a more efficient optimizer with higher learning rate
        model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            metrics=['mae']
        )
        
        # Further reduce epochs and increase batch size
        model.fit(
            X_train, y_train,
            epochs=10,  # Reduced from 20
            batch_size=256,  # Increased from 128
            validation_data=(X_test, y_test),
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=2,
                    restore_best_weights=True
                )
            ]
        )
    return model

# Optimized model loading and training
def load_or_train_model(X_train, y_train, X_test, y_test, time_step, n_features, ticker, scaler):
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.h5")
    scaler_path = os.path.join(SCALER_DIR, f"{ticker}_scaler.pkl")
    
    # First try to load both model and scaler
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            with st.spinner("Loading existing model..."):
                model = tf.keras.models.load_model(model_path)
                st.info("Loaded existing model")
                return model, scaler
        except Exception as e:
            st.warning(f"Error loading model: {e}")
            # If loading fails, delete the corrupted files
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(scaler_path):
                os.remove(scaler_path)
    
    # If loading failed or files don't exist, train new model
    with st.spinner("Training new model..."):
        model = train_model(X_train, y_train, X_test, y_test, time_step, n_features)
        # Save model and scaler
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        st.success("Model trained and saved successfully!")
        return model, scaler

# Main app logic
def main():
    # Stock selection
    selected_stock = st.sidebar.selectbox(
        "Select Stock:",
        options=list(stocks.keys()),
        format_func=lambda x: stocks[x]
    )

    # Load stock data
    df = load_stock_data(selected_stock)
    if df is None:
        return

    # Preprocess data
    try:
        time_step = 60
        n_features = 5  # Open, High, Low, Close, Volume
        X_train, y_train, X_test, y_test, scaler, train_size, scaled_data = preprocess_data(df, time_step)
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return

    # Load or train model
    retrain = st.sidebar.checkbox("Retrain Model")
    if retrain:
        model, scaler = load_or_train_model(X_train, y_train, X_test, y_test, time_step, n_features, selected_stock, scaler)
    else:
        try:
            model_path = os.path.join(MODEL_DIR, f"{selected_stock}_model.h5")
            scaler_path = os.path.join(SCALER_DIR, f"{selected_stock}_scaler.pkl")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                with st.spinner("Loading existing model..."):
                    model = tf.keras.models.load_model(model_path)
                    scaler = joblib.load(scaler_path)
                    st.info("Loaded existing model and scaler")
            else:
                model, scaler = load_or_train_model(X_train, y_train, X_test, y_test, time_step, n_features, selected_stock, scaler)
        except Exception as e:
            st.warning(f"Error loading model: {e}")
            model, scaler = load_or_train_model(X_train, y_train, X_test, y_test, time_step, n_features, selected_stock, scaler)

    # Add a clear section for predictions
    st.markdown("---")
    st.subheader("📊 Stock Price Predictions")
    
    # Predict button in a more prominent location
    if st.button("🔮 Generate Predictions", key="predict_button"):
        st.info("Prediction button clicked. Starting prediction...")
        try:
            with st.spinner("Generating predictions..."):
                # Predict on train and test data
                train_predict = model.predict(X_train, verbose=0)
                test_predict = model.predict(X_test, verbose=0)

                # Reverse scaling for Close price
                # Create dummy arrays to match scaler shape (5 features)
                dummy_train = np.zeros((train_predict.shape[0], 5))
                dummy_test = np.zeros((test_predict.shape[0], 5))
                dummy_y_test = np.zeros((y_test.shape[0], 5))
                
                # Set Close (index 3) to predictions
                dummy_train[:, 3] = train_predict[:, 0]
                dummy_test[:, 3] = test_predict[:, 0]
                dummy_y_test[:, 3] = y_test
                
                # Inverse transform
                train_predict = scaler.inverse_transform(dummy_train)[:, 3]
                test_predict = scaler.inverse_transform(dummy_test)[:, 3]
                y_test_actual = scaler.inverse_transform(dummy_y_test)[:, 3]

                # Ensure lengths match
                min_length = min(len(df.index[train_size+2*time_step+1:]), len(y_test_actual), len(test_predict))
                y_test_actual = y_test_actual[:min_length]
                test_predict = test_predict[:min_length]
                test_dates = df.index[train_size+2*time_step+1:train_size+2*time_step+1+min_length]

                # Calculate errors
                errors = np.abs(y_test_actual - test_predict)
                percentage_errors = (errors / y_test_actual) * 100

                # Future predictions (19 trading days from last date)
                future_days = 19
                future_inputs = X_test[-1].copy()
                future_predictions = []
                for _ in range(future_days):
                    future_pred_scaled = model.predict(future_inputs.reshape(1, time_step, n_features), verbose=0)
                    future_predictions.append(future_pred_scaled[0, 0])
                    # Shift inputs and add new prediction
                    future_inputs = np.roll(future_inputs, -1, axis=0)
                    future_inputs[-1, 3] = future_pred_scaled[0, 0]  # Update Close
                    # Keep other features as last known values (simplified)
                    future_inputs[-1, :3] = future_inputs[-2, :3]   # Open, High, Low
                    future_inputs[-1, 4] = future_inputs[-2, 4]     # Volume

                # Reverse scaling for future predictions
                dummy_future = np.zeros((len(future_predictions), 5))
                dummy_future[:, 3] = future_predictions
                future_predictions = scaler.inverse_transform(dummy_future)[:, 3]

                # Generate future trading dates
                last_date = df.index[-1]
                future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=future_days)

                # Plot results
                st.subheader(f"📈 {stocks[selected_stock]} Stock Price Prediction Plot")
                fig, ax = plt.subplots(figsize=(14, 7))
                ax.plot(df.index[time_step:train_size+time_step], 
                        df['Close'][time_step:train_size+time_step], 
                        label="Actual Train Data", color="blue")
                ax.plot(test_dates, y_test_actual, label="Actual Test Data", color="red")
                ax.plot(test_dates, test_predict, label="Predicted Test Data", linestyle='dashed', color="green")
                ax.plot(future_dates, future_predictions, 
                        label=f"Future Predictions ({future_dates[0].strftime('%d %b')} - {future_dates[-1].strftime('%d %b %Y')})", 
                        linestyle='dotted', color='purple')
                ax.legend()
                ax.set_title(f"{stocks[selected_stock]} Stock Price Prediction")
                ax.set_xlabel("Date")
                ax.set_ylabel("Stock Price")
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True)
                st.pyplot(fig)

                # Display actual vs. predicted test prices with errors
                st.subheader("📊 Actual vs. Predicted Test Prices with Errors")
                results_df = pd.DataFrame({
                    'Date': test_dates,
                    'Actual Price': np.round(y_test_actual, 2),
                    'Predicted Price': np.round(test_predict, 2),
                    'Absolute Error': np.round(errors, 2),
                    'Percentage Error (%)': np.round(percentage_errors, 2)
                })
                st.dataframe(results_df, height=300)

                # Display future predictions
                st.subheader(f"🔮 Future Predictions ({future_dates[0].strftime('%d %b')} - {future_dates[-1].strftime('%d %b %Y')})")
                future_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': np.round(future_predictions, 2)
                })
                st.dataframe(future_df, height=300)

                # Display tomorrow's prediction
                today_date = date.today()
                tomorrow_date = today_date + timedelta(days=1)
                tomorrow_pred = future_predictions[0]
                st.write(f"📅 **Today's Date**: {today_date}")
                st.write(f"🔮 **Predicted Stock Price for Tomorrow ({tomorrow_date})**: {tomorrow_pred:.2f}")
            st.success("Prediction completed!")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()