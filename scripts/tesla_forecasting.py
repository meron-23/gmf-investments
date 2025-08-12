import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import pmdarima as pm
from statsmodels.tsa.stattools import adfuller

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- Step 1: Load and Prepare Data ---
df = pd.read_csv('data/TSLA_clean.csv', parse_dates=['Date'], index_col='Date')
tsla_prices = df['Adj Close']

# Chronological train-test split
train = tsla_prices[:'2023-12-31']
test = tsla_prices['2024-01-01':]

print(f"Train length: {len(train)}, Test length: {len(test)}")

# --- Step 2: Check Stationarity (ADF Test) ---
result = adfuller(train)
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
if result[1] > 0.05:
    print("Series is non-stationary, differencing needed for ARIMA.")
else:
    print("Series is stationary, no differencing needed for ARIMA.")

# --- Step 3: ARIMA Modeling ---
print("\nFitting ARIMA model...")
arima_model = pm.auto_arima(train, seasonal=False, stepwise=True,
                            suppress_warnings=True, error_action='ignore')
print(f"Best ARIMA order: {arima_model.order}")

# Forecast length = length of test set
n_periods = len(test)
arima_forecast, conf_int = arima_model.predict(n_periods=n_periods, return_conf_int=True)

# Create DataFrame for forecast
arima_forecast_index = test.index
arima_pred_df = pd.DataFrame({'forecast': arima_forecast}, index=arima_forecast_index)

# Evaluate ARIMA
mae_arima = mean_absolute_error(test, arima_forecast)
rmse_arima = np.sqrt(mean_squared_error(test, arima_forecast))
mape_arima = np.mean(np.abs((test - arima_forecast) / test)) * 100
print(f"ARIMA MAE: {mae_arima:.4f}, RMSE: {rmse_arima:.4f}, MAPE: {mape_arima:.2f}%")

# --- Step 4: LSTM Modeling ---

# Scale data to [0,1]
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train = scaler.fit_transform(train.values.reshape(-1,1))
scaled_test = scaler.transform(test.values.reshape(-1,1))

# Create supervised learning data for LSTM
def create_dataset(series, window_size=60):
    X, y = [], []
    for i in range(window_size, len(series)):
        X.append(series[i-window_size:i, 0])
        y.append(series[i, 0])
    return np.array(X), np.array(y)

window_size = 60
X_train, y_train = create_dataset(scaled_train, window_size)
X_test, y_test = create_dataset(np.concatenate([scaled_train[-window_size:], scaled_test]), window_size)

# Reshape for LSTM [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train LSTM
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=2)

# Predict on test
lstm_pred_scaled = model.predict(X_test)
lstm_pred = scaler.inverse_transform(lstm_pred_scaled)

# Align LSTM predictions with test index
lstm_pred_index = test.index[window_size:]
lstm_pred_series = pd.Series(lstm_pred.flatten(), index=lstm_pred_index)

# Evaluate LSTM
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))
mae_lstm = mean_absolute_error(y_test_actual, lstm_pred)
rmse_lstm = np.sqrt(mean_squared_error(y_test_actual, lstm_pred))
mape_lstm = np.mean(np.abs((y_test_actual.flatten() - lstm_pred.flatten()) / y_test_actual.flatten())) * 100
print(f"LSTM MAE: {mae_lstm:.4f}, RMSE: {rmse_lstm:.4f}, MAPE: {mape_lstm:.2f}%")

# --- Step 5: Plot Results ---

plt.figure(figsize=(14,6))
plt.plot(test.index, test, label='Actual Prices')
plt.plot(arima_pred_df.index, arima_pred_df['forecast'], label='ARIMA Forecast')
plt.plot(lstm_pred_series.index, lstm_pred_series, label='LSTM Forecast')
plt.title('Tesla Stock Price Forecast Comparison')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# --- Step 6: Summary of performance ---
print("\nModel Performance Summary:")
print(f"ARIMA - MAE: {mae_arima:.4f}, RMSE: {rmse_arima:.4f}, MAPE: {mape_arima:.2f}%")
print(f"LSTM  - MAE: {mae_lstm:.4f}, RMSE: {rmse_lstm:.4f}, MAPE: {mape_lstm:.2f}%")
