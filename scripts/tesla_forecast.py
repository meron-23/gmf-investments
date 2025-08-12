import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay

# Assumes you have these from Task 2
# arima_model: trained pmdarima ARIMA model
# model: trained Keras LSTM model
# scaler: MinMaxScaler fitted on Tesla prices
# tsla_prices: pd.Series with historical Tesla Adj Close prices indexed by date
# window_size: int, e.g. 60

# --------------------
# Load Tesla price data
tsla_prices = pd.read_csv('data/TSLA_clean.csv', parse_dates=['Date'], index_col='Date')['Adj Close']

# --------------------
# ARIMA Forecast 12 months (~252 trading days)
forecast_steps = 252
last_date = tsla_prices.index[-1]
future_dates = pd.bdate_range(start=last_date + BDay(1), periods=forecast_steps)

# Forecast with ARIMA
arima_forecast, arima_conf_int = arima_model.predict(n_periods=forecast_steps, return_conf_int=True)

arima_forecast_df = pd.DataFrame({
    'Forecast': arima_forecast,
    'Lower CI': arima_conf_int[:, 0],
    'Upper CI': arima_conf_int[:, 1]
}, index=future_dates)

# --------------------
# LSTM Recursive Forecast 12 months
last_window = tsla_prices[-window_size:].values.reshape(-1, 1)
last_window_scaled = scaler.transform(last_window)

lstm_forecast_scaled = []
current_input = last_window_scaled.copy()

for _ in range(forecast_steps):
    X_input = current_input.reshape((1, window_size, 1))
    pred_scaled = model.predict(X_input, verbose=0)[0, 0]
    lstm_forecast_scaled.append(pred_scaled)
    # Slide the window forward by one day
    current_input = np.append(current_input[1:], [[pred_scaled]], axis=0)

lstm_forecast_scaled = np.array(lstm_forecast_scaled).reshape(-1, 1)
lstm_forecast = scaler.inverse_transform(lstm_forecast_scaled).flatten()

lstm_forecast_series = pd.Series(lstm_forecast, index=future_dates)

# --------------------
# Plot historical + forecast ARIMA
plt.figure(figsize=(15,7))
plt.plot(tsla_prices.index, tsla_prices, label='Historical Price')
plt.plot(arima_forecast_df.index, arima_forecast_df['Forecast'], label='ARIMA Forecast', color='orange')
plt.fill_between(arima_forecast_df.index, arima_forecast_df['Lower CI'], arima_forecast_df['Upper CI'], 
                 color='orange', alpha=0.3, label='ARIMA Confidence Interval')

# Plot LSTM forecast
plt.plot(lstm_forecast_series.index, lstm_forecast_series, label='LSTM Forecast', color='green')

plt.title('Tesla Stock Price Forecast - Next 12 Months')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
