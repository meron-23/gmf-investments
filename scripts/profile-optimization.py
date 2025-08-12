import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting

# Load your cleaned historical price data for all assets
tsla = pd.read_csv("data/TSLA_clean.csv", parse_dates=["Date"], index_col="Date")['Adj Close']
bnd = pd.read_csv("data/BND_clean.csv", parse_dates=["Date"], index_col="Date")['Adj Close']
spy = pd.read_csv("data/SPY_clean.csv", parse_dates=["Date"], index_col="Date")['Adj Close']

# Align dates across all three assets
prices = pd.concat([tsla, bnd, spy], axis=1).dropna()
prices.columns = ['TSLA', 'BND', 'SPY']

# Calculate daily returns
returns = prices.pct_change().dropna()

# --- Expected Returns Vector ---

# For TSLA, use your forecasted expected return (annualized)
# Example: Suppose your best model forecast gives you 12% annual return for TSLA
forecasted_tsla_return = 0.12  

# For BND and SPY, calculate historical annualized mean returns
trading_days = 252
mean_returns = returns.mean() * trading_days
expected_returns_vector = mean_returns.copy()
expected_returns_vector['TSLA'] = forecasted_tsla_return

print("Expected Annual Returns:\n", expected_returns_vector)

# --- Covariance Matrix (annualized) ---
cov_matrix = returns.cov() * trading_days

# --- Optimize Portfolio ---
ef = EfficientFrontier(expected_returns_vector.values, cov_matrix.values)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

print("\nOptimized Portfolio Weights (Max Sharpe):")
print(cleaned_weights)

# Get portfolio performance
expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(verbose=True)

# --- Generate Efficient Frontier Data ---
fig, ax = plt.subplots(figsize=(10,7))
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)

# Find Minimum Volatility portfolio
ef_min_vol = EfficientFrontier(expected_returns_vector.values, cov_matrix.values)
min_vol_weights = ef_min_vol.min_volatility()
min_vol_performance = ef_min_vol.portfolio_performance(verbose=False)
min_vol_weights_clean = ef_min_vol.clean_weights()

# Plot min volatility portfolio
ax.scatter(min_vol_performance[1], min_vol_performance[0], marker="*", s=200, c="r", label="Minimum Volatility")

# Plot max sharpe portfolio
ax.scatter(annual_volatility, expected_annual_return, marker="*", s=200, c="g", label="Maximum Sharpe Ratio")

ax.legend()
plt.title("Efficient Frontier with Key Portfolios")
plt.xlabel("Annualized Volatility (Risk)")
plt.ylabel("Annualized Return")
plt.show()

# --- Summary ---
print("\nMinimum Volatility Portfolio Weights:")
print(min_vol_weights_clean)
print(f"Expected Return: {min_vol_performance[0]:.2%}")
print(f"Volatility: {min_vol_performance[1]:.2%}")
print(f"Sharpe Ratio: {min_vol_performance[2]:.2f}")

print("\nMaximum Sharpe Portfolio (Recommended):")
print(cleaned_weights)
print(f"Expected Return: {expected_annual_return:.2%}")
print(f"Volatility: {annual_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
