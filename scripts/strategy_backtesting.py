import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define backtesting window
backtest_start = pd.to_datetime("2024-08-01")
backtest_end = pd.to_datetime("2025-07-31")

# Slice prices to backtesting period
prices_bt = prices.loc[backtest_start:backtest_end]

# Calculate daily returns during backtest
returns_bt = prices_bt.pct_change().dropna()

# Strategy: Use optimal weights from Task 4 max Sharpe portfolio, hold entire backtest period
weights = np.array([cleaned_weights['TSLA'], cleaned_weights['BND'], cleaned_weights['SPY']])

# Portfolio daily returns
strategy_returns = returns_bt.dot(weights)

# Benchmark: static 60% SPY / 40% BND portfolio
benchmark_weights = np.array([0.0, 0.4, 0.6])  # No TSLA in benchmark
benchmark_returns = returns_bt.dot(benchmark_weights)

# Compute cumulative returns
strategy_cum_returns = (1 + strategy_returns).cumprod() - 1
benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1

# Calculate Sharpe Ratios (assume 0% risk-free, daily returns scaled to annual)
def annualized_sharpe(returns, trading_days=252):
    return (returns.mean() / returns.std()) * np.sqrt(trading_days)

strategy_sharpe = annualized_sharpe(strategy_returns)
benchmark_sharpe = annualized_sharpe(benchmark_returns)

# Total return over backtest
strategy_total_return = strategy_cum_returns[-1]
benchmark_total_return = benchmark_cum_returns[-1]

# Plot cumulative returns
plt.figure(figsize=(12,7))
plt.plot(strategy_cum_returns.index, strategy_cum_returns, label="Strategy Portfolio")
plt.plot(benchmark_cum_returns.index, benchmark_cum_returns, label="Benchmark 60% SPY / 40% BND")
plt.title("Backtest: Cumulative Returns (Aug 2024 - Jul 2025)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.show()

# Print summary
print("Backtest Results:")
print(f"Strategy Total Return: {strategy_total_return:.2%}")
print(f"Strategy Annualized Sharpe Ratio: {strategy_sharpe:.2f}")
print(f"Benchmark Total Return: {benchmark_total_return:.2%}")
print(f"Benchmark Annualized Sharpe Ratio: {benchmark_sharpe:.2f}")

if strategy_total_return > benchmark_total_return:
    print("\nYour model-driven strategy outperformed the benchmark in total return.")
else:
    print("\nBenchmark outperformed your strategy in total return.")

if strategy_sharpe > benchmark_sharpe:
    print("Your strategy has better risk-adjusted returns (Sharpe Ratio).")
else:
    print("Benchmark has better risk-adjusted returns (Sharpe Ratio).")
