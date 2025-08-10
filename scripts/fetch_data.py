# scripts/fetch_data.py
import yfinance as yf
import pandas as pd
from pathlib import Path

# Create data folder if not exists
Path("data/raw").mkdir(parents=True, exist_ok=True)

TICKERS = ["TSLA", "BND", "SPY"]
START_DATE = "2015-07-01"
END_DATE = "2025-07-31"

data = {}
for ticker in TICKERS:
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE)
    df.index = pd.to_datetime(df.index)
    df.to_csv(f"../data/raw/{ticker}.csv")
    data[ticker] = df

print("✅ Data download complete.")
