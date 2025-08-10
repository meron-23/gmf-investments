# scripts/clean_data.py
import pandas as pd
from pathlib import Path

cleaned_data = {}
for ticker in ["TSLA", "BND", "SPY"]:
    df = pd.read_csv(f"data/raw/{ticker}.csv", parse_dates=["Date"], index_col="Date")
    
    # Check missing values
    print(f"{ticker} missing values:\n{df.isna().sum()}")
    
    # Forward-fill missing values
    df = df.ffill()
    
    # Ensure numeric columns are floats
    df = df.astype(float)
    
    cleaned_data[ticker] = df
    df.to_csv(f"data/{ticker}_clean.csv")

print("✅ Cleaning complete.")
