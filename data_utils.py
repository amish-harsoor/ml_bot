import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame):
    df = df.copy()
    
    # Ensure we are working with a flat index before computing
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["Return"] = df["Close"].pct_change()

    # Lags
    for i in range(1, 31):
        df[f"lag_{i}"] = df["Return"].shift(i)

    # Indicators
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_30"] = df["Close"].rolling(30).mean()
    df["volatility_10"] = df["Return"].rolling(10).std()
    
    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    return df.dropna()