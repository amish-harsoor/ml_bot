# main.py - NIFTY 50 MULTI-STOCK PREDICTOR (JSON FIX)
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import yfinance as yf
import joblib
from pathlib import Path
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import uvicorn
from datetime import datetime
import numpy as np

app = FastAPI()
MODEL_PATH = Path("nifty50_model.joblib")

NIFTY50_TICKERS = [
    "RELIANCE.NS","HDFCBANK.NS","INFY.NS","HDFCLIFE.NS","ICICIBANK.NS","TCS.NS",
    "KOTAKBANK.NS","HINDUNILVR.NS","SBIN.NS","BHARTIARTL.NS","ITC.NS","ASIANPAINT.NS",
    "AXISBANK.NS","LT.NS","MARUTI.NS","SUNPHARMA.NS","TITAN.NS","BAJFINANCE.NS",
    "WIPRO.NS","ULTRACEMCO.NS","NESTLEIND.NS","TATAMOTORS.NS","JSWSTEEL.NS",
    "POWERGRID.NS","NTPC.NS","TECHM.NS","ONGC.NS","COALINDIA.NS","TATASTEEL.NS",
    "HCLTECH.NS","M&M.NS","ADANIPORTS.NS","GRASIM.NS","CIPLA.NS","INDUSINDBK.NS",
    "BRITANNIA.NS","HEROMOTOCO.NS","DRREDDY.NS","EICHERMOT.NS","BPCL.NS",
    "SHRIRAMFIN.NS","DIVISLAB.NS","BAJAJFINSV.NS","UPL.NS","IOC.NS",
    "HINDALCO.NS","VEDL.NS","GAIL.NS","BAJAJ-AUTO.NS","ADANIENT.NS"
]

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

def train_nifty50():
    print("Training Nifty 50 multi-stock model...")
    dfs = []
    for t in NIFTY50_TICKERS:
        try:
            print(f"Downloading {t} ... ", end="")
            data = yf.download(t, period="8y", auto_adjust=True, progress=False, multi_level_index=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            if len(data) < 300:
                print("skip (too short)")
                continue
            
            df = add_features(data)
            
            if len(df) > 100:
                dfs.append(df)
                print(f"OK ({len(df)} rows)")
            else:
                print("skip (features failed)")
        except Exception as e:
            print(f"failed: {e}")

    if not dfs:
        raise RuntimeError("No data downloaded! Check internet or yfinance version.")

    combined = pd.concat(dfs, ignore_index=True)

    X = combined.select_dtypes(include=["float64", "int64"]).drop(columns=["Target"], errors="ignore")
    y = combined["Target"]

    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]

    if len(X) == 0:
        raise ValueError("All data was filtered out! Check feature generation.")

    split = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0
    )

    model.fit(
        X_train_s, y_train,
        eval_set=[(X_test_s, y_test)],
        verbose=False
    )

    package = {
        "model": model,
        "scaler": scaler,
        "features": X.columns.tolist(),
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    joblib.dump(package, MODEL_PATH)
    print(f"MODEL READY! {len(combined):,} samples → {MODEL_PATH}")
    return package

@app.on_event("startup")
async def startup():
    if not MODEL_PATH.exists():
        train_nifty50()
    else:
        print("Model already exists – loaded")

def get_model():
    if not MODEL_PATH.exists():
        return train_nifty50()
    return joblib.load(MODEL_PATH)

@app.get("/")
async def home():
    return FileResponse("index.html")

@app.get("/predict/{ticker}")
async def predict(ticker: str):
    pkg = get_model()
    symbol = ticker.upper() + ".NS" if not ticker.upper().endswith(".NS") else ticker.upper()

    data = yf.download(symbol, period="100d", auto_adjust=True, progress=False, multi_level_index=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if data.empty:
        raise HTTPException(404, "Ticker not found or no data")

    df = add_features(data)
    if df.empty:
        raise HTTPException(400, "Not enough data to generate features")

    try:
        X_latest = df[pkg["features"]].iloc[-1:].copy()
    except KeyError as e:
        raise HTTPException(500, f"Feature mismatch: {e}")

    X_scaled = pkg["scaler"].transform(X_latest)
    
    # FIX IS HERE: Explicitly convert numpy types to standard python float
    prob = float(pkg["model"].predict_proba(X_scaled)[0][1])
    price = float(data["Close"].iloc[-1])

    return {
        "ticker": ticker.upper().replace(".NS", ""),
        "price": round(price, 2),
        "prediction": "UP" if prob > 0.5 else "DOWN",
        "confidence": round(prob, 4),
        "model": "Nifty 50 Multi-Stock AI"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)