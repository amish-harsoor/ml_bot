from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import yfinance as yf
import uvicorn
from config import MODEL_PATH
from data_utils import add_features
from model_training import train_nifty50, get_model
from backtesting import backtest_ticker

app = FastAPI()

@app.on_event("startup")
async def startup():
    if not MODEL_PATH.exists():
        train_nifty50()
    else:
        print("Model already exists – loaded")

@app.get("/")
async def home():
    return FileResponse("index.html")

@app.get("/predict/{ticker}")
async def predict(ticker: str):
    pkg = get_model()
    symbol = ticker.upper() + ".NS" if not ticker.upper().endswith(".NS") else ticker.upper()

    data = yf.download(symbol, period="1y", auto_adjust=True, progress=False, multi_level_index=False)
    
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

@app.get("/backtest/{ticker}/{start_date}/{end_date}")
async def backtest(ticker: str, start_date: str, end_date: str):
    symbol = ticker.upper() + ".NS" if not ticker.upper().endswith(".NS") else ticker.upper()
    result = backtest_ticker(symbol, start_date, end_date)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result

@app.get("/get_prices/{ticker}/{start_date}/{end_date}")
async def get_prices(ticker: str, start_date: str, end_date: str):
    symbol = ticker.upper() + ".NS" if not ticker.upper().endswith(".NS") else ticker.upper()
    data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True, progress=False, multi_level_index=False)

    if data.empty:
        raise HTTPException(404, "No price data available")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    prices = data['Close'].ffill().tolist()
    dates = data.index.strftime('%Y-%m-%d').tolist()

    return {
        "ticker": symbol,
        "dates": dates,
        "prices": prices,
        "start_price": float(prices[0]) if prices else 0,
        "end_price": float(prices[-1]) if prices else 0
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)