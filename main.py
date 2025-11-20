import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import yfinance as yf
import uvicorn
import matplotlib.pyplot as plt
import io
import base64
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

    data = yf.download(symbol, period="3mo", auto_adjust=True, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    elif isinstance(data.columns, pd.Index):
        # Ensure columns are strings
        data.columns = data.columns.astype(str)

    if data.empty:
        raise HTTPException(404, "Ticker not found or no data")

    df = add_features(data)
    if df.empty:
        raise HTTPException(400, "Not enough data to generate features")

    try:
        X_latest = df[pkg["features"]].iloc[-1:].copy()
    except KeyError as e:
        raise HTTPException(500, f"Feature mismatch: {e}")

    # Data is already normalized from add_features pipeline
    X_scaled = X_latest
    
    # Get prediction probability
    prob = float(pkg["model"].predict_proba(X_scaled)[0][1])
    price = float(data["Close"].iloc[-1])
    
    # Calculate additional metrics for better UX
    # Recent 5-day performance
    recent_5d = data['Close'].tail(6)
    if len(recent_5d) >= 2:
        recent_return = ((recent_5d.iloc[-1] - recent_5d.iloc[-6]) / recent_5d.iloc[-6]) * 100
    else:
        recent_return = 0
    
    # Calculate risk level
    if prob >= 0.7:
        risk_level = "Low"
        prediction_strength = "Strong"
    elif prob >= 0.6 or prob <= 0.4:
        risk_level = "Medium"
        prediction_strength = "Moderate"
    else:
        risk_level = "High"
        prediction_strength = "Weak"
    
    # Market sentiment
    if prob >= 0.65:
        sentiment = "Bullish"
    elif prob <= 0.35:
        sentiment = "Bearish"
    else:
        sentiment = "Neutral"
    
    # Confidence level description
    confidence_pct = abs(prob - 0.5) * 200  # Convert to 0-100% confidence
    if confidence_pct >= 70:
        confidence_desc = "Very High"
    elif confidence_pct >= 50:
        confidence_desc = "High"
    elif confidence_pct >= 30:
        confidence_desc = "Medium"
    else:
        confidence_desc = "Low"

    return {
        "ticker": ticker.upper().replace(".NS", ""),
        "price": round(price, 2),
        "prediction": "UP" if prob > 0.5 else "DOWN",
        "confidence": round(prob, 4),
        "model": "Nifty 50 Multi-Stock AI",
        "additional_metrics": {
            "recent_5d_return": round(recent_return, 2),
            "risk_level": risk_level,
            "prediction_strength": prediction_strength,
            "market_sentiment": sentiment,
            "confidence_level": confidence_desc,
            "confidence_percentage": round(confidence_pct, 1)
        }
    }

@app.get("/backtest/{ticker}/{start_date}/{end_date}")
async def backtest(ticker: str, start_date: str, end_date: str):
    symbol = ticker.upper() + ".NS" if not ticker.upper().endswith(".NS") else ticker.upper()
    result = backtest_ticker(symbol, start_date, end_date)
    if "error" in result:
        raise HTTPException(400, result["error"])
    # Generate P&L plot
    if 'portfolio_history' in result:
        plt.figure(figsize=(10, 5))
        plt.plot(result['portfolio_history'])
        plt.title(f'Portfolio Value Over Time for {ticker}')
        plt.xlabel('Days')
        plt.ylabel('Portfolio Value')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        result['pnl_plot'] = img_base64
        plt.close()
    return result

@app.get("/get_prices/{ticker}/{start_date}/{end_date}")
async def get_prices(ticker: str, start_date: str, end_date: str):
    symbol = ticker.upper() + ".NS" if not ticker.upper().endswith(".NS") else ticker.upper()
    data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)

    if data.empty:
        raise HTTPException(404, "No price data available")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    elif isinstance(data.columns, pd.Index):
        # Ensure columns are strings
        data.columns = data.columns.astype(str)

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