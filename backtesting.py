import pandas as pd
import yfinance as yf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from data_utils import add_features
from model_training import get_model

def backtest_ticker(ticker, start_date, end_date):
    """
    Backtest the Nifty50 model on a specific ticker over a date range.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'RELIANCE.NS')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
        dict: Backtesting results with metrics
    """
    pkg = get_model()
    
    # Download data with buffer for feature calculation (need ~250 days for SMA200)
    buffer_days = 250
    start_buffer = pd.to_datetime(start_date) - pd.Timedelta(days=buffer_days)
    
    data = yf.download(ticker, start=start_buffer, end=end_date, auto_adjust=True, progress=False, multi_level_index=False)
    
    if data.empty:
        return {"error": "No data available for the specified ticker and date range"}
    
    # Handle MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Add technical features
    df = add_features(data)
    
    if df.empty:
        return {"error": "Not enough data to generate features (need at least ~200 trading days)"}
    
    # Filter to backtest period
    df_bt = df[(df.index >= start_date) & (df.index <= end_date)]
    
    if df_bt.empty:
        return {"error": f"No data available in the backtest period {start_date} to {end_date}"}
    
    # Prepare features for prediction
    try:
        X = df_bt[pkg["features"]]
    except KeyError as e:
        return {"error": f"Feature mismatch: {e}. Model may need retraining."}
    
    # Scale features
    X_scaled = pkg["scaler"].transform(X)
    
    # Make predictions
    preds = pkg["model"].predict(X_scaled)
    actuals = df_bt["Target"].values
    
    # Calculate metrics
    acc = accuracy_score(actuals, preds)
    prec = precision_score(actuals, preds, zero_division=0)
    rec = recall_score(actuals, preds, zero_division=0)
    f1 = f1_score(actuals, preds, zero_division=0)
    cm = confusion_matrix(actuals, preds).tolist()
    
    # Calculate additional stats
    total_predictions = len(preds)
    correct_predictions = int(acc * total_predictions)
    up_predictions = sum(preds)
    down_predictions = total_predictions - up_predictions
    actual_ups = sum(actuals)
    actual_downs = total_predictions - actual_ups
    
    return {
        "ticker": ticker,
        "period": f"{start_date} to {end_date}",
        "samples": total_predictions,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        "predictions": preds.tolist(),  # Add predictions array for simulation
        "actuals": actuals.tolist(),    # Add actuals for reference
        "dates": df_bt.index.strftime('%Y-%m-%d').tolist(),  # Add dates
        "summary": {
            "correct_predictions": correct_predictions,
            "predicted_up": int(up_predictions),
            "predicted_down": int(down_predictions),
            "actual_up": int(actual_ups),
            "actual_down": int(actual_downs)
        }
    }

def backtest_multiple_tickers(tickers, start_date, end_date):
    """
    Backtest the model on multiple tickers.
    
    Args:
        tickers (list): List of ticker symbols
        start_date (str): Start date
        end_date (str): End date
    
    Returns:
        dict: Results for each ticker
    """
    results = {}
    for ticker in tickers:
        print(f"Backtesting {ticker}...")
        results[ticker] = backtest_ticker(ticker, start_date, end_date)
    return results