import pandas as pd
import yfinance as yf
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef
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

    # Download Nifty for regime classification
    nifty = yf.download('^NSEI', start=start_buffer, end=end_date, progress=False, auto_adjust=True, multi_level_index=False)
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    nifty_bt = nifty[(nifty.index >= start_date) & (nifty.index <= end_date)]
    nifty_bt['Quarter'] = nifty_bt.index.to_period('Q')
    q_returns = nifty_bt.groupby('Quarter')['Close'].agg(lambda x: x.iloc[-1]/x.iloc[0] - 1 if len(x)>1 else 0)
    regimes = q_returns > 0  # bull if positive return

    # Assign regimes
    df_bt = df_bt.copy()
    df_bt['Quarter'] = df_bt.index.to_period('Q')
    df_bt['Regime'] = df_bt['Quarter'].map(lambda q: 'bull' if regimes.get(q, False) else 'correction')

    # Prepare features for prediction
    try:
        X = df_bt[pkg["features"]]
    except KeyError as e:
        return {"error": f"Feature mismatch: {e}. Model may need retraining."}

    # Scale features
    X_scaled = pkg["scaler"].transform(X)

    # Make calibrated predictions
    preds_proba = pkg["calibrated_model"].predict_proba(X_scaled)[:, 1]
    preds = (preds_proba > 0.5).astype(int)
    actuals = df_bt["Target"].values

    # Apply filters
    # VIX filter: if VIX > 20, set to flat (0)
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False, auto_adjust=True, multi_level_index=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix = vix.reindex(df_bt.index).ffill()
    for i in range(len(preds)):
        if vix['Close'].iloc[i] > 20:
            preds[i] = 0

    # Volume confirmation: if pred=1 but volume <= SMA20, set to 0
    for i in range(len(preds)):
        if preds[i] == 1 and df_bt['Volume'].iloc[i] <= df_bt['Vol_SMA_20'].iloc[i]:
            preds[i] = 0

    # Calculate win rates by regime
    bull_mask = df_bt['Regime'] == 'bull'
    corr_mask = df_bt['Regime'] == 'correction'
    bull_acc = accuracy_score(actuals[bull_mask], preds[bull_mask]) if bull_mask.sum() > 0 else None
    corr_acc = accuracy_score(actuals[corr_mask], preds[corr_mask]) if corr_mask.sum() > 0 else None
    regime_win_rate = {
        "bull": round(bull_acc, 4) if bull_acc is not None else None,
        "correction": round(corr_acc, 4) if corr_acc is not None else None
    }
    
    # Calculate metrics
    acc = accuracy_score(actuals, preds)
    prec = precision_score(actuals, preds, zero_division=0)
    rec = recall_score(actuals, preds, zero_division=0)
    f1 = f1_score(actuals, preds, zero_division=0)
    cm = confusion_matrix(actuals, preds).tolist()
    mcc = matthews_corrcoef(actuals, preds)

    # Simulate cumulative return with 0.1% transaction costs
    df_bt = df_bt.copy()
    df_bt['Next_Return'] = df_bt['Close'].pct_change().shift(-1)
    position = 0  # 0: cash, 1: long
    portfolio_value = 1.0
    trade_cost = 0.001
    daily_returns = []
    portfolio_history = [1.0]
    for i in range(len(preds) - 1):
        pred = preds[i]
        ret = df_bt['Next_Return'].iloc[i]
        if not np.isnan(ret):
            if pred == 1 and position == 0:
                portfolio_value *= (1 - trade_cost)
                position = 1
            elif pred == 0 and position == 1:
                portfolio_value *= (1 - trade_cost)
                position = 0
            if position == 1:
                portfolio_value *= (1 + ret)
                daily_returns.append(ret)
            else:
                daily_returns.append(0.0)
            portfolio_history.append(portfolio_value)
    cumulative_return = portfolio_value - 1.0

    # Sharpe ratio (annualized, rf=0)
    if daily_returns:
        mean_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns)
        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0

        # Monte Carlo simulations
        mc_cum_returns = []
        mc_max_drawdowns = []
        for _ in range(1000):
            sampled_returns = np.random.choice(daily_returns, size=len(daily_returns), replace=True)
            cum_ret = np.prod(1 + np.array(sampled_returns)) - 1
            mc_cum_returns.append(cum_ret)
            cum_vals = np.cumprod(1 + np.array(sampled_returns))
            peak = np.maximum.accumulate(cum_vals)
            drawdown = (cum_vals - peak) / peak
            max_dd = drawdown.min() if len(drawdown) > 0 else 0
            mc_max_drawdowns.append(max_dd)
        mc_mean_return = np.mean(mc_cum_returns)
        mc_std_return = np.std(mc_cum_returns)
        mc_max_dd = np.mean(mc_max_drawdowns)
    else:
        sharpe = 0
        mc_mean_return = mc_std_return = mc_max_dd = 0

    
    # Calculate additional stats
    total_predictions = len(preds)
    correct_predictions = int(acc * total_predictions)
    up_predictions = sum(preds)
    down_predictions = total_predictions - up_predictions
    actual_ups = sum(actuals)
    actual_downs = total_predictions - actual_ups
    
    # Log results to CSV
    results_row = {
        'ticker': ticker,
        'start_date': start_date,
        'end_date': end_date,
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'mcc': round(mcc, 4),
        'cumulative_return': round(cumulative_return, 4),
        'sharpe_ratio': round(sharpe, 4),
        'mc_mean_return': round(mc_mean_return, 4),
        'mc_std_return': round(mc_std_return, 4),
        'mc_avg_max_drawdown': round(mc_max_dd, 4),
        'bull_win_rate': regime_win_rate.get('bull'),
        'correction_win_rate': regime_win_rate.get('correction')
    }
    results_df = pd.DataFrame([results_row])
    if os.path.exists('backtest_results.csv'):
        existing = pd.read_csv('backtest_results.csv')
        results_df = pd.concat([existing, results_df], ignore_index=True)
    results_df.to_csv('backtest_results.csv', index=False)

    return {
        "ticker": ticker,
        "period": f"{start_date} to {end_date}",
        "samples": total_predictions,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "mcc": round(mcc, 4),
        "cumulative_return": round(cumulative_return, 4),
        "sharpe_ratio": round(sharpe, 4),
        "mc_mean_return": round(mc_mean_return, 4),
        "mc_std_return": round(mc_std_return, 4),
        "mc_avg_max_drawdown": round(mc_max_dd, 4),
        "regime_win_rate": regime_win_rate,
        "portfolio_history": [round(v, 4) for v in portfolio_history],
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