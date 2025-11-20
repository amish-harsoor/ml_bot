import pandas as pd
import yfinance as yf
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime
import numpy as np
from config import MODEL_PATH, NIFTY50_TICKERS
from data_utils import add_features

def train_nifty50():
    print("Training Nifty 50 Professional Model...")
    dfs = []

    for t in NIFTY50_TICKERS:
        try:
            print(f"Downloading {t} ... ", end="")
            ticker = yf.Ticker(t)
            # Grab maximum available history from inception
            data = yf.download(t, period="max", auto_adjust=True, progress=False, multi_level_index=False)
            
            # Robust MultiIndex handling
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if len(data) < 400: # Need more data now for 200 SMA
                print("skip (too short)")
                continue
                
            df = add_features(data)
            
            if len(df) > 200:
                dfs.append(df)
                print(f"OK ({len(df)} rows)")
            else:
                print("skip (features failed)")
                
        except Exception as e:
            print(f"failed: {e}")

    if not dfs:
        raise RuntimeError("No data downloaded! Check internet or yfinance version.")

    combined = pd.concat(dfs, ignore_index=True)
    
    # Filter for numeric columns only
    X = combined.select_dtypes(include=["float64", "int64"]).drop(columns=["Target"], errors="ignore")
    y = combined["Target"]

    # Clean infinite values
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]

    if len(X) == 0:
        raise ValueError("All data was filtered out! Check feature generation.")

    # Time-series safe split (no shuffling)
    split = int(0.85 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Enhanced XGBoost Hyperparameters
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=8,               # Deeper trees for complex features
        learning_rate=0.01,        # Slower learning for better generalization
        subsample=0.7,             # Prevent overfitting
        colsample_bytree=0.7,      # Feature sampling
        gamma=0.1,                 # Min loss reduction
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        early_stopping_rounds=50,  # Stop if no improvement
        verbosity=0
    )

    print(f"Fitting model on {len(X_train)} samples...")
    model.fit(
        X_train_s, y_train,
        eval_set=[(X_test_s, y_test)],
        verbose=100
    )

    # Calibrate probabilities with Platt scaling
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    calibrated_model.fit(X_train_s, y_train)

    package = {
        "model": model,
        "calibrated_model": calibrated_model,
        "scaler": scaler,
        "features": X.columns.tolist(),
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

    MODEL_PATH.unlink(missing_ok=True)
    joblib.dump(package, MODEL_PATH)
    print(f"MODEL READY! {len(combined):,} samples -> {MODEL_PATH}")
    return package

def get_model():
    if not MODEL_PATH.exists():
        return train_nifty50()
    return joblib.load(MODEL_PATH)
