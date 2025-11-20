import pandas as pd
import yfinance as yf
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import numpy as np
from config import MODEL_PATH, NIFTY50_TICKERS
from data_utils import add_features

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

def get_model():
    if not MODEL_PATH.exists():
        return train_nifty50()
    return joblib.load(MODEL_PATH)