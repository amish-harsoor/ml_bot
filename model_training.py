import pandas as pd
import yfinance as yf
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE
from datetime import datetime
import numpy as np
from config import MODEL_PATH, NIFTY50_TICKERS
from data_utils import add_features, get_fundamentals

def train_nifty50():
    print("Training Nifty 50 Professional Model...")
    print("Fetching Nifty and VIX data...")
    nifty_data = yf.download('^NSEI', period="max", auto_adjust=True, progress=False)
    nifty_data = nifty_data[nifty_data.index >= '1995-01-01']
    vix_data = yf.download('^INDIAVIX', period="max", auto_adjust=True, progress=False)
    vix_data = vix_data[vix_data.index >= '1995-01-01']
    dfs = []

    for t in NIFTY50_TICKERS:
        try:
            print(f"Downloading {t} ... ", end="")
            
            # Download with robust error handling
            try:
                data = yf.download(t, period="max", auto_adjust=True, progress=False)
                if data.empty:
                    print("skip (no data)")
                    continue
            except Exception as download_error:
                print(f"download failed: {download_error}")
                continue

            # Robust MultiIndex handling
            if isinstance(data.columns, pd.MultiIndex):
                # Handle MultiIndex where second level might be ticker symbol (possibly as list)
                try:
                    data.columns = data.columns.get_level_values(0)
                except Exception as e:
                    print(f"MultiIndex handling failed: {e}")
                    continue
            elif isinstance(data.columns, pd.Index):
                # Ensure columns are strings
                data.columns = data.columns.astype(str)

            # Verify required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                print("skip (missing columns)")
                continue

            # Filter from 1995
            data = data[data.index >= '1995-01-01']

            if len(data) < 400: # Need more data now for 200 SMA
                print("skip (too short)")
                continue

            # Skip fundamentals completely if any issues - core model doesn't need them
            fundamentals = None

            df = add_features(data, nifty_df=nifty_data, vix_df=vix_data, fundamentals_df=fundamentals)

            if len(df) > 200:
                # Normalize per ticker
                scaler = StandardScaler()
                numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('Target', errors='ignore')
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

                dfs.append(df)
                print(f"OK ({len(df)} rows)")
            else:
                print("skip (features failed)")

        except Exception as e:
            print(f"failed: {e}")
            continue

    if not dfs:
        raise RuntimeError("No data downloaded! Check internet or yfinance version.")

    combined = pd.concat(dfs, ignore_index=False).sort_index()

    # Filter for numeric columns only
    X = combined.select_dtypes(include=["float64", "int64"]).drop(columns=["Target"], errors="ignore")
    y = combined["Target"]

    # Clean infinite values
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]

    if len(X) == 0:
        raise ValueError("All data was filtered out! Check feature generation.")

    # SMOTE to balance classes (down days)
    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X, y)

    # TimeSeriesSplit for validation
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X_sm))
    train_index, test_index = splits[-1]  # Use the last split for final train/test
    X_train, X_test = X_sm.iloc[train_index], X_sm.iloc[test_index]
    y_train, y_test = y_sm.iloc[train_index], y_sm.iloc[test_index]

    # No global scaler since normalized per ticker
    X_train_s = X_train
    X_test_s = X_test

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
        "features": X_sm.columns.tolist(),
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
