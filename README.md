# Nifty 50 AI Stock Predictor

A powerful, fully automatic machine learning bot that predicts tomorrow's price direction (UP/DOWN) for **any Nifty 50 stock** — trained on **all 50 stocks combined** using live data from Yahoo Finance.

- One single XGBoost model trained on 10+ years of data from all Nifty 50 companies
- Real-time predictions in <1 second
- Pure HTML interface — no React, no Node.js, no bloat
- Auto-trains on startup, no manual work needed
- Optional CSV upload to retrain with your own data

Live example: http://127.0.0.1:8000

---

## How It Works

This ML bot uses machine learning to predict whether a Nifty 50 stock's price will go UP or DOWN the next trading day. Here's a simple step-by-step explanation of how it works:

### 1. Training the Model (Done Automatically on First Run)

- **Data Collection**: The bot downloads 8 years of historical daily stock data for all 50 Nifty 50 companies from Yahoo Finance. This includes price data (Open, High, Low, Close, Volume) for stocks like RELIANCE, TCS, HDFCBANK, etc.

- **Feature Engineering**: For each stock's data, the bot creates useful features that help predict price movements:
  - **Price Returns**: Daily percentage change in closing price
  - **Lag Features**: Previous day's returns (up to 30 days back) to capture patterns
  - **Moving Averages**: 10-day and 30-day simple moving averages of closing price
  - **Volatility**: 10-day standard deviation of returns to measure price swings
  - **RSI (Relative Strength Index)**: 14-day momentum indicator (overbought/oversold levels)
  - **Volume Ratio**: Today's volume compared to 20-day average volume
  - **Time Features**: Day of the week and month (markets sometimes have weekly/monthly patterns)

- **Target Creation**: For each day, the bot creates a target: 1 if the price goes UP the next day, 0 if it goes DOWN.

- **Combining Data**: All features and targets from all 50 stocks are combined into one big dataset. This "multi-stock" approach means the model learns patterns that work across different companies.

- **Model Training**: An XGBoost classifier (a powerful machine learning algorithm) is trained on 80% of the data. XGBoost is good at handling complex patterns and avoiding overfitting. The model learns to predict the probability of price going UP based on the features.

- **Saving the Model**: The trained model, along with a scaler (to normalize new data) and the list of features, is saved to a file called `nifty50_model.joblib`.

### 2. Making Predictions

- **User Input**: You enter a stock ticker (like "RELIANCE") in the web interface.

- **Fresh Data Download**: The bot downloads the latest 100 days of data for that specific stock from Yahoo Finance.

- **Feature Calculation**: The same features are calculated on this new data using the `data_utils.py` functions.

- **Prediction**: The latest day's features are fed into the trained XGBoost model. The model outputs a probability (between 0 and 1) of the price going UP tomorrow.
  - If probability > 0.5, prediction is "UP"
  - If probability ≤ 0.5, prediction is "DOWN"

- **Output**: The bot returns the current price, the prediction (UP/DOWN), and the confidence level (the probability as a percentage).

### 3. Web Interface

- The bot runs a simple web server using FastAPI (a Python web framework).
- The frontend is pure HTML, CSS, and JavaScript — no complex frameworks needed.
- You can type a ticker or click quick-select buttons for popular stocks.
- Results are displayed with animations and color coding (green for UP, red for DOWN).

### Technical Details

- **Backend**: Python with FastAPI for the API, yfinance for data, XGBoost for ML, joblib for model storage.
- **Frontend**: HTML/CSS/JS with fetch API for requests.
- **Model**: XGBoost Classifier with 800 trees, max depth 6, trained on ~100,000+ data points from all stocks.
- **Speed**: Training takes 5-10 minutes on first run; predictions are instant (<1 second).
- **Accuracy**: The model aims for better-than-random predictions, but remember: past performance doesn't guarantee future results, and stock predictions are inherently uncertain.

---

### Features

- Trained on **all 50 Nifty stocks** (RELIANCE.NS, TCS.NS, HDFCBANK.NS, etc.)
- Predicts **next-day price movement** with probability
- 100% local — runs on your laptop in 2 seconds
- Beautiful, clean HTML UI
- Supports any Nifty 50 ticker: `RELIANCE`, `TCS`, `INFY`, `HDFCBANK`, etc.
- Model saved as `nifty50_model.joblib` (~20–40 MB)

---

### Demo Screenshot

![Nifty 50 Predictor](https://i.ibb.co.com/9h1Yv1Q/nifty-predictor-demo.png)

---