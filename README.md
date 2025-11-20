# Nifty 50 AI Stock Predictor

A powerful, fully automatic machine learning bot that predicts tomorrow's price direction (UP/DOWN) for **any Nifty 50 stock** — trained on **all 50 stocks combined** using live data from Yahoo Finance.

- One single XGBoost model trained on 10+ years of data from all Nifty 50 companies  
- Real-time predictions in <1 second  
- Pure HTML interface — no React, no Node.js, no bloat  
- Auto-trains on startup, no manual work needed  
- Optional CSV upload to retrain with your own data  

Live example: http://127.0.0.1:8000

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