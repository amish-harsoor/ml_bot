import warnings
warnings.filterwarnings('ignore')

from model_training import train_nifty50
from config import NIFTY50_TICKERS

print(f'Testing with first 3 tickers: {NIFTY50_TICKERS[:3]}')

# Temporarily modify the tickers for testing
import config
config.NIFTY50_TICKERS = NIFTY50_TICKERS[:3]

try:
    result = train_nifty50()
    print('SUCCESS: Model training completed!')
    print(f'Model features: {len(result["features"])}')
except Exception as e:
    print(f'FAILED: {e}')
    import traceback
    traceback.print_exc()