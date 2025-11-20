import yfinance as yf
import pandas as pd
from data_utils import add_features

print('Testing single ticker processing...')
t = 'RELIANCE.NS'
try:
    data = yf.download(t, period='max', auto_adjust=True, progress=False)
    print(f'Downloaded: {len(data)} rows')
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        print(f'MultiIndex flattened: {list(data.columns)}')
    
    data = data[data.index >= '1995-01-01']
    print(f'After 1995 filter: {len(data)} rows')
    
    if len(data) < 400:
        print('Too short data')
    else:
        print('Testing add_features...')
        df = add_features(data, fundamentals_df=None)
        print(f'Features added: {len(df)} rows, {len(df.columns)} columns')
        print(f'Sample columns: {list(df.columns[:10])}')
        
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()