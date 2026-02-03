import pandas as pd
import pandas_market_calendars as pm
import yfinance as yf
import requests
import json
import datetime as dt
import pytz
import os
import sys


# Set the timezone to Asia/Kolkata
tz = pytz.timezone('Asia/Kolkata')
# Get the current date and time in the specified timezone
now = dt.datetime.now(tz)
# Get the current date in the specified timezone
today = now.date()
# Get the current time in the specified timezone
current_time = now.time()
# Get the current date and time in UTC
now_utc = dt.datetime.now(pytz.utc)
# Get the current date in UTC

symbols = ['^NSEI', '^NSEBANK', 'RELIANCE.NS', 'TATAMOTORS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS','AXISBANK.NS','BAJFINANCE.NS']

def getdata_yf(symbol,period='20y',interval='1d'):
    
    new_dir = f'./Underlying_data_YF'
    os.makedirs(new_dir, exist_ok=True)
    file_path = os.path.join(new_dir, f'{interval}_underlyng_data.json')
    
    data = yf.download(symbol, period=period, interval=interval, progress = False)
    data.to_json(file_path, orient='records')
      
    return None


getdata_yf(symbols)

# loading data for a symbol

def load_underlying_data_yf(symbol, interval='1d'):
   
    file_name = f'{interval}_underlyng_data.json'
    file_path = os.path.join('./Underlying_data_YF', file_name)
    data = pd.read_json(file_path, orient='records')
    print(data.columns)
    
    if symbol in data.columns:
        symbol_data = data[[symbol]]
    else:
        raise ValueError(f"Symbol '{symbol}' not found in the data.")
    
    # Optional: Format the date column if needed
    # data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
    # data['Date'] = data['Date'].dt.strftime('%d-%b-%Y')
    
    return symbol_data
    

df = load_underlying_data_yf('^NSEI')

# --------------------------------------------------------------------------








