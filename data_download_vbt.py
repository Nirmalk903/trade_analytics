import vectorbt as vbt
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime as dt_time
import time
import re

RENAME_DICT = {
    'Open': 'Open',
    'High': 'High',
    'Low': 'Low',
    'Close': 'Close',
    'Volume': 'Volume'
}

def adjust_for_corporate_actions(data):
    """
    Adjust OHLC prices for corporate actions (splits, bonuses) using Adj_Close.
    Yahoo Finance provides Adj_Close which accounts for splits and dividends.
    We use the ratio between Adj_Close and Close to adjust all historical prices.
    
    Args:
        data: DataFrame with columns Date, Open, High, Low, Close, Adj_Close, Volume
    
    Returns:
        DataFrame with adjusted OHLC prices
    """
    if 'Adj_Close' not in data.columns or data.empty:
        print("[WARNING] Adj_Close column not found or data is empty. Skipping corporate action adjustment.")
        return data
    
    # Calculate adjustment ratio (Adj_Close / Close)
    # This ratio reflects all corporate actions (splits, bonuses)
    data['Adjustment_Ratio'] = data['Adj_Close'] / data['Close']
    
    # Replace NaN or infinite ratios with 1 (no adjustment)
    data['Adjustment_Ratio'] = data['Adjustment_Ratio'].replace([np.inf, -np.inf], 1.0).fillna(1.0)
    
    # Apply adjustment ratio to OHLC prices
    data['Open'] = data['Open'] * data['Adjustment_Ratio']
    data['High'] = data['High'] * data['Adjustment_Ratio']
    data['Low'] = data['Low'] * data['Adjustment_Ratio']
    data['Close'] = data['Adj_Close']  # Use adjusted close as the close price
    
    # Volume should be adjusted inversely for splits (multiply by inverse of split ratio)
    # For example, 1:2 split means double the shares, so volume should be doubled
    # However, Yahoo Finance typically keeps volume unadjusted, so we'll leave it as is
    
    # Drop temporary columns
    data = data.drop(columns=['Adjustment_Ratio', 'Adj_Close'], errors='ignore')
    
    print(f"[INFO] Applied corporate action adjustments to OHLC prices.")
    return data

def get_symbols(dt, top_n=17):
    # Always use the latest LA-MOST-ACTIVE-UNDERLYING-*.csv file
    from glob import glob
    import re
    most_active_dir = './Most_Active_Underlying'
    pattern = os.path.join(most_active_dir, 'LA-MOST-ACTIVE-UNDERLYING-*.csv')
    files = glob(pattern)
    if not files:
        print(f"No LA-MOST-ACTIVE-UNDERLYING-*.csv files found in {most_active_dir}")
        return [], []
    # Extract dates and pick the latest
    date_re = re.compile(r'LA-MOST-ACTIVE-UNDERLYING-(\d{2}-[A-Za-z]{3}-\d{4})\.csv$')
    dated_files = []
    for f in files:
        m = date_re.search(f)
        if m:
            try:
                file_date = dt_time.strptime(m.group(1), '%d-%b-%Y')
                dated_files.append((file_date, f))
            except Exception as e:
                print(f"[WARNING] Could not parse date from {f}: {e}")
    if not dated_files:
        print(f"No valid dated LA-MOST-ACTIVE-UNDERLYING files found.")
        return [], []
    latest_file = max(dated_files, key=lambda x: x[0])[1]
    print(f"[INFO] Using most recent file: {latest_file}")

    nifty_fity_path = os.path.join('./Nifty_Fifty', "MW-NIFTY-50.csv")
    if os.path.exists(nifty_fity_path):
        nifty_fifty = pd.read_csv(nifty_fity_path)
        nifty_fifty.columns = nifty_fifty.columns.str.strip()
        if 'SYMBOL' in nifty_fifty.columns:
            nifty_symbols = nifty_fifty['SYMBOL'].tolist()
        else:
            for alt in ('Symbol', 'SYMBOL ', 'SYMBOL\n'):
                if alt in nifty_fifty.columns:
                    nifty_symbols = nifty_fifty[alt].tolist()
                    break
            else:
                nifty_symbols = nifty_fifty.iloc[:, 0].astype(str).tolist()
        liquid_symbols = nifty_symbols + ['NIFTY', 'BANKNIFTY']
    else:
        print(f"File {nifty_fity_path} does not exist.")
        return [], []

    most_active = pd.read_csv(latest_file)
    most_active.columns = most_active.columns.str.strip()
    # Normalize column names for both old and new formats
    col_map = {}
    for col in most_active.columns:
        c = col.strip().lower().replace(' ', '').replace('(', '').replace(')', '').replace('₹', '').replace('-', '').replace('.', '').replace('/', '').replace('_', '')
        if c.startswith('symbol'):
            col_map[col] = 'Symbol'
        elif 'volume' in c and 'future' in c:
            col_map[col] = 'Volume (Contracts) - Futures'
        elif 'volume' in c and 'option' in c:
            col_map[col] = 'Volume (Contracts) - Options'
        elif 'volume' in c and 'total' in c:
            col_map[col] = 'Volume (Contracts) - Total'
        elif 'value' in c and 'future' in c:
            col_map[col] = 'Value (₹ Lakhs) - Futures'
        elif 'value' in c and 'option' in c:
            col_map[col] = 'Value (₹ Lakhs) - Options (Premium)'
        elif 'value' in c and 'total' in c:
            col_map[col] = 'Value (₹ Lakhs) - Total'
        elif 'openinterest' in c:
            col_map[col] = 'Open Interest (Contracts)'
        elif c == 'underlying':
            col_map[col] = 'Underlying'
    most_active.rename(columns=col_map, inplace=True)
    if 'Symbol' not in most_active.columns:
        raise KeyError(f"'Symbol' column not found in {latest_file}. Columns: {most_active.columns.tolist()}")
    most_active = most_active[~most_active['Symbol'].isin([
        'MIDCPNIFTY', 'FINNIFTY', 'NIFTYIT', 'NIFTYNXT50', 'NIFTYPSUBANK',
        'NIFTYINFRA', 'NIFTYMETAL', 'NIFTYPHARMA', 'NIFTYMEDIA',
        'NIFTYAUTO', 'NIFTYCONSUMPTION', 'NIFTYENERGY', 'NIFTYFMCG',
        'NIFTYHEALTHCARE'
    ])].reset_index(drop=True)
    most_active = most_active[most_active['Symbol'].isin(liquid_symbols)].reset_index(drop=True)
    most_active['YF_Symbol'] = most_active['Symbol'].apply(
        lambda x: '^NSEI' if x == 'NIFTY' else '^NSEBANK' if x == 'BANKNIFTY' else f'{x}.NS'
    )
    # Sort by value if present, else by volume
    sort_col = 'Value (₹ Lakhs) - Total' if 'Value (₹ Lakhs) - Total' in most_active.columns else (
        'totTurnover' if 'totTurnover' in most_active.columns else None)
    if sort_col:
        most_active = most_active.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
    most_active = most_active.head(top_n)

    symbols = most_active['Symbol'].tolist()
    yf_symbols = most_active['YF_Symbol'].tolist()
    return symbols, yf_symbols

def getdata_vbt(symbols, period='20y', interval='1d'):
    for symbol in symbols:
        print(f'Loading data for {symbol}')
        yf_symbol = "^NSEI" if symbol == 'NIFTY' else "^NSEBANK" if symbol == 'BANKNIFTY' else f'{symbol}.NS'
        new_dir = f'./Underlying_data_vbt'
        os.makedirs(new_dir, exist_ok=True)
        file_name = f'{symbol}_{interval}.csv'
        file_path = os.path.join(new_dir, file_name)

        data = vbt.YFData.download(yf_symbol, period=period, interval=interval).data[yf_symbol]
        data = data.rename(columns={'Adj Close': 'Adj_Close', **RENAME_DICT})
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'], utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        
        # Apply corporate action adjustments (splits, bonuses) using Adj_Close
        data = adjust_for_corporate_actions(data)
        
        data.to_csv(file_path, index=False)
        print(f"Data for {symbol} saved successfully.")
    return None

def get_underlying_data_vbt(symbols, period='20y', interval='1d'):
    for idx, symbol in enumerate(symbols):
        print(f'Loading data for {int(idx)+1}: {symbol}')
        yf_symbol = '^NSEI' if symbol == 'NIFTY' else "^NSEBANK" if symbol == 'BANKNIFTY' else f'{symbol}.NS'
        new_dir = f'./Underlying_data_vbt'
        os.makedirs(new_dir, exist_ok=True)
        file_name = f'{symbol}_{interval}.csv'
        file_path = os.path.join(new_dir, file_name)

        # Check if file exists and get last date
        if os.path.exists(file_path):
            existing_data = pd.read_csv(file_path, parse_dates=['Date'])
            if not existing_data.empty:
                # Re-download full history to ensure corporate actions are properly adjusted
                print(f"Re-downloading full history for {symbol} to adjust for corporate actions...")
                data = vbt.YFData.download(yf_symbol, period=period, interval=interval).data[yf_symbol]
                data = data.rename(columns={'Adj Close': 'Adj_Close', **RENAME_DICT})
                data.reset_index(inplace=True)
                data['Date'] = pd.to_datetime(data['Date'], utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
                
                # Apply corporate action adjustments
                data = adjust_for_corporate_actions(data)
                
                data.to_csv(file_path, index=False)
                print(f"Data for {symbol} updated with corporate action adjustments.")
            else:
                # If file exists but is empty, download all data
                data = vbt.YFData.download(yf_symbol, period=period, interval=interval).data[yf_symbol]
                data = data.rename(columns={'Adj Close': 'Adj_Close', **RENAME_DICT})
                data.reset_index(inplace=True)
                data['Date'] = pd.to_datetime(data['Date'], utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
                
                # Apply corporate action adjustments
                data = adjust_for_corporate_actions(data)
                
                data.to_csv(file_path, index=False)
                print(f"Data for {symbol} saved successfully.")
        else:
            # If file does not exist, download all data
            data = vbt.YFData.download(yf_symbol, period=period, interval=interval).data[yf_symbol]
            data = data.rename(columns={'Adj Close': 'Adj_Close', **RENAME_DICT})
            data.reset_index(inplace=True)
            data['Date'] = pd.to_datetime(data['Date'], utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
            
            # Apply corporate action adjustments
            data = adjust_for_corporate_actions(data)
            
            data.to_csv(file_path, index=False)
            print(f"Data for {symbol} saved successfully.")

    return None

def get_dates_from_most_active_files(folder='./Most_Active_Underlying'):
    # Regex to extract date from filenames like LA-MOST-ACTIVE-UNDERLYING-23-May-2024.csv
    date_pattern = re.compile(r'LA-MOST-ACTIVE-UNDERLYING-(\d{2}-[A-Za-z]{3}-\d{4})\.csv')
    dates = []
    for fname in os.listdir(folder):
        match = date_pattern.match(fname)
        if match:
            dates.append(match.group(1))
    return pd.to_datetime(sorted([dt_time.strptime(d, "%d-%b-%Y") for d in dates]))



