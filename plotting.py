import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime as dt_time
from data_download_vbt import get_dates_from_most_active_files



  
def plot_garch_vs_rsi(symbols, data_dir='./Engineered_data'):
    """
    Reads feature files for each symbol and creates a scatter plot of GARCH Vol Percentile vs RSI.
    Highlights NIFTY and BANKNIFTY markers in different colors.
    """
    dt = dt_time.now().strftime('%Y-%m-%d')
    dt1 = get_dates_from_most_active_files()[-1]
    image_dir='./Images'
    if not os.path.exists(image_dir):
        print(f"Creating directory: {image_dir}")   
        os.makedirs(image_dir)
    
    records = []
    for symbol in symbols:
        file_path = os.path.join(data_dir, f"{symbol}_1d_features.json")
        if not os.path.exists(file_path):
            print(f"File not found for {symbol}: {file_path}")
            continue
        df = pd.read_json(file_path, orient='records', lines=True)
        if 'garch_vol_percentile' not in df.columns or 'RSI' not in df.columns:
            print(f"Required columns not found in {file_path}")
            continue
        last_row = df.iloc[-1]
        garch_val = last_row['garch_vol_percentile']
        rsi_val = last_row['RSI']

        if pd.isna(garch_val) or pd.isna(rsi_val):
            print(f"Skipping {symbol} due to NaN values.")
            continue
        records.append({
            'symbol': symbol,
            'garch_vol_percentile': int(garch_val),
            'RSI': int(rsi_val)
        })  
    df1 = pd.DataFrame(records)
    print(df1)

    plt.figure(figsize=(8, 6))
    # Plot all except NIFTY and BANKNIFTY
    mask_nifty = df1['symbol'].str.upper().isin(['NIFTY', '^NSEI'])
    mask_banknifty = df1['symbol'].str.upper().isin(['BANKNIFTY', '^NSEBANK'])
    mask_others = ~(mask_nifty | mask_banknifty)

    plt.scatter(df1.loc[mask_others, 'RSI'], df1.loc[mask_others, 'garch_vol_percentile'],
                alpha=0.5, label='Others', color='blue')
    plt.scatter(df1.loc[mask_nifty, 'RSI'], df1.loc[mask_nifty, 'garch_vol_percentile'],
                alpha=0.9, label='NIFTY', color='red', marker='o', s=80, edgecolor='black')
    plt.scatter(df1.loc[mask_banknifty, 'RSI'], df1.loc[mask_banknifty, 'garch_vol_percentile'],
                alpha=0.9, label='BANKNIFTY', color='green', marker='s', s=80, edgecolor='black')

    # Add symbol labels to each marker
    for _, row in df1.iterrows():
        plt.text(row['RSI'], row['garch_vol_percentile'], row['symbol'], fontsize=9, ha='right', va='bottom')

    plt.ylabel('GARCH Vol Percentile')
    plt.xlabel('RSI')
    plt.title('GARCH Vol Percentile vs RSI')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(image_dir, f'garch_vs_rsi_{dt}.png')
    plt.savefig(save_path)
    print(f"Saving plot to {save_path}")
    plt.close()
    return None



import json

def get_atm_iv_from_optionchain(symbol, options_expiry=None):
    """
    Reads the ATM option chain JSON for a symbol and returns call_iv and put_iv for a given expiry date.
    """
    data_dir='./ATM_OptionchainJSON'
    file_path = os.path.join(data_dir, f"{symbol}_ATM_OptionChain.json")
    if not os.path.exists(file_path):
        print(f"File not found for {symbol}: {file_path}")
        return None, None

    with open(file_path, 'r') as f:
        data = json.load(f)

    # The structure may vary, but typically expiryDate is a key in the data or in a list of records
    # Find the record for the given expiry_date
    for record in reversed(data):
        if record.get('Expiry') == options_expiry:
            call_iv = record.get('call_iv')
            put_iv = record.get('put_iv')
            return call_iv, put_iv

    print(f"No data found for expiry date {options_expiry} in {file_path}")
    return None, None



def plot_garch_vs_avg_iv(symbols, options_expiry=None):
    """
    Scatter plot of GARCH Vol vs average of call_iv and put_iv for each symbol.
    Highlights NIFTY and BANKNIFTY markers in different colors.
    """
    data_dir='./Engineered_data'
    records = []
    for symbol in symbols:
        file_path = os.path.join(data_dir, f"{symbol}_1d_features.json")
        if not os.path.exists(file_path):
            print(f"File not found for {symbol}: {file_path}")
            continue
        df = pd.read_json(file_path, orient='records', lines=True)
        if (
            'garch_vol' not in df.columns 
        ):
            print(f"Required columns not found in {file_path}")
            continue
        last_row = df.iloc[-1]
        call_iv, put_iv = get_atm_iv_from_optionchain(symbol,options_expiry)
        
        avg_iv = call_iv
        records.append({
            'symbol': symbol,
            'garch_vol': last_row['garch_vol'],
            'avg_iv': avg_iv
        })
    df1 = pd.DataFrame(records)

    plt.figure(figsize=(8, 6))
    mask_nifty = df1['symbol'].str.upper().isin(['NIFTY', '^NSEI'])
    mask_banknifty = df1['symbol'].str.upper().isin(['BANKNIFTY', '^NSEBANK'])
    mask_others = ~(mask_nifty | mask_banknifty)

    plt.scatter(df1.loc[mask_others, 'avg_iv'], df1.loc[mask_others, 'garch_vol'],
                alpha=0.5, label='Others', color='blue')
    plt.scatter(df1.loc[mask_nifty, 'avg_iv'], df1.loc[mask_nifty, 'garch_vol'],
                alpha=0.9, label='NIFTY', color='red', marker='o', s=80, edgecolor='black')
    plt.scatter(df1.loc[mask_banknifty, 'avg_iv'], df1.loc[mask_banknifty, 'garch_vol'],
                alpha=0.9, label='BANKNIFTY', color='green', marker='s', s=80, edgecolor='black')

    for _, row in df1.iterrows():
        plt.text(row['avg_iv'], row['garch_vol'], row['symbol'], fontsize=9, ha='right', va='bottom')

    plt.ylabel('GARCH Vol')
    plt.xlabel('Average IV (call_iv & put_iv)')
    plt.title('GARCH Vol vs Average IV')
    plt.legend()
    plt.grid(True)
    plt.savefig('garch_vs_avg_iv_scatter_plot.png')
    plt.close()
    return None

# plot_garch_vs_avg_iv("NIFTY")