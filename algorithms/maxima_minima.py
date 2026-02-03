import os
import pandas as pd

def load_all_features(symbols, data_dir="Engineered_data"):
    """
    Load feature data for all symbols and concatenate into a single DataFrame.
    """
    dfs = []
    for symbol in symbols:
        file_path = os.path.join(data_dir, f"{symbol}_1d_features.json")
        if os.path.exists(file_path):
            df = pd.read_json(file_path, orient='records', lines=True)
            df['symbol'] = symbol
            dfs.append(df)
    if dfs:
        all_features = pd.concat(dfs, ignore_index=True)
        return all_features
    else:
        return pd.DataFrame()

def filter_low_rsi_percentile(df, rsi_col='RSI_percentile', threshold=35):
    """
    Filter stocks with daily RSI percentile < threshold.
    Returns columns: symbol, volatility, volatility percentile, daily RSI percentile, monthly RSI percentile.
    """
    filtered = df[df[rsi_col] < threshold][
        ['symbol', 'Date', 'garch_vol', 'vol_percentile', 'RSI_percentile', 'RSI_percentile_monthly']
    ]
    return filtered

# Example usage:


import pandas as pd
import numpy as np
from Options_Utility import highlight_rows  # Placeholder for future implementation
from black_scholes_functions import *
import os
from tqdm import tqdm
from get_options_data import fetch_and_save_options_chain, enrich_option_chain,load_enriched_option_chain, load_atm_chain
from data_download_vbt import getdata_vbt, get_underlying_data_vbt, get_symbols,  get_dates_from_most_active_files
from feature_engineering import add_features
import matplotlib.pyplot as plt
from plotting import plot_garch_vs_rsi, plot_garch_vs_avg_iv
# from market_calendar import  stock_earnings_calendar
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


symbols = get_symbols(get_dates_from_most_active_files()[-1],top_n=17)[0]




# symbols = ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS', ...]  # your symbol list
all_features_df = load_all_features(symbols)
filtered_df = filter_low_rsi_percentile(all_features_df)

filtered_df.head()